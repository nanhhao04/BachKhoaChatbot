import os
import logging
import gc
import torch
import asyncio
import threading
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_community.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
from pymilvus import connections, utility, Collection
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.base import LLM

# Advanced retriever imports
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers import BM25Retriever

# Fix imports
from app.utils import extract_source_from_text
from app.schema import Citation

# Setup
logger = logging.getLogger(__name__)
load_dotenv()

# Singleton instance
_chatbot_instance = None
_init_lock = threading.Lock()


# Get collection count from Milvus
def get_collection_count(collection_name: str) -> int:
    collection = Collection(collection_name)
    collection.load()
    return collection.num_entities


class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put_nowait(token)

    async def get_tokens(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield f"data: {token}\n\n"


# Wrapper for Hugging Face pipeline
class HuggingFaceLLMWrapper(LLM):
    def __init__(self, hf_pipeline):
        super().__init__()
        self.pipeline = hf_pipeline

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output = self.pipeline(prompt, return_full_text=False)[0]['generated_text']
        return output.strip()

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"


# Contextual retriever with conversation history
class ContextualRetriever:
    def __init__(self, base_retriever, memory, llm):
        self.base_retriever = base_retriever
        self.memory = memory
        self.llm = llm

    def get_relevant_documents(self, query: str):
        context = self._get_conversation_context()
        expanded_query = self._expand_query_with_context(query, context)
        docs = self.base_retriever.get_relevant_documents(expanded_query)
        return self._rerank_documents(docs, query, context)

    def _get_conversation_context(self):
        messages = self.memory.chat_memory.messages
        if not messages:
            return ""

        recent_messages = messages[-4:]
        context_parts = []

        for msg in recent_messages:
            if hasattr(msg, 'content'):
                content = msg.content.strip()
                if len(content) > 10:
                    context_parts.append(content)

        return " ".join(context_parts[-3:])

    def _expand_query_with_context(self, query: str, context: str):
        if not context:
            return query

        pronouns = ["này", "đó", "thế", "vậy", "như vậy", "gì", "nào"]
        if any(pronoun in query.lower() for pronoun in pronouns):
            return f"{context}. {query}"

        return query

    def _rerank_documents(self, docs, original_query, context):
        if not context:
            return docs

        scored_docs = []
        context_keywords = set(context.lower().split())
        query_keywords = set(original_query.lower().split())
        all_keywords = context_keywords.union(query_keywords)

        for doc in docs:
            doc_text = doc.page_content.lower()
            matches = sum(1 for kw in all_keywords if kw in doc_text)
            score = matches / len(all_keywords) if all_keywords else 0
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]


# Adaptive retriever that changes strategy based on query type
class AdaptiveRetriever:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

        self.retrievers = {
            "specific": vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 20, "score_threshold": 0.6}
            ),
            "general": vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 30, "fetch_k": 50, "lambda_mult": 0.8}
            ),
            "comparative": vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 40}
            )
        }

    def get_relevant_documents(self, query: str):
        query_type = self._classify_query(query)
        retriever = self.retrievers.get(query_type, self.retrievers["general"])
        logger.info(f"Using {query_type} retriever for query: {query[:50]}...")
        return retriever.get_relevant_documents(query)

    def _classify_query(self, query: str) -> str:
        query_lower = query.lower()

        specific_indicators = [
            "bao nhiêu", "điểm", "gpa", "tín chỉ", "học phí",
            "điều kiện", "yêu cầu", "quy định", "thủ tục"
        ]
        if any(indicator in query_lower for indicator in specific_indicators):
            return "specific"

        comparative_indicators = [
            "khác nhau", "so sánh", "giống", "khác", "nào tốt hơn",
            "ưu nhược điểm", "phân biệt"
        ]
        if any(indicator in query_lower for indicator in comparative_indicators):
            return "comparative"

        return "general"


class ChatbotRAG:
    def __init__(self):
        logger.info("Initializing ChatbotRAG...")
        self.initialized = False

        self.collection_name = os.getenv("MILVUS_COLLECTION", "student_support_chatbot")
        self._connect_milvus()
        self._verify_collection()

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/e5-small-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self._setup_vectorstore()
        self._setup_enhanced_chain()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        self.initialized = True
        logger.info("ChatbotRAG initialization complete")

    # Connect to Milvus server
    def _connect_milvus(self):
        hosts = [os.getenv("MILVUS_HOST", "host.docker.internal"), "localhost", "127.0.0.1"]
        port = int(os.getenv("MILVUS_PORT", "19530"))

        for host in hosts:
            if connections.has_connection("default"):
                connections.disconnect("default")

            connections.connect(alias="default", host=host, port=port, timeout=5.0)

            if connections.has_connection("default"):
                logger.info(f"Connected to Milvus at {host}:{port}")
                self.connection_args = {"host": host, "port": port}
                return

    # Verify collection exists and has data
    def _verify_collection(self):
        docs_count = get_collection_count(self.collection_name)
        logger.info(f"Using existing collection '{self.collection_name}' with {docs_count} documents")

    # Setup vector store
    def _setup_vectorstore(self):
        self.vectorstore = Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args=self.connection_args
        )
        logger.info(f"Vectorstore loaded successfully")

    # Create enhanced retriever with multiple strategies
    def _create_enhanced_retriever(self):
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 30}
        )

        mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 30,
                "fetch_k": 50,
                "lambda_mult": 0.7
            }
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[base_retriever, mmr_retriever],
            weights=[0.6, 0.4]
        )

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever,
            llm=self.llm,
            include_original=True
        )
        logger.info("Using Multi-Query + Ensemble Retriever")
        return multi_query_retriever

    # Create contextual retriever
    def _create_contextual_retriever(self):
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 35}
        )
        return ContextualRetriever(base_retriever, self.memory, self.llm)

    # Create hybrid retriever combining semantic and keyword search
    def _create_hybrid_retriever(self):
        semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 30}
        )

        all_docs = self._get_all_documents_from_collection()
        if all_docs:
            keyword_retriever = BM25Retriever.from_documents(all_docs)
            keyword_retriever.k = 30

            hybrid_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, keyword_retriever],
                weights=[0.7, 0.3]
            )
            logger.info("Using Hybrid Retriever (Semantic + BM25)")
            return hybrid_retriever

        return semantic_retriever

    # Create adaptive retriever
    def _create_adaptive_retriever(self):
        return AdaptiveRetriever(self.vectorstore, self.llm)

    # Get all documents from collection
    def _get_all_documents_from_collection(self):
        collection = Collection(self.collection_name)
        collection.load()

        results = collection.query(
            expr="",
            output_fields=["text", "source"],
            limit=5000
        )

        documents = []
        for result in results:
            doc = Document(
                page_content=result.get("text", ""),
                metadata={"source": result.get("source", "")}
            )
            documents.append(doc)

        logger.info(f"Retrieved {len(documents)} documents for BM25 indexing")
        return documents

    # Setup conversation chain
    def _setup_enhanced_chain(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )

        self.prompt = PromptTemplate(
            template="""
            Bạn là một chatbot hỗ trợ sinh viên thân thiện và chuyên nghiệp.

            Câu hỏi hiện tại: {question}
            Dữ liệu tham khảo: {context}
            Lịch sử hội thoại: {chat_history}

            ❗ Hướng dẫn quan trọng:
            - CHỈ sử dụng thông tin từ dữ liệu tham khảo được cung cấp.
            - Trích dẫn đầy đủ các chi tiết cụ thể (GPA, điểm rèn luyện, điều kiện, quy định...).
            - Ưu tiên thông tin chính xác nhất và liên quan nhất với câu hỏi.
            - Nếu không có thông tin phù hợp trong dữ liệu, hãy nói: "Không có thông tin này trong cơ sở dữ liệu của tôi".
            - Sắp xếp thông tin theo mức độ ưu tiên và liên quan.
            - Tránh lặp lại nội dung không cần thiết.

            ➕ Nếu có đường link gốc đi kèm, hãy ghi rõ ở cuối mỗi phần trả lời như sau:
              (Tham khảo: [đường link trích dẫn])

            Trả lời bằng tiếng Việt, chính xác và có cấu trúc rõ ràng.
            """,
            input_variables=["question", "context", "chat_history"]
        )

        USE_LOCAL = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

        if USE_LOCAL:
            model_id = os.getenv("LOCAL_LLM_PATH", "Qwen/Qwen1.5-1.8B")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

            hf_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                temperature=0.1,
                device=0 if torch.cuda.is_available() else -1
            )

            self.llm = HuggingFaceLLMWrapper(hf_pipeline)
        else:
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
                streaming=True,
                temperature=0.1
            )

        retriever_type = os.getenv("RETRIEVER_TYPE", "enhanced").lower()

        if retriever_type == "contextual":
            enhanced_retriever = self._create_contextual_retriever()
        elif retriever_type == "hybrid":
            enhanced_retriever = self._create_hybrid_retriever()
        elif retriever_type == "adaptive":
            enhanced_retriever = self._create_adaptive_retriever()
        else:
            enhanced_retriever = self._create_enhanced_retriever()

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=enhanced_retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    # Get collection information
    def get_collection_info(self) -> Dict[str, Any]:
        docs_count = get_collection_count(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "document_count": docs_count,
            "status": "ready"
        }

    # Check if query is repeated
    def _is_repeated(self, query: str) -> bool:
        history = self.memory.chat_memory.messages
        if not history:
            return False

        for msg in history[-4:] if len(history) >= 4 else history:
            if hasattr(msg, 'content') and query.lower() in msg.content.lower():
                return True
        return False

    # Enhance query with context
    def _enhance_query(self, query: str) -> str:
        history = self.memory.chat_memory.messages
        if not history:
            return query

        recent_context = []
        for msg in history[-2:]:
            if hasattr(msg, 'content') and len(msg.content) < 200:
                recent_context.append(msg.content)

        if recent_context and any(keyword in query.lower() for keyword in ['gì', 'nào', 'như thế nào']):
            context_str = ' '.join(recent_context)
            return f"{context_str}. {query}"

        return query

    # Process query with streaming
    async def answer_query_stream(self, query: str, handler: StreamingCallbackHandler) -> Dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        enhanced_query = self._enhance_query(query)

        if self._is_repeated(enhanced_query):
            count = len(self.memory.chat_memory.messages) // 2 + 1
            enhanced_query = f"{enhanced_query} (lần {count} - cần thông tin chi tiết hơn)"

        response = await self.chain.ainvoke({"question": enhanced_query}, callbacks=[handler])
        await handler.queue.put(None)

        citations = self._process_citations(response.get("source_documents", []))

        return {"answer": response.get("answer", ""), "citations": citations}

    # Process query without streaming
    def answer_query(self, query: str) -> Dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        enhanced_query = self._enhance_query(query)

        if self._is_repeated(enhanced_query):
            count = len(self.memory.chat_memory.messages) // 2 + 1
            enhanced_query = f"{enhanced_query} (lần {count} - cần thông tin chi tiết hơn)"

        response = self.chain({"question": enhanced_query})
        citations = self._process_citations(response.get("source_documents", []))

        return {"answer": response.get("answer", ""), "citations": citations}

    # Process and deduplicate citations
    def _process_citations(self, source_documents: List[Document]) -> List[Citation]:
        citations = []
        seen_content = set()

        for doc in source_documents:
            content = doc.page_content.strip()
            if not content or len(content) < 50:
                continue

            content_hash = hash(content[:300])
            if content_hash in seen_content:
                continue

            seen_content.add(content_hash)

            extracted = extract_source_from_text(doc.page_content)
            citations.append(Citation(
                text=extracted['content'][:2500],
                source=extracted['source']
            ))

        citations.sort(key=lambda x: len(x.text), reverse=True)
        return citations[:20]

    # Clear conversation memory
    def clear_memory(self):
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")

    # Get memory summary
    def get_memory_summary(self) -> Dict[str, Any]:
        if not self.memory:
            return {"message_count": 0, "messages": []}

        messages = self.memory.chat_memory.messages
        return {
            "message_count": len(messages),
            "messages": [{"type": type(msg).__name__, "content": msg.content[:100] + "..."}
                         for msg in messages[-5:]]
        }


# Get singleton instance
def get_chatbot_instance() -> ChatbotRAG:
    global _chatbot_instance

    with _init_lock:
        if _chatbot_instance is None:
            _chatbot_instance = ChatbotRAG()

    return _chatbot_instance


# Check data availability
def check_data_availability() -> Dict[str, Any]:
    collection_name = os.getenv("MILVUS_COLLECTION", "student_support_chatbot")
    hosts = [os.getenv("MILVUS_HOST", "host.docker.internal"), "localhost", "127.0.0.1"]
    port = int(os.getenv("MILVUS_PORT", "19530"))

    connected = False
    for host in hosts:
        if connections.has_connection("default"):
            connections.disconnect("default")

        connections.connect(alias="default", host=host, port=port, timeout=5.0)
        if connections.has_connection("default"):
            connected = True
            break

    docs_count = get_collection_count(collection_name)

    return {
        "available": True,
        "collection_name": collection_name,
        "document_count": docs_count
    }