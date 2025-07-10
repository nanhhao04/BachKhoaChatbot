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
try:
    from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
    from langchain.retrievers.multi_vector import MultiVectorRetriever
    from langchain.storage import InMemoryStore
except ImportError:
    # Fallback for older langchain versions
    MultiQueryRetriever = None
    EnsembleRetriever = None
    MultiVectorRetriever = None
    InMemoryStore = None

try:
    from langchain.retrievers import BM25Retriever
except ImportError:
    BM25Retriever = None

# Fix imports - try multiple possible locations
try:
    from app.utils import extract_source_from_text
    from app.schema import Citation
except ImportError:
    try:
        from utils import extract_source_from_text
        from schema import Citation
    except ImportError:
        try:
            from .utils import extract_source_from_text
            from .schema import Citation
        except ImportError:
            # Create placeholder functions/classes if imports fail
            def extract_source_from_text(text):
                """Placeholder function - implement or fix import"""
                return {"content": text[:200], "source": "Unknown"}


            class Citation:
                def __init__(self, text="", source=""):
                    self.text = text
                    self.source = source

# Setup
logger = logging.getLogger(__name__)
load_dotenv()

# Singleton instance
_chatbot_instance = None
_init_lock = threading.Lock()


def get_collection_count(collection_name: str) -> int:
    """Lấy số lượng entities trong collection với PyMilvus mới"""
    try:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            return collection.num_entities
        return 0
    except Exception as e:
        logger.error(f"Error getting collection count: {e}")
        return 0


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


class HuggingFaceLLMWrapper(LLM):
    """Wrapper for Hugging Face pipeline to work with LangChain"""

    def __init__(self, hf_pipeline):
        super().__init__()
        self.pipeline = hf_pipeline

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Execute the pipeline and return generated text"""
        try:
            output = self.pipeline(prompt, return_full_text=False)[0]['generated_text']
            return output.strip()
        except Exception as e:
            logger.error(f"Error in HuggingFace pipeline: {e}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn."

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"


class ContextualRetriever:
    """Custom retriever có xem xét context từ lịch sử hội thoại"""

    def __init__(self, base_retriever, memory, llm):
        self.base_retriever = base_retriever
        self.memory = memory
        self.llm = llm

    def get_relevant_documents(self, query: str):
        # Lấy context từ lịch sử
        context = self._get_conversation_context()

        # Expand query với context
        expanded_query = self._expand_query_with_context(query, context)

        # Retrieve với expanded query
        docs = self.base_retriever.get_relevant_documents(expanded_query)

        # Re-rank dựa trên context
        return self._rerank_documents(docs, query, context)

    def _get_conversation_context(self):
        """Lấy context từ 2-3 turn hội thoại gần nhất"""
        messages = self.memory.chat_memory.messages
        if not messages:
            return ""

        recent_messages = messages[-4:]  # 2 turns
        context_parts = []

        for msg in recent_messages:
            if hasattr(msg, 'content'):
                content = msg.content.strip()
                if len(content) > 10:  # Skip very short messages
                    context_parts.append(content)

        return " ".join(context_parts[-3:])  # Last 3 meaningful messages

    def _expand_query_with_context(self, query: str, context: str):
        """Mở rộng query với context"""
        if not context:
            return query

        # Nếu query ngắn và có từ chỉ định (này, đó, thế...)
        pronouns = ["này", "đó", "thế", "vậy", "như vậy", "gì", "nào"]
        if any(pronoun in query.lower() for pronoun in pronouns):
            return f"{context}. {query}"

        return query

    def _rerank_documents(self, docs, original_query, context):
        """Re-rank documents dựa trên relevance với context"""
        if not context:
            return docs

        # Simple scoring based on keyword overlap
        scored_docs = []
        context_keywords = set(context.lower().split())
        query_keywords = set(original_query.lower().split())
        all_keywords = context_keywords.union(query_keywords)

        for doc in docs:
            doc_text = doc.page_content.lower()
            # Count keyword matches
            matches = sum(1 for kw in all_keywords if kw in doc_text)
            score = matches / len(all_keywords) if all_keywords else 0
            scored_docs.append((doc, score))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]


class AdaptiveRetriever:
    """Adaptive retriever thay đổi strategy dựa trên query type"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

        # Different retrievers for different query types
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
        """Phân loại query để chọn retrieval strategy phù hợp"""
        query_lower = query.lower()

        # Specific questions (có số liệu, điều kiện cụ thể)
        specific_indicators = [
            "bao nhiêu", "điểm", "gpa", "tín chỉ", "học phí",
            "điều kiện", "yêu cầu", "quy định", "thủ tục"
        ]
        if any(indicator in query_lower for indicator in specific_indicators):
            return "specific"

        # Comparative questions
        comparative_indicators = [
            "khác nhau", "so sánh", "giống", "khác", "nào tốt hơn",
            "ưu nhược điểm", "phân biệt"
        ]
        if any(indicator in query_lower for indicator in comparative_indicators):
            return "comparative"

        # General questions
        return "general"


class ChatbotRAG:
    def __init__(self):
        logger.info("Initializing ChatbotRAG...")
        self.initialized = False
        try:
            # Configuration
            self.collection_name = os.getenv("MILVUS_COLLECTION", "student_support_chatbot")

            # Connect to Milvus
            self._connect_milvus()

            # Kiểm tra collection tồn tại
            self._verify_collection()

            # Setup embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="intfloat/e5-small-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

            # Setup vectorstore và conversation chain
            self._setup_vectorstore()
            self._setup_enhanced_chain()

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            self.initialized = True
            logger.info("ChatbotRAG initialization complete")

        except Exception as e:
            logger.exception("Initialization error")
            raise e

    def _connect_milvus(self):
        """Connect to Milvus server"""
        hosts = [os.getenv("MILVUS_HOST", "host.docker.internal"), "localhost", "127.0.0.1"]
        port = int(os.getenv("MILVUS_PORT", "19530"))

        for host in hosts:
            try:
                if connections.has_connection("default"):
                    connections.disconnect("default")

                connections.connect(alias="default", host=host, port=port, timeout=5.0)

                if connections.has_connection("default"):
                    logger.info(f"Connected to Milvus at {host}:{port}")
                    self.connection_args = {"host": host, "port": port}
                    return
            except Exception as e:
                logger.warning(f"Cannot connect to Milvus at {host}:{port}: {e}")

        raise ConnectionError("Failed to connect to Milvus server")

    def _verify_collection(self):
        """Verify that collection exists and has data"""
        try:
            if not utility.has_collection(self.collection_name):
                raise ValueError(
                    f"Collection '{self.collection_name}' does not exist. Please create it first using data_processor.py")

            docs_count = get_collection_count(self.collection_name)
            if docs_count == 0:
                raise ValueError(
                    f"Collection '{self.collection_name}' is empty. Please add data using data_processor.py")

            logger.info(f"Using existing collection '{self.collection_name}' with {docs_count} documents")

        except Exception as e:
            logger.error(f"Collection verification failed: {e}")
            raise e

    def _setup_vectorstore(self):
        """Setup vector store from existing collection"""
        try:
            self.vectorstore = Milvus(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
                connection_args=self.connection_args
            )
            logger.info(f"Vectorstore loaded successfully")

        except Exception as e:
            logger.error(f"Error setting up vectorstore: {e}")
            raise e

    def _create_enhanced_retriever(self):
        """Tạo enhanced retriever với multiple strategies"""
        try:
            # 1. Base similarity retriever
            base_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 30}
            )

            # 2. MMR (Maximum Marginal Relevance) retriever for diversity
            mmr_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 30,
                    "fetch_k": 50,  # Fetch more candidates
                    "lambda_mult": 0.7  # Balance relevance vs diversity
                }
            )

            # Try to use advanced retrievers if available
            if EnsembleRetriever is not None:
                # 4. Ensemble Retriever - kết hợp multiple retrievers
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[base_retriever, mmr_retriever],
                    weights=[0.6, 0.4]  # Weight cho từng retriever
                )

                # Try MultiQueryRetriever if available
                if MultiQueryRetriever is not None:
                    try:
                        # 3. Multi-Query Retriever - tạo nhiều query variants
                        multi_query_retriever = MultiQueryRetriever.from_llm(
                            retriever=ensemble_retriever,
                            llm=self.llm,
                            include_original=True
                        )
                        logger.info("Using Multi-Query + Ensemble Retriever")
                        return multi_query_retriever
                    except Exception as e:
                        logger.warning(f"MultiQueryRetriever failed: {e}, using Ensemble only")

                logger.info("Using Ensemble Retriever (Similarity + MMR)")
                return ensemble_retriever
            else:
                logger.info("Using MMR Retriever (advanced features not available)")
                return mmr_retriever

        except Exception as e:
            logger.error(f"Error creating enhanced retriever: {e}")
            # Fallback to basic similarity
            logger.info("Falling back to basic similarity retriever")
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 35}
            )

    def _create_contextual_retriever(self):
        """Tạo retriever có xem xét context từ lịch sử hội thoại"""
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 35}
        )

        return ContextualRetriever(base_retriever, self.memory, self.llm)

    def _create_hybrid_retriever(self):
        """Tạo hybrid retriever kết hợp semantic và keyword search"""

        # Semantic retriever (existing)
        semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 30}
        )

        # Try to create keyword-based retriever if BM25 is available
        if BM25Retriever is not None:
            try:
                # Lấy tất cả documents để tạo BM25 index
                all_docs = self._get_all_documents_from_collection()
                if all_docs:
                    keyword_retriever = BM25Retriever.from_documents(all_docs)
                    keyword_retriever.k = 30

                    # Ensemble retriever
                    if EnsembleRetriever is not None:
                        hybrid_retriever = EnsembleRetriever(
                            retrievers=[semantic_retriever, keyword_retriever],
                            weights=[0.7, 0.3]  # Ưu tiên semantic hơn keyword
                        )
                        logger.info("Using Hybrid Retriever (Semantic + BM25)")
                        return hybrid_retriever

            except Exception as e:
                logger.error(f"Error creating BM25 retriever: {e}")

        logger.warning("BM25Retriever not available or failed, using semantic only")
        return semantic_retriever

    def _create_adaptive_retriever(self):
        """Tạo adaptive retriever thay đổi strategy dựa trên query type"""
        return AdaptiveRetriever(self.vectorstore, self.llm)

    def _get_all_documents_from_collection(self):
        """Lấy tất cả documents từ collection để tạo keyword index"""
        try:
            # Query all documents from vectorstore
            collection = Collection(self.collection_name)
            collection.load()

            # Get all entities (limit to reasonable number)
            results = collection.query(
                expr="",  # Empty expression = get all
                output_fields=["text", "source"],  # Adjust field names based on your schema
                limit=5000  # Adjust limit as needed
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

        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

    def _setup_enhanced_chain(self):
        """Setup conversation chain với enhanced retriever"""
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )

        # Create enhanced prompt template
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
            # ✅ Dùng Hugging Face Qwen
            model_id = os.getenv("LOCAL_LLM_PATH", "Qwen/Qwen1.5-1.8B")

            try:
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
            except Exception as e:
                logger.error(f"Error loading local LLM: {e}")
                logger.info("Falling back to OpenAI API")
                USE_LOCAL = False

        if not USE_LOCAL:
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
                streaming=True,
                temperature=0.1
            )

        # ✅ CHỌN ENHANCED RETRIEVER STRATEGY
        retriever_type = os.getenv("RETRIEVER_TYPE", "enhanced").lower()

        if retriever_type == "contextual":
            enhanced_retriever = self._create_contextual_retriever()
        elif retriever_type == "hybrid":
            enhanced_retriever = self._create_hybrid_retriever()
        elif retriever_type == "adaptive":
            enhanced_retriever = self._create_adaptive_retriever()
        else:  # default: enhanced (ensemble)
            enhanced_retriever = self._create_enhanced_retriever()

        # Create chain với enhanced retriever
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=enhanced_retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about current collection"""
        try:
            if utility.has_collection(self.collection_name):
                docs_count = get_collection_count(self.collection_name)
                return {
                    "collection_name": self.collection_name,
                    "document_count": docs_count,
                    "status": "ready"
                }
            else:
                return {
                    "collection_name": self.collection_name,
                    "document_count": 0,
                    "status": "not_found"
                }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "status": "error",
                "error": str(e)
            }

    def _is_repeated(self, query: str) -> bool:
        """Check if query is repeated in recent history"""
        history = self.memory.chat_memory.messages
        if not history:
            return False

        for msg in history[-4:] if len(history) >= 4 else history:
            if hasattr(msg, 'content') and query.lower() in msg.content.lower():
                return True
        return False

    def _enhance_query(self, query: str) -> str:
        """Enhance query with context from conversation history"""
        history = self.memory.chat_memory.messages
        if not history:
            return query

        # Add context from recent questions if relevant
        recent_context = []
        for msg in history[-2:]:  # Last 2 messages
            if hasattr(msg, 'content') and len(msg.content) < 200:  # Short messages likely questions
                recent_context.append(msg.content)

        if recent_context and any(keyword in query.lower() for keyword in ['gì', 'nào', 'như thế nào']):
            context_str = ' '.join(recent_context)
            return f"{context_str}. {query}"

        return query

    async def answer_query_stream(self, query: str, handler: StreamingCallbackHandler) -> Dict[str, Any]:
        """Process query and stream response"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Enhance query with context
            enhanced_query = self._enhance_query(query)

            # Handle repeated queries
            if self._is_repeated(enhanced_query):
                count = len(self.memory.chat_memory.messages) // 2 + 1
                enhanced_query = f"{enhanced_query} (lần {count} - cần thông tin chi tiết hơn)"

            # Get response from chain
            response = await self.chain.ainvoke({"question": enhanced_query}, callbacks=[handler])
            await handler.queue.put(None)  # Signal end of stream

            # Process sources with better deduplication
            citations = self._process_citations(response.get("source_documents", []))

            return {"answer": response.get("answer", ""), "citations": citations}
        except Exception as e:
            logger.exception(f"Error in streaming: {e}")
            await handler.queue.put(None)
            raise e

    def answer_query(self, query: str) -> Dict[str, Any]:
        """Process query and return response (non-streaming)"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Enhance query with context
            enhanced_query = self._enhance_query(query)

            if self._is_repeated(enhanced_query):
                count = len(self.memory.chat_memory.messages) // 2 + 1
                enhanced_query = f"{enhanced_query} (lần {count} - cần thông tin chi tiết hơn)"

            response = self.chain({"question": enhanced_query})

            # Process sources with better deduplication
            citations = self._process_citations(response.get("source_documents", []))

            return {"answer": response.get("answer", ""), "citations": citations}
        except Exception as e:
            logger.exception(f"Error in response: {e}")
            raise e

    def _process_citations(self, source_documents: List[Document]) -> List[Citation]:
        """Process and deduplicate citations"""
        citations = []
        seen_content = set()

        for doc in source_documents:
            content = doc.page_content.strip()
            if not content or len(content) < 50:  # Skip very short content
                continue

            # Create a hash for deduplication (first 100 chars)
            content_hash = hash(content[:300])
            if content_hash in seen_content:
                continue

            seen_content.add(content_hash)

            try:
                extracted = extract_source_from_text(doc.page_content)
                citations.append(Citation(
                    text=extracted['content'][:2500],  # Limit citation length
                    source=extracted['source']
                ))
            except Exception as e:
                logger.warning(f"Error extracting source: {e}")
                # Fallback citation
                citations.append(Citation(
                    text=content[:2500],
                    source=doc.metadata.get('source', 'Unknown')
                ))

        # Sort citations by relevance (longer content usually more relevant)
        citations.sort(key=lambda x: len(x.text), reverse=True)

        return citations[:20]  # Limit to top 20 citations

    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of conversation memory"""
        if not self.memory:
            return {"message_count": 0, "messages": []}

        messages = self.memory.chat_memory.messages
        return {
            "message_count": len(messages),
            "messages": [{"type": type(msg).__name__, "content": msg.content[:100] + "..."}
                         for msg in messages[-5:]]  # Last 5 messages
        }


def get_chatbot_instance() -> ChatbotRAG:
    """Get or create singleton instance of ChatbotRAG"""
    global _chatbot_instance

    with _init_lock:
        if _chatbot_instance is None:
            _chatbot_instance = ChatbotRAG()

    return _chatbot_instance


def check_data_availability() -> Dict[str, Any]:
    """Check if data is available in Milvus without initializing full chatbot"""
    try:
        # Connect to Milvus
        collection_name = os.getenv("MILVUS_COLLECTION", "student_support_chatbot")
        hosts = [os.getenv("MILVUS_HOST", "host.docker.internal"), "localhost", "127.0.0.1"]
        port = int(os.getenv("MILVUS_PORT", "19530"))

        connected = False
        for host in hosts:
            try:
                if connections.has_connection("default"):
                    connections.disconnect("default")

                connections.connect(alias="default", host=host, port=port, timeout=5.0)
                if connections.has_connection("default"):
                    connected = True
                    break
            except:
                continue

        if not connected:
            return {"available": False, "error": "Cannot connect to Milvus"}

        # Check collection
        if not utility.has_collection(collection_name):
            return {"available": False, "error": f"Collection '{collection_name}' does not exist"}

        docs_count = get_collection_count(collection_name)
        if docs_count == 0:
            return {"available": False, "error": f"Collection '{collection_name}' is empty"}

        return {
            "available": True,
            "collection_name": collection_name,
            "document_count": docs_count
        }

    except Exception as e:
        return {"available": False, "error": str(e)}