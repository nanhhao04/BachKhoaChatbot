import os
import argparse
import logging
import hashlib
import re
from typing import List, Dict, Any
from collections import Counter
import pymilvus
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.collection_name = os.getenv('MILVUS_COLLECTION', 'student_support_chatbot')
        self.dimension = 384
        self._connect_milvus()
        self.embedder = SentenceTransformer('intfloat/e5-small-v2')
        self._setup_collection()

    def _connect_milvus(self):
        """Connect to Milvus with fallback hosts"""
        hosts = [('host.docker.internal', 19530), ('localhost', 19530), ('127.0.0.1', 19530)]

        for host, port in hosts:
            try:
                connections.connect(alias="default", host=host, port=port, timeout=5)
                logger.info(f"Connected to Milvus at {host}:{port}")
                return
            except Exception as e:
                logger.warning(f"Cannot connect to {host}:{port}: {e}")

        raise ConnectionError("Failed to connect to Milvus")

    def _setup_collection(self):
        """Setup collection with simplified schema"""
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=511),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=128)
        ]

        schema = CollectionSchema(fields=fields, description="Student support collection")

        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            if not self.collection.has_index():
                self._create_index()
        else:
            self.collection = Collection(name=self.collection_name, schema=schema)
            self._create_index()

    def _create_index(self):
        """Create COSINE index"""
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better similarity comparison"""
        text = ' '.join(text.split()).lower()
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _create_content_hash(self, text: str, source: str = "", page: int = 0) -> str:
        """Create unique hash for content deduplication"""
        normalized = self._normalize_text(text)
        content_id = f"{normalized}#{source}#{page}"
        return hashlib.md5(content_id.encode('utf-8')).hexdigest()

    def _remove_duplicates(self, chunks: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """Remove duplicate chunks based on simple text matching"""
        if not chunks:
            return chunks

        unique_chunks = {}
        seen_hashes = set()

        for chunk in chunks:
            content_hash = self._create_content_hash(chunk['text'], chunk.get('source', ''), chunk.get('page', 0))

            # Skip exact duplicates
            if content_hash in seen_hashes:
                continue

            # Simple text similarity check using normalized text
            normalized_text = self._normalize_text(chunk['text'])
            is_duplicate = False

            for existing_hash, existing_chunk in unique_chunks.items():
                existing_normalized = self._normalize_text(existing_chunk['text'])

                # Simple similarity check based on common words
                words1, words2 = set(normalized_text.split()), set(existing_normalized.split())
                if words1 and words2:
                    jaccard_similarity = len(words1 & words2) / len(words1 | words2)

                    if jaccard_similarity >= threshold:
                        # Keep longer chunk
                        if len(chunk['text']) > len(existing_chunk['text']):
                            del unique_chunks[existing_hash]
                            seen_hashes.discard(existing_hash)
                            break
                        else:
                            is_duplicate = True
                            break

            if not is_duplicate:
                unique_chunks[content_hash] = chunk
                seen_hashes.add(content_hash)

        logger.info(f"Removed {len(chunks) - len(unique_chunks)} duplicates from {len(chunks)} chunks")
        return list(unique_chunks.values())

    def _extract_pdf_chunks(self, pdf_path: str) -> List[Dict]:
        """Extract chunks from PDF with simple text processing"""
        try:
            doc = fitz.open(pdf_path)
            chunks = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))  # Sort by position

                page_text = ""
                for block in blocks:
                    text = block[4].strip()
                    if text:
                        text = text.replace("-\n", "").replace("\n", " ")
                        page_text += text + " "

                # Split into meaningful chunks
                sentences = re.split(r'[.!?]+', page_text)
                current_chunk = ""

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    if len(current_chunk + sentence) > 300:  # Chunk size limit
                        if current_chunk and len(current_chunk) > 80:
                            chunks.append({
                                'text': current_chunk.strip(),
                                'source': os.path.basename(pdf_path),
                                'page': page_num + 1
                            })
                        current_chunk = sentence
                    else:
                        current_chunk += (" " + sentence if current_chunk else sentence)

                # Add final chunk
                if current_chunk and len(current_chunk) > 100:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'source': os.path.basename(pdf_path),
                        'page': page_num + 1
                    })

            return chunks
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return []

    def _process_pdf_directory(self, pdf_dir: str) -> List[Dict]:
        """Process all PDFs in directory"""
        all_chunks = []

        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, filename)
                chunks = self._extract_pdf_chunks(pdf_path)
                all_chunks.extend(chunks)

        return all_chunks

    def process_pdfs_to_vectorstore(self, pdf_dir: str, force: bool = False) -> Dict[str, Any]:
        """Process PDFs and create vectorstore"""
        try:
            # Check existing collection
            info = self.get_collection_info()
            if info['exists'] and info['count'] > 0 and not force:
                return {
                    'success': False,
                    'error': f"Collection exists with {info['count']} documents. Use --force to overwrite."
                }

            if force and info['exists']:
                self.delete_collection()
                self._setup_collection()

            # Process PDFs
            logger.info(f"Processing PDFs from {pdf_dir}")
            chunks = self._process_pdf_directory(pdf_dir)

            if not chunks:
                return {'success': False, 'error': "No content extracted"}

            # Remove duplicates
            unique_chunks = self._remove_duplicates(chunks)

            # Create embeddings
            texts = [chunk['text'] for chunk in unique_chunks]
            embeddings = self.embedder.encode(texts)

            # Prepare documents
            documents = []
            for i, chunk in enumerate(unique_chunks):
                documents.append({
                    'text': chunk['text'],
                    'vector': embeddings[i].tolist(),
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'content_hash': self._create_content_hash(chunk['text'], chunk['source'], chunk['page'])
                })

            # Insert to Milvus
            self._insert_documents(documents)

            return {
                'success': True,
                'message': f'Successfully processed {len(os.listdir(pdf_dir))} PDFs',
                'details': {
                    'total_chunks': len(chunks),
                    'unique_chunks': len(unique_chunks),
                    'documents_inserted': len(documents)
                }
            }

        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            return {'success': False, 'error': str(e)}

    def _insert_documents(self, documents: List[Dict]):
        """Insert documents to Milvus in batches"""
        batch_size = 100
        self.collection.load()

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Prepare batch data
            batch_data = [
                [doc['text'] for doc in batch],
                [doc['vector'] for doc in batch],
                [doc['source'] for doc in batch],
                [doc['page'] for doc in batch],
                [doc['content_hash'] for doc in batch]
            ]

            self.collection.insert(batch_data)
            self.collection.flush()

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            if not utility.has_collection(self.collection_name):
                return {'exists': False, 'count': 0}

            collection = Collection(self.collection_name)
            collection.load()

            return {
                'exists': True,
                'count': collection.num_entities,
                'indexes': len(collection.indexes)
            }
        except Exception as e:
            return {'exists': None, 'count': 0, 'error': str(e)}

    def delete_collection(self) -> bool:
        """Delete collection"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False




def main():
    parser = argparse.ArgumentParser(description='PDF Processing for Student Support Chatbot')
    parser.add_argument('--pdf-dir', help='Directory containing PDF files')
    parser.add_argument('--action', choices=['create', 'delete', 'info', 'test'], default='create')
    parser.add_argument('--force', action='store_true', help='Force recreate collection')
    parser.add_argument('--test-query', help='Query for testing similarity')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')

    args = parser.parse_args()

    try:
        processor = DataProcessor()

        if args.action == 'info':
            print(f"Collection info: {processor.get_collection_info()}")
        elif args.action == 'delete':
            print(f"Delete result: {processor.delete_collection()}")
        elif args.action == 'create':
            if not args.pdf_dir:
                print("Error: --pdf-dir required")
                return
            print(f"Result: {processor.process_pdfs_to_vectorstore(args.pdf_dir, args.force)}")


    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()