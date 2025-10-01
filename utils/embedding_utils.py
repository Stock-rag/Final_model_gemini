"""
Shared embedding and search utilities
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any
from config import Config

class EmbeddingManager:
    """Manages embeddings and vector search across the system"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the sentence transformer model"""
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Embedding model loaded successfully")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = self.model.encode(texts)
        return np.array(embeddings, dtype="float32")

    def create_faiss_index(self, embeddings: np.ndarray, use_cosine: bool = True) -> faiss.Index:
        """Create FAISS index from embeddings"""
        dim = embeddings.shape[1]

        if use_cosine:
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
        else:
            index = faiss.IndexFlatL2(dim)  # L2 distance

        index.add(embeddings)
        return index

    def search(self, query: str, index: faiss.Index, texts: List[str],
               metadata: List[Dict] = None, top_k: int = None, use_cosine: bool = True) -> List[Tuple]:
        """Search for similar texts using FAISS"""
        top_k = top_k or Config.SEARCH_CONFIG["top_k"]

        # Encode query
        query_emb = self.encode([query])

        if use_cosine:
            faiss.normalize_L2(query_emb)

        # Search
        scores, indices = index.search(query_emb, top_k)

        # Format results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            result_data = {
                "text": texts[idx],
                "score": float(score),
                "index": int(idx)
            }
            if metadata and idx < len(metadata):
                result_data["metadata"] = metadata[idx]

            results.append(result_data)

        return results

class NewsEmbeddingProcessor:
    """Specialized processor for news data"""

    def __init__(self, embedding_manager: EmbeddingManager = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()

    def process_news_data(self, news_dict: Dict[str, List[Dict]]) -> Tuple[List[str], List[Dict], np.ndarray]:
        """Process news data into texts, metadata, and embeddings"""
        texts = []
        metadata = []

        for ticker, articles in news_dict.items():
            for article in articles:
                headline = article.get("headline", "").strip()
                summary = article.get("summary", "").strip()

                # Combine headline and summary
                combined_text = f"{headline}. {summary}" if summary else headline
                texts.append(combined_text)

                # Store metadata
                metadata.append({
                    "ticker": ticker,
                    "headline": headline,
                    "summary": summary,
                    "source": article.get("source", ""),
                    "url": article.get("url", "")
                })

        # Generate embeddings
        embeddings = self.embedding_manager.encode(texts)

        return texts, metadata, embeddings

    def create_searchable_index(self, news_dict: Dict[str, List[Dict]]) -> Tuple[faiss.Index, List[str], List[Dict]]:
        """Create a searchable FAISS index from news data"""
        texts, metadata, embeddings = self.process_news_data(news_dict)
        index = self.embedding_manager.create_faiss_index(embeddings, use_cosine=True)

        return index, texts, metadata