import finnhub
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# Init client
finnhub_client = finnhub.Client(api_key="d2rdj81r01qlk22srilgd2rdj81r01qlk22srim0")

# --- 1. Extract tickers from query (demo with lookup dict) ---
TICKER_LOOKUP = {
    "apple": "AAPL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    # add more as needed
}

def fetch_stocks(query):
    query_lower = query.lower()
    tickers = [ticker for name, ticker in TICKER_LOOKUP.items() if name in query_lower]
    return tickers

# --- 2. Fetch company-specific news ---
def fetch_news(tickers, limit=100):
    results = {}
    for ticker in tickers:
        news = finnhub_client.company_news(
            ticker,
            _from="2024-09-03",
            to="2025-09-02"
        )
        results[ticker] = news[:limit]
    return results

# --- 3. Preprocess (flatten dict -> list) ---
def preprocess_news(news_dict):
    texts, meta = [], []
    for ticker, articles in news_dict.items():
        for article in articles:

            summary = article.get("summary", "").strip()
            headline = article.get("headline", "").strip()

            combined_text = f"{headline}. {summary}"
            texts.append(combined_text)

            meta.append({
                "ticker": ticker,
                "headline": headline,
                "summary": summary,
                "source": article["source"],
                "url": article["url"]
            })
    return texts, meta

# --- 4. Embeddings ---
def generate_embeddings(texts, model):
    embeddings = model.encode(texts)
    return np.array(embeddings, dtype="float32")

# --- 5. Store in FAISS ---
def store_in_faiss(embeddings):
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)         # normalize embeddings for cosine
    index = faiss.IndexFlatIP(dim)         # inner product = cosine similarity after normalization
    index.add(embeddings)
    return index

# --- 6. Search ---
def search_faiss(query, model, index, texts, meta, top_k=3):
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)          # normalize query too
    D, I = index.search(np.array(query_emb, dtype="float32"), top_k)
    results = [(texts[i], meta[i], D[0][idx]) for idx, i in enumerate(I[0])]
    return results

# --- Example Run ---
if __name__ == "__main__":
    query = "how are Apple, Google and Tesla doing?"

    tickers = fetch_stocks(query)
    print("Detected tickers:", tickers)

    # Fetch + preprocess
    news_dict = fetch_news(tickers, limit=20)
    texts, meta = preprocess_news(news_dict)

    # Embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = generate_embeddings(texts, model)

    # FAISS index
    index = store_in_faiss(embeddings)

    # Search
    results = search_faiss(query, model, index, texts, meta, top_k=10)

    print("\nTop Results for Query:", query)
    for headline, m, score in results:
        print(f"[{m['ticker']}] {headline} (score={score:.2f})\n  Source: {m['source']} | URL: {m['url']}\n")