import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

API_KEY = "d2kfc19r01qs23a21bbgd2kfc19r01qs23a21bc0"
NEWS_URL = f"https://finnhub.io/api/v1/news?category=general&token={API_KEY}"

# 1. Fetch Data
def fetch_news(url=NEWS_URL, limit=100):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data[:limit]  # return top N news articles
    else:
        raise Exception(f"Failed to fetch news: {response.status_code}, {response.text}")

# 2. Preprocess Data (extract headlines or content for embeddings)
def preprocess_news(news_data):
    texts = [article["headline"]  for article in news_data]
    for article in news_data:     
        print(f"Headline: {article['headline']}")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}\n")
    return texts

# 3. Generate Embeddings
def generate_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return np.array(embeddings, dtype="float32")

# 4. Store in FAISS
def store_in_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 5. Search in FAISS
def search_faiss(query, model, index, texts, top_k=3):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype="float32"), top_k)
    results = [(texts[i], D[0][idx]) for idx, i in enumerate(I[0])]
    return results

# ---- Example Run ----
if __name__ == "__main__":
    # Fetch and preprocess
    news_data = fetch_news(limit=100)
    texts = preprocess_news(news_data)
    
    # Generate embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = generate_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Store in FAISS
    index = store_in_faiss(embeddings)
    
    # Search Example
    query = "stock market"
    results = search_faiss(query, model, index, texts, top_k=3)
    
    print("\nTop Results for Query:", query)
    for res, score in results:
        print(f"Text: {res}\nScore: {score}\n")
