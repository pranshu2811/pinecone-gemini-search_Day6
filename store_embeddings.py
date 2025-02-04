import uuid
from pinecone import Pinecone
from generate_embeddings import get_embedding
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def store_embeddings(texts):
    """Generate and store multiple embeddings in Pinecone, avoiding duplicates."""
    vectors = []
    
    for text in texts:
        embedding = get_embedding(text)
        if embedding:
            vector_id = text.replace(" ", "_").lower()  # Use text as a unique ID
            vectors.append({"id": vector_id, "values": embedding, "metadata": {"text": text}})

    if vectors:
        index.upsert(vectors=vectors)  # Store only unique embeddings

if __name__ == "__main__":
    texts_to_store = [
        "Hello world",
        "Artificial Intelligence is the future",
        "Pinecone is great for vector search"
    ]
    store_embeddings(texts_to_store)
