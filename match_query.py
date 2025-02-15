from langchain_huggingface import HuggingFaceEmbeddings
from upstash_vector import Index
from dotenv import load_dotenv
import os

def search_similar_chunks(query: str, top_k: int = 20) -> list:
    """
    Search for text chunks similar to the input query.
    
    Args:
        query (str): The search query text
        top_k (int): Number of top results to return (default: 5)
    
    Returns:
        list: List of dictionaries containing matching passages and their scores
    """
    load_dotenv()
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )
    
    vector_store = Index(
        url=os.getenv("UPSTASH_VECTOR_REST_URL"),
        token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    )
    
    query_embedding = embeddings.embed_query(query)
    
    results = vector_store.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    formatted_results = []
    for result in results:
        formatted_results.append({
            'score': result.score,
            'content': result.metadata.get('content', 'No content found') if result.metadata else 'No metadata found',
            'source': result.metadata.get('source', 'Unknown source') if result.metadata else 'Unknown source'
        })
    
    return formatted_results