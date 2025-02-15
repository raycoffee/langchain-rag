from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from upstash_vector import Index
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/books"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_documents(documents):
    # For question-answering, smaller chunks often work better
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # Even smaller chunks
        chunk_overlap=50,    # Smaller overlap
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        keep_separator=True  # Keep the separators to maintain context
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={
            'normalize_embeddings': True,  # Ensures cosine similarity works better
            'batch_size': 32
        }
    )
    
    vector_store = Index(
        url=os.getenv("UPSTASH_VECTOR_REST_URL"),
        token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    )
    
    # Batch process embeddings for better efficiency
    texts = [chunk.page_content for chunk in chunks]
    all_embeddings = embeddings.embed_documents(texts)
    
    # Create vector records
    vectors = [
        {
            "id": str(hash(chunk.page_content)),
            "vector": embedding,
            "metadata": {
                "source": chunk.metadata.get('source', ''),
                "content": chunk.page_content  # Store content in metadata
            }
        }
        for chunk, embedding in zip(chunks, all_embeddings)
    ]
    
    # Batch upsert for better performance
    vector_store.upsert(vectors=vectors)
    
    return vector_store


def main():
    # Load and process documents
    documents = load_documents()
    chunks = split_documents(documents)
    

    # Create the vector store
    vector_store = create_vector_store(chunks)
    return vector_store

main()