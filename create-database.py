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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):

    # Initialize the embedding model with one that outputs 1536 dimensions
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"  # Changed model to match required dimensions
    )

    # Create an Upstash vector store
    vector_store = Index(
        url=os.getenv("UPSTASH_VECTOR_REST_URL"),
        token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    )
    
    # Add documents to the vector store one by one
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk.page_content)
        vector_store.upsert(
            vectors=[
                {
                    "id": str(hash(chunk.page_content)),
                    "vector": embedding,
                    "metadata": chunk.metadata,
                    "content": chunk.page_content
                }
            ]
        )
    
    return vector_store

def main():
    # Load and process documents
    documents = load_documents()
    chunks = split_documents(documents)
    

    # Create the vector store
    # vector_store = create_vector_store(chunks)
    # return vector_store

main()