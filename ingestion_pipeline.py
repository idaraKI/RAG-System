import os
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path = "docs"):
    """Load all text files from the docs directory."""
    print(f"Loading documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.Please create it and add your company files.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
    )

    documents = loader.load()
 
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in the directory {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"source: {doc.metadata['source']}")
        print(f"content length: {len(doc.page_content)} characters")
        print(f"content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}") 

    return documents
        

def split_documents(documents, chunk_size=500, chunk_overlap=0):
    """Split documents into smaller chunks with overlap."""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks.")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist  ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("--- Creating Chroma vector store ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")

    return vector_store

def main():
    print("Starting ingestion pipeline...")
    
    documents = load_documents(docs_path="docs")

    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)
    

if __name__ == "__main__":
    main()