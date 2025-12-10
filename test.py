from dotenv import load_dotenv
load_dotenv()


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# List ALL documents stored in Chroma
docs = db.get()

print("Number of embeddings:", len(docs["ids"]))

for i, doc in enumerate(docs["documents"][:5]):  # show first 5 docs
    print(f"\nDocument {i+1}:")
    print(doc[:300])  # preview 300 chars
