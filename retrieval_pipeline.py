from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

query = "What is Rayda's device procurement process?"

retriever = db.as_retriever(search_kwargs={"k": 2,})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("--- context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    

combined_input = f"""Based on the following documents,please answer this questions: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, concise, and accurate answer using only the information from the documents above.If the answer is not contained within the documents, respond with "I currently do not have that information, i can escalate the issue to my superior."."""

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a Rayda internal knowledge assistant"),
    HumanMessage(content=combined_input),
]

response = model.invoke(messages)

print("\n--- Generated Response ---")

print("Content :")
print(response.content)
