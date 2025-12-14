import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from ingestion_pipeline import run_ingestion_pipeline, PERSIST_DIR

load_dotenv()


# Streamlit UI 
st.set_page_config(page_title="Rayda RAG System", layout="wide")
st.title("Rayda RAG System")
st.write("Ask any question based on Rayda's internal documents.")

# --- Auto-run ingestion pipeline if vector store is missing ---
if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    st.info("No vector store found. Running ingestion pipeline...")
    run_ingestion_pipeline()
    st.success("Ingestion completed!")

# --- Load vector store ---
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

retriever = db.as_retriever(search_kwargs={"k": 7})

# --- Session State (Chat Memory) ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display previous chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Chat Input ---
user_query = st.chat_input("Ask a question about Rayda...")

if user_query:
    # --- Store user message ---
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.write(user_query)

# --- Retrieve documents ---
    relevant_docs = retriever.invoke(user_query)

    # --- Build LLM input ---
    combined_input = f"""You are a helpful assistant supporting Rayda.

### RULES ABOUT KNOWLEDGE USE
1. **If the user's question is about Rayda, its processes, policies, operations, or anything internal**, 
   ONLY use the Rayda documents provided below.

2. **If the documents do not contain the answer, respond politely without guessing. Use a friendly message.**
    

3. **If the user's question is general :**
   You may answer using your own general knowledge.

4. **Never invent or guess details about Rayda.**

5. Your tone should be professional, friendly, and conversational.

6. Even if the user phrases the question differently from the document wording, try to understand the meaning and find relevant content.

### USER QUESTION:
{user_query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Guidelines:
- First determine whether the question is Rayda-related or general.
# - If Rayda-related → use ONLY the documents.
- If not Rayda-related → answer normally.
- If documents lack the answer → give the fallback message above.
"""

    # Run model
    model = ChatOpenAI(model="gpt-4o")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    with st.spinner("Generating answer..."):
        response = model.invoke(messages)

    st.chat_message("RAG Answer")
    st.write(response.content)

    # store assistant response in session state for chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
