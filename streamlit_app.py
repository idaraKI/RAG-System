import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


# --- Load Vector Store ---
PERSIST_DIR = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)


# Streamlit UI 
st.set_page_config(page_title="Rayda RAG System", layout="wide")
st.title("Rayda RAG System")
st.write("Ask any question based on Rayda's internal documents.")

# Input field
query = st.text_input("Enter your question:", "")

if st.button("Submit"):
    if not query.strip():
        st.error("Please enter a question.")
    else:
        # Retrieve documents
        retriever = db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)

        # Display retrieved context
        st.subheader("Retrieved Context")
        if len(relevant_docs) == 0:
            st.warning("No relevant documents found.")
        else:
            for i, doc in enumerate(relevant_docs, 1):
                with st.expander(f"Document {i}"):
                    st.write(doc.page_content)

        # Build LLM input
        combined_input = f"""Based on the following documents, answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Guidelines:
- Use only the information from the documents above.
- If the answer is not present in the documents, reply: "I don't know."
"""

        # Run model
        model = ChatOpenAI(model="gpt-4o")
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=combined_input),
        ]

        with st.spinner("Generating answer..."):
            response = model.invoke(messages)

        st.subheader("RAG Answer")
        st.write(response.content)
