import streamlit as st
import os
from ingestion_pipeline import load_documents, split_documents, create_vector_store
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


# --- Load Vector Store ---
PERSIST_DIR = "db/chroma_db"

# --- making sure that the ingestion_pipeline runs if the chroma db is not available ---
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    docs = load_documents("docs")
    chunks = split_documents(docs)
    create_vector_store(chunks, persist_directory=PERSIST_DIR)

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
        retriever = db.as_retriever(search_kwargs={"k": 2})
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
        combined_input = f"""You are a helpful assistant supporting Rayda.

### RULES ABOUT KNOWLEDGE USE
1. **If the user's question is about Rayda, its processes, policies, operations, or anything internal**, 
   ONLY use the Rayda documents provided below.

2. **If the documents do not contain the answer:**
   Respond exactly with:
   "I currently do not have that information in the company documents."

3. **If the user's question is general (NOT Rayda-specific):**
   You may answer using your own general knowledge.

4. **Never invent or guess details about Rayda that are not in the documents.**

5. Your tone should be professional, friendly, and conversational.

### USER QUESTION:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Guidelines:
- First determine whether the question is Rayda-related or general.
- If Rayda-related → use ONLY the documents.
- If not Rayda-related → answer normally.
- If documents lack the answer → give the fallback message above.
"""

# Guidelines:
# - Use only the information from the documents above.
# - If the answer is not present in the documents, reply: "I don't know."
# """

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
