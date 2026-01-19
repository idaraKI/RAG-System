import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from ingestion_pipeline import run_ingestion_pipeline, PERSIST_DIR

load_dotenv()

# -------------------------------
# LLM-based intent classifier
# -------------------------------
def classify_query_intent(query: str) -> str:
    """
    Returns one of:
    - INTERNAL
    - PUBLIC
    - GENERAL
    """
    prompt = f"""
Classify the user's question into ONE category.

INTERNAL:
- Company internal processes, policies, operations, documents

PUBLIC:
- Social media, follower counts, news, press, public announcements

GENERAL:
- General knowledge not specific to Rayda

Question:
{query}

Answer with ONLY one word:
INTERNAL, PUBLIC, or GENERAL
"""
    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke([
        SystemMessage(content="You classify user intent."),
        HumanMessage(content=prompt),
    ])
    return response.content.strip().upper()


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Rayda RAG System", layout="wide")
st.image("assets/rayda_logo.png", width=60)
st.title("Rayda RAG System")
st.write("Ask any question based on Rayda's internal documents.")

# -------------------------------
# Vector store bootstrapping
# -------------------------------
if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    st.info("No vector store found. Running ingestion pipeline...")
    run_ingestion_pipeline()
    st.success("Ingestion completed!")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
)

retriever = db.as_retriever(search_kwargs={"k": 7})

duckduckgo_search = DuckDuckGoSearchAPIWrapper()

# -------------------------------
# Session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.image("assets/rayda_logo.png", width=60)
st.sidebar.title("Sample Questions")

sample_questions = [
    "Why should I use Rayda?",
    "What is Rayda's device procurement process?",
    "How does Rayda manage company assets?",
    "What documents are stored in the Fixed Asset Document Manager?",
    "How are devices approved and delivered at Rayda?",
    "What happens to devices at end of life?"
]

if st.sidebar.button("Use a sample question"):
    st.session_state["prefill"] = st.sidebar.radio(
        "Pick one:", sample_questions
    )

# -------------------------------
# Display chat history
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------------------
# Chat input
# -------------------------------
user_query = st.chat_input("Ask a question about Rayda...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # -------------------------------
    # Step 1: Intent classification
    # -------------------------------
    intent = classify_query_intent(user_query)

    documents = []
    source_type = ""

    # -------------------------------
    # Step 2: Route by intent
    # -------------------------------
    if intent == "PUBLIC":
        st.info("Searching the web using DuckDuckGo...")
        web_results = duckduckgo_search.run(user_query, num_results=5)
        documents = [r.strip() for r in web_results.split("\n") if r.strip()]
        source_type = "public web sources"

    elif intent == "INTERNAL":
        docs = retriever.invoke(user_query)
        documents = [doc.page_content for doc in docs]
        source_type = "Rayda internal documents"

    else:  # GENERAL
        source_type = "general knowledge"

    # -------------------------------
    # Step 3: Build prompt
    # -------------------------------
    combined_input = f"""
You are a helpful assistant supporting Rayda.

### RULES ABOUT KNOWLEDGE USE

1. If the question is INTERNAL → use ONLY Rayda internal documents.
2. If the question is PUBLIC → you MAY use public web sources.
3. If information is missing → say so politely.
4. Never invent facts about Rayda.

### USER QUESTION:
{user_query}

SOURCE:
{source_type}

CONTENT:
{chr(10).join([f"- {doc}" for doc in documents])}
"""

    # -------------------------------
    # Step 4: Answer
    # -------------------------------
    model = ChatOpenAI(model="gpt-4o")

    with st.spinner("Generating answer..."):
        response = model.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=combined_input),
        ])

    with st.chat_message("assistant"):
        st.write(response.content)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )
