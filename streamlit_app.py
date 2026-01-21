import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from tavily import TavilyClient
from ingestion_pipeline import run_ingestion_pipeline, PERSIST_DIR

load_dotenv()

OPENAI_MODEL = "gpt-4o"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing in your .env file")
    st.stop()

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# --- INTENT CLASSIFIER ---
def classify_query_intent(query: str) -> str:
    """
    Returns: INTERNAL | PUBLIC | GENERAL
    """
    prompt = f"""
Classify the user's question into ONE category.

INTERNAL:
- Rayda internal processes, policies, operations, internal documents

PUBLIC:
- Rayda social media, follower counts, news, press, public announcements

GENERAL:
- General knowledge not specific to Rayda

Question:
{query}

Answer with ONLY ONE WORD:
INTERNAL, PUBLIC, or GENERAL
"""

    model = ChatOpenAI(model=OPENAI_MODEL)
    response = model.invoke([
        SystemMessage(content="You classify user intent."),
        HumanMessage(content=prompt),
    ])

    return response.content.strip().upper()

# --- TAVILY WEB SEARCH ---
def tavily_web_search(query: str, max_results: int = 5) -> list[str]:
    response = tavily_client.search(
        query=query,
        max_results=max_results
    )

    documents = []
    for r in response.get("results", []):
        content = r.get("content")
        if content:
            documents.append(content)

    return documents


# --- STREAMLIT UI ---
st.set_page_config(page_title="Rayda RAG System", layout="wide")
st.image("assets/rayda_logo.png", width=60)
st.title("Rayda RAG System")
st.write("Ask any question based on Rayda's internal documents.")

# --- INGESTION CHECK ---
if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    st.info("No vector store found. Running ingestion pipeline...")
    run_ingestion_pipeline()
    st.success("Ingestion completed!")

# --- LOAD VECTOR STORE ---
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
)

retriever = db.as_retriever(search_kwargs={"k": 7})


# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "prefill_input" not in st.session_state:
    st.session_state.prefill_input = ""

# --- SIDEBAR ---
st.sidebar.image("assets/rayda_logo.png", width=60)
st.sidebar.title("Sample Questions")

sample_questions = [
    "Why should I use Rayda?",
    "What is Rayda's device procurement process?",
    "How does Rayda manage company assets?",
    "What documents are stored in the Fixed Asset Document Manager?",
    "How are devices approved and delivered at Rayda?",
    "What happens to devices at end of life?",
    "What is Rayda’s LinkedIn follower count?",
    "Has Rayda been mentioned in the news recently?",
]

selected_question = st.sidebar.radio(
    "Try one of these:",
    sample_questions
)

if st.sidebar.button("Use this sample question"):
    st.session_state.prefill_input = selected_question

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat input ---
user_query = st.chat_input("Ask a question about Rayda...")

# --- INJECT SAMPLE QUESTIONS ---
if not user_query and st.session_state.prefill_input:
    user_query = st.session_state.prefill_input
    st.session_state.prefill_input = ""

# --- MAIN LOGIC ---
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # --- STEP 1: INTENT CLASSIFICATION ---
    intent = classify_query_intent(user_query)

    documents = []
    source_type = ""

   # --- STEP 2: ROUTE BY INTENT ---
    if intent == "INTERNAL":
        docs = retriever.invoke(user_query)
        documents = [doc.page_content for doc in docs]
        source_type = "Rayda internal documents"

    elif intent == "PUBLIC":
        st.info("Searching the web ...")
        documents = tavily_web_search(user_query)
        source_type = "Public web sources (Tavily)"

    else:  # GENERAL
        source_type = "General knowledge"

   # --- STEP 3: BUILD PROMPT ---
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

    # --- STEP 4: ANSWER ---
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
