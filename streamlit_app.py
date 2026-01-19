import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from ingestion_pipeline import run_ingestion_pipeline, PERSIST_DIR

load_dotenv()


# --- decide if internal docs are relevant ---
def docs_are_relevant(query: str, documents: list[str]) -> bool:
    """Simple heuristic: docs are relevant if they have enough text."""
    if not documents:
        return False

    combined_text = " ".join(documents).strip()

    # If docs are too short, they are probably not useful
    if len(combined_text) < 300:
        return False

    return True

# Streamlit UI 
st.set_page_config(page_title="Rayda RAG System", layout="wide")
st.image("assets/rayda_logo.png", width=60)
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

 # --- DuckDuckGo Search ---
duckduckgo_search = DuckDuckGoSearchAPIWrapper()


# --- Session State (Chat Memory) ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prefill_input" not in st.session_state:
    st.session_state.prefill_input = ""

# --- Side bar ---
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

selected_question = st.sidebar.radio(
    "Try one of these:",
    sample_questions
)

# --- Add a button to fill the input with a sample question without auto-sending ---
if st.sidebar.button("Use this sample question"):
    st.session_state.prefill_input = selected_question

# --- Display previous chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# --- Chat Input ---
user_query = st.chat_input(
    "Ask a question about Rayda...",
    key="user_input"
)

# If user_query is empty, use the prefilled input from the sample question
if not user_query and st.session_state.prefill_input:
    user_query = st.session_state.prefill_input
    # Clear the prefill so it doesn't keep triggering
    st.session_state.prefill_input = ""

#--- Input Query ---
if user_query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Step 1: Retrieve internal docs
    docs = retriever.invoke(user_query)
    documents = [doc.page_content for doc in docs]
    web_sources = []

    # Step 2: Decide whether to search web
    use_web = not docs_are_relevant(documents)

    # Step 3: Web search if needed
    if use_web:
        st.info("Searching the web using DuckDuckGo...")
        web_results = duckduckgo_search.run(user_query, num_results=5)

        documents = []
        for r in web_results.split("\n"):
            if "(" in r and r.endswith(")"):
                content, url = r.rsplit("(", 1)
                documents.append(content.strip())
                web_sources.append(url.rstrip(")"))
            else:
                documents.append(r.strip())
        source_type = "public web sources"
    else:
        source_type = "Rayda internal documents"


    # --- Build LLM input ---
    combined_input = f"""
You are a helpful assistant supporting Rayda.

### RULES ABOUT KNOWLEDGE USE

1. If the user's question is about Rayda’s INTERNAL operations, processes, policies, or proprietary information,
   ONLY use Rayda internal documents.

2. If the user's question is about Rayda but relates to PUBLIC information
   (e.g. social media, news, press mentions, follower counts, public announcements),
   you MAY use public web sources.

3. If the documents and web sources do not contain the answer,
   respond politely without guessing.

4. Never invent or guess details about Rayda.

5. Your tone should be professional, friendly, and conversational.

### USER QUESTION:
{user_query}

SOURCE:
{source_type}

Documents:
{chr(10).join([f"- {doc}" for doc in documents])}
"""

    # --- Run LLM ---
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
