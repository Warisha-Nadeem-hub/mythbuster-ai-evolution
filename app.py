import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==========================================
st.set_page_config(page_title="MythBuster AI Evolution", page_icon="🔮", layout="wide")
st.title("🔮 MythBuster AI: From Traditional ML to RAG")
st.write("Switch between tabs below to see how AI architecture evolved over time!")

# Securely grab the Groq API key from environment variables or Streamlit secrets


# Securely grab the Groq API key checking Streamlit secrets first, then environment, then fallback to sidebar
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") or st.sidebar.text_input("Enter Groq API Key:", type="password")

# ==========================================
# 2. DATASET & BACKEND SETUP (Cached for speed)
# ==========================================
@st.cache_resource
def initialize_backends():
    # Phase 1 Data & Model
    myth_data = {
        "intent": ["chupacabra", "chupacabra", "chupacabra", "bermuda_triangle", "bermuda_triangle", "bermuda_triangle", "bloody_mary", "bloody_mary", "bloody_mary", "slenderman", "slenderman", "slenderman"],
        "user_phrase": ["Have you heard about the Chupacabra?", "Is the Chupacabra real?", "Tell me about the creature attacking livestock.", "Why do planes disappear in the Bermuda Triangle?", "Is there a mystery behind the Bermuda ships?", "Bermuda Triangle history", "What happens if I say Bloody Mary in the mirror?", "Is the Bloody Mary mirror ghost real?", "Tell me the legend of the mirror witch", "Who is Slenderman?", "Is Slenderman real or a myth?", "Tell me about the tall man in the suit."]
    }
    df = pd.DataFrame(myth_data)
    ml_model = make_pipeline(TfidfVectorizer(stop_words='english'), LinearSVC(random_state=42))
    ml_model.fit(df['user_phrase'], df['intent'])
    
    ml_responses = {
        "chupacabra": "🔴 DEBUNKED (Traditional ML): Mange-ridden coyotes.",
        "bermuda_triangle": "🔴 DEBUNKED (Traditional ML): Standard ocean traffic risk percentages.",
        "bloody_mary": "🔴 DEBUNKED (Traditional ML): Psychological Troxler's Fading effect.",
        "slenderman": "🔴 DEBUNKED (Traditional ML): 2009 Something Awful forum photoshop contest creation."
    }

    # Phase 3 RAG Knowledge Base - Dynamically loaded from text file
    knowledge_base_path = os.path.join("data", "knowledge_base.txt")
    
    if os.path.exists(knowledge_base_path):
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            # Read each line, remove empty spaces, and ignore empty lines
            knowledge_base = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Fallback security default in case the file path isn't found locally during initial tests
        knowledge_base = [
            "Myth: Polybius Arcade Game. Fact: An urban legend claims a 1981 arcade game caused amnesia and insomnia in players before vanishing. In reality, no cabinets or ROMs exist, and it was tracked back to a hoax origin on the website Coinop.org in the early 2000s."
        ]
    
    class LangChainEmbeddingsWrapper(Embeddings):
        def __init__(self): self.model = SentenceTransformer('all-MiniLM-L6-v2')
        def embed_documents(self, texts): return self.model.encode(texts).tolist()
        def embed_query(self, text): return self.model.encode(text).tolist()

    vector_db = FAISS.from_texts(texts=knowledge_base, embedding=LangChainEmbeddingsWrapper())
    
    return ml_model, ml_responses, vector_db

ml_model, ml_responses, vector_db = initialize_backends()

# ==========================================
# 3. INTERACTIVE MULTI-TAB CHAT INTERFACE
# ==========================================
tab1, tab2, tab3 = st.tabs(["Phase 1: Traditional ML", "Phase 2: Vanilla LLM", "Phase 3: RAG Architecture"])

# --- TAB 1: TRADITIONAL ML ---
with tab1:
    st.header("Phase 1: Intent Classification (Scikit-Learn)")
    st.caption("This bot pattern-matches keywords. It works well for known questions but breaks on new topics.")
    ml_input = st.text_input("Ask Phase 1 Bot a myth question:", key="ml_in")
    if ml_input:
        pred = ml_model.predict([ml_input])[0]
        st.info(f"**Predicted Intent Tag:** `{pred}`")
        st.success(f"**Bot:** {ml_responses[pred]}")

# --- TAB 2: VANILLA LLM ---
with tab2:
    st.header("Phase 2: Generative Core (Llama 3.1)")
    st.caption("Fluent and highly intelligent, but prone to hallucinations if asked about highly niche folklore details.")
    llm_input = st.text_input("Ask Phase 2 Bot anything:", key="llm_in")
    if llm_input:
        if not GROQ_API_KEY:
            st.error("Please enter a Groq API key in the sidebar.")
        else:
            client = Groq(api_key=GROQ_API_KEY)
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a witty, analytical Myth-Buster AI."},
                    {"role": "user", "content": llm_input}
                ],
                temperature=0.7
            )
            st.success(f"**Bot:** {res.choices[0].message.content}")

# --- TAB 3: RAG ARCHITECTURE ---
with tab3:
    st.header("Phase 3: Retrieval-Augmented Generation (FAISS + Llama)")
    st.caption("The modern standard. It queries a vector database first, then forces the LLM to summarize only verified data.")
    rag_input = st.text_input("Ask Phase 3 Bot about 'Polybius', 'Flatwoods', or 'Black Eyed Children':", key="rag_in")
    if rag_input:
        if not GROQ_API_KEY:
            st.error("Please enter a Groq API key in the sidebar.")
        else:
            # 1. Retrieve
            search_results = vector_db.similarity_search(rag_input, k=1)
            context = search_results[0].page_content if search_results else "No records found."
            st.warning(f"🎯 **Vector DB Retrieved Context Chunk:** {context}")
            
            # 2. Generate
            client = Groq(api_key=GROQ_API_KEY)
            rag_prompt = f"Use this context to answer: Context:\n{context}\n\nQuestion: {rag_input}\nAnswer:"
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": rag_prompt}],
                temperature=0.2
            )
            st.success(f"**Bot:** {res.choices[0].message.content}")