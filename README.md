# 🔮 MythBuster AI: The Evolution from Traditional ML to RAG

A production-ready conversational AI platform built to demonstrate the technological paradigm shift from traditional statistical Machine Learning to modern Retrieval-Augmented Generation (RAG). 

This app features an interactive multi-tab **Streamlit UI** allowing users to test and contrast how different AI architectures handle intent, reasoning, and factual boundaries when analyzing urban legends and obscure folklore.

---

## 🚀 Architectural Overview

This portfolio piece maps out three distinct phases of conversational AI development:

1. **Phase 1: Traditional ML (Intent Classifier)** * Uses **TF-IDF Vectorization** and a **Linear Support Vector Machine (LinearSVC)** via Scikit-Learn.
   * Matches user input to rigid category tags to deliver static pre-written responses. 
   * Demonstrates the brittle nature of out-of-distribution handling in classic ML.
   
2. **Phase 2: Vanilla LLM (Generative)**
   * Integrates an open-source foundational model (**Llama 3.1 8B Instant**) via the Groq API.
   * Employs prompt engineering (`system prompts`) to mold a witty, analytical persona.
   * Demonstrates human-like conversational fluidity but highlights the risk of unchecked hallucinations on niche topics.

3. **Phase 3: Retrieval-Augmented Generation (RAG)**
   * Bridges the gap by anchoring the LLM's reasoning loop to a private, localized knowledge base (`data/knowledge_base.txt`).
   * Generates text embeddings using HuggingFace's `all-MiniLM-L6-v2` and indexes them inside an in-memory **FAISS** vector store.
   * Performs real-time cosine similarity lookups, extracts context chunks, and constrains the LLM to write 100% truth-grounded answers.

---
## 🚀 How to Use the App

To protect API usage limits and costs, this application requires users to provide their own Groq API key. 

1. **Get your API Key**: Go to the [Groq Cloud Console](https://groq.com) and generate a free API key.
2. **Open the Application**: Visit the live app link here: https://mythbuster-ai-evolution-j2amprk54dahdjkepbcjpq.streamlit.app/
3. **Activate the Chatbot**: Paste your key into the **"Enter your personal Groq API Key"** input box in the left sidebar. 

*Note: Your API key is processed strictly within your own browser session and is never saved or shared.*

## 📁 Repository Structure

```text
mythbuster-ai-evolution/
│
├── data/
│   └── knowledge_base.txt      # Raw text database of verified folklore facts
│
├── notebooks/
│   └── evolution_draft.ipynb   # Google Colab experimental scratchpad 
│
├── app.py                      # Production Multi-Tab Streamlit UI 
├── requirements.txt            # Python ecosystem dependencies
└── README.md                   # System documentation
