# 🩺 Derma-Agent: Agentic RAG Dermatological Assistant

An intelligent medical triage system that combines **Computer Vision (OCR)** and **Retrieval-Augmented Generation (RAG)** to identify skin irritants and provide clinical context.

## 🚀 Key Features
* **Agentic Reasoning:** Powered by Llama-3.3-70B via Groq for step-by-step triage.
* **RAG Architecture:** Uses FAISS and HuggingFace Embeddings for high-accuracy medical document retrieval.
* **Vision Integration:** EasyOCR detects ingredients in product labels for real-time hazard analysis.
* **Modular Design:** Separate data, assets, and logic layers for scalability.

## 🧠 Agentic Workflow & Architecture

Unlike a standard LLM that provides generic responses, **Derma-Agent** operates as an autonomous agent using the following loop:

1. **Perception (Vision Tool):** The agent uses `EasyOCR` to ingest physical data (product ingredient labels) from the user's environment.
2. **Context Retrieval (RAG):** It autonomously queries a `FAISS` Vector Database to find clinical guidelines relevant to the user's specific skin complaint.
3. **Multi-Step Reasoning:** The Llama-3.3-70B "Brain" synthesizes the OCR data and the RAG context. It follows a specific protocol:
    * *Self-Correction:* If symptoms are vague, it initiates a "Clarification Loop" instead of giving a remedy.
    * *Cross-Referencing:* It matches scanned ingredients against known triggers for the suspected condition.
4. **Action:** It outputs a professional triage assessment and high-risk alerts.

## 🛠️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone [your-github-link]