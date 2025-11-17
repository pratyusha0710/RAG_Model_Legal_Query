RAG-Based Legal Query Assistant

This project investigates and compares two different Retrieval-Augmented Generation (RAG) architectures for answering questions based on legal documents. The primary goal is to analyze the trade-offs between a "Naive RAG" system and a more complex "Agentic RAG" system that includes a self-correction loop.

This system is built entirely with open-source models and libraries.

System A (Naive RAG): A standard, linear RAG pipeline.
Query -> Retrieve -> Generate -> Answer

System B (Simple Agent RAG): An experimental agentic model inspired by self-correction principles.
Query -> Retrieve -> Generate -> Evaluate (Is answer "GOOD" or "BAD"?)

If "GOOD" -> Return Answer

If "BAD" -> Refine Query -> Retrieve -> Generate -> Return Answer

Tech Stack

LLM: Ollama (using llama3)

Embeddings: Ollama (using all-minilm)

Vector Database: ChromaDB

Orchestration: LangChain

Agent Logic: LangGraph

Frontend: Streamlit

1. Setup & Installation

Install Ollama:
You must have Ollama installed and running on your local machine.

Pull the required models:

ollama pull llama3
ollama pull all-minilm


Clone the repository and install dependencies:

git clone [YOUR_REPO_URL]
cd [YOUR_REPO_NAME]
pip install -r requirements.txt


(Note: You will need to create a requirements.txt file. Based on your scripts, it should include langchain, langchain-chroma, langchain-ollama, langgraph, streamlit)

2. Ingestion (Mandatory First Step)

Before running any system, you must create the vector database from your legal documents.

Place your document (e.g., Constitution of India.pdf) in the data/ directory.

Run the ingestion script (assuming you have one named ingest.py):

python ingest.py


This will create the legal_chroma_db directory containing the vector embeddings.

3. How to Run the Systems

You can run each system individually from the command line or use the comparison app.

A. System A (Naive RAG) - CLI

This runs the simple, naive RAG pipeline in your terminal.

python system_a.py


B. System B (Simple Agent RAG) - CLI

This runs the self-correcting agent in your terminal. You will see the agent's internal "thought process," including its "Verdict" and any "Refined Queries."

python system_b.py


C. Side-by-Side Comparison App

This is the main evaluation tool. It runs both System A and System B on the same query and displays the results, latency, and sources side-by-side.

streamlit run app_comparison.py


Example Test Queries

You can use the following questions (from rag_test_suite.md) to test the performance of both systems in the app_comparison.py interface.

Specific Factual Questions (Baseline)

What is Article 17 of the Constitution?

What does Article 21A guarantee?

What is the procedure for the impeachment of the President as described in Article 61?

Vague or Ambiguous Questions (To test System B's refinement)

What does the constitution say about equality?

What are the duties of the government?

Tell me about the powers of "the State".

"Trick" or Nuanced Questions

What happens to my Fundamental Rights during a Proclamation of Emergency?

Can Parliament make a law on a subject in the State List?

What is the right to property?

Out-of-Scope Questions (To test resilience)

What is the penalty for theft under the Indian Penal Code?

What is the procedure for filing an FIR (First Information Report)?
