import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

# --- CONFIGURATION ---
DB_PATH = "legal_chroma_db"
EMBEDDING_MODEL = "all-minilm" # Using the model from your app.py
LLM_MODEL = "llama3"

# --- ROBUST IMPORT BLOCK (Copied from system_a.py) ---
try:
    # 1. Try the standard import
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    print("✅ Standard LangChain imports successful.")
except ImportError:
    print("⚠️ Standard import failed. Switching to Manual Chain Definition (Safe Mode).")

    # 2. Fallback: We define the logic using standard Python functions
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_stuff_documents_chain(llm, prompt):
        # Simple chain: Prompt -> LLM -> String
        return prompt | llm | StrOutputParser()

    def create_retrieval_chain(retriever, combine_docs_chain):
        # We create a custom function that orchestrates the whole flow
        def run_rag(input_dict):
            query = input_dict["input"]
            
            # 1. Retrieve Documents (Pass ONLY the string query)
            docs = retriever.invoke(query)
            
            # 2. Format Documents
            formatted_docs = format_docs(docs)
            
            # 3. Generate Answer
            answer = combine_docs_chain.invoke({
                "context": formatted_docs,
                "input": query
            })
            
            # 4. Return the dictionary structure expected by your UI
            return {
                "answer": answer,
                "context": docs,
                "input": query
            }
            
        # Wrap it in RunnableLambda so it acts like a standard Chain
        return RunnableLambda(run_rag)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Naive RAG (System A)",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 Naive RAG (System A)")
st.caption("This system demonstrates the 'Naive RAG' architecture. It retrieves context from a pre-built legal database and generates an answer.")

# --- LOAD MODELS AND DATABASE (Cached) ---
@st.cache_resource
def load_system():
    print("Loading models and database...")
    
    # Check if DB exists
    if not os.path.exists(DB_PATH):
        st.error(f"Error: Database not found at '{DB_PATH}'. Please run `python ingest.py` first.")
        return None

    # Initialize components (with base_url fix)
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434"
    )
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url="http://localhost:11434"
    )
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create the Prompt Template
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a legal assistant. Answer the question based ONLY on the following context. 
        If you don't know the answer, reply "No Context available for this question". [/Instructions] 
        
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]
        """
    )
    
    # Create the chain (This will now use the "Safe Mode" if needed)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("Models and database loaded successfully.")
    return retrieval_chain

try:
    qa_chain = load_system()
except Exception as e:
    st.error(f"Failed to load models. Is Ollama running? Error: {e}")
    qa_chain = None

# --- USER INTERFACE ---
if qa_chain:
    query = st.text_input("Enter your legal query:", placeholder="What does the constitution say about equality?")

    if st.button("Submit Query"):
        if query:
            with st.spinner("Processing your query... This may take a moment."):
                try:
                    # Run the RAG chain
                    result = qa_chain.invoke({"input": query})
                    
                    # --- Display the Results ---
                    st.header("Answer")
                    st.write(result["answer"])
                    
                    # --- Display the Source Documents (for your evaluation) ---
                    with st.expander("Show Source Documents"):
                        st.write("The following chunks were used as context to generate the answer:")
                        for i, doc in enumerate(result["context"]):
                            st.divider()
                            source = doc.metadata.get('source', 'Unknown Source').split('/')[-1] # Clean up path
                            page = doc.metadata.get('page', 0) # Use 0 as default
                            st.write(f"**Source {i+1}:** {source} (Page {page + 1})") # Add 1 for display
                            st.info(doc.page_content)
                            
                except Exception as e:
                    st.error(f"An error occurred while processing your query: {e}")
        else:
            st.warning("Please enter a query.")
else:
    st.warning("The RAG system is not loaded. Please check the errors above.")