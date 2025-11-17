from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import operator

# --- CONFIGURATION ---
DB_PATH = "legal_chroma_db"
EMBEDDING_MODEL = "all-minilm" 
LLM_MODEL = "llama3"

# --- ROBUST IMPORT BLOCK (The "Bypass") ---
try:
    # 1. Try the standard import
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    print("⚠️ Standard import failed. Switching to Manual Chain Definition (Safe Mode).")

    # 2. Fallback: We define the logic using standard Python functions
    # This is "Bulletproof" because it avoids complex chain logic that might fail
    
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

def get_qa_chain():
    # 1. Initialize Models (With Safety Fixes)
    print("Loading models...")
    
    # Force base_url to avoid port errors
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434"
    )
    
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url="http://localhost:11434"
    )

    # 2. Load the EXISTING Vector DB
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 3. Create Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. Create the Prompt Template
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a legal assistant. Answer the question based ONLY on the following context. 
        If you don't know the answer, reply "No Context available for this question". [/Instructions] 
        
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]
        """
    )

    # 5. Build the Chain (This handles both Standard and Safe Mode)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- Run the Chain ---
# --- Run the Chain (Interactive Mode) ---
if __name__ == "__main__":
    try:
        # Load the chain only once
        print("Loading models and preparing chain...")
        chain = get_qa_chain()
        print("✅ System ready. Ask a question about the Constitution of India.")

        while True:
            # 1. Get user input
            user_query = input("\n> Ask a question (or type 'exit' to quit): ")
            
            if user_query.lower() == 'exit':
                print("Exiting. Goodbye!")
                break
            
            if not user_query.strip():
                continue

            # 2. Invoke the chain with the user's query
            print("Thinking...")
            response = chain.invoke({"input": user_query})
            
            # 3. Print the results
            print("\nAnswer:", response["answer"])
            
            # Only print sources if the answer wasn't a "no context" fallback
            if "No Context" not in response["answer"] and response["context"]:
                print("\n--- Sources Used ---")
                for doc in response["context"]:
                    source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
                    page_num = doc.metadata.get('page', 0) + 1 # Assuming 0-based index
                    print(f"- {source_name} (Page {page_num})")
            
            print("\n" + "-"*50)

    except Exception as e:
        print(f"\nAn error occurred: {e}")