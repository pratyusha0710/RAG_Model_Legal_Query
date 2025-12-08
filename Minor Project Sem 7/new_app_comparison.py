import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import os
import time
import csv
from datetime import datetime

# --- CONFIGURATION ---
DB_PATH = "legal_chroma_db"
EMBEDDING_MODEL = "all-minilm"
LLM_MODEL = "llama3"
CSV_FILE = "experiment_results.csv"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAG System Comparison",
    page_icon="⚖️",
    layout="wide"
)
st.title("⚖️ RAG System Comparison")
st.caption("A side-by-side evaluation of 'Naive RAG' (System A) vs. 'Simple Agent RAG' (System B).")

# --- LOGGING FUNCTION ---
def log_to_csv(query, sys_a_latency, sys_a_answer, sys_b_latency, sys_b_answer, sys_b_verdict):
    """Logs the experiment results to a CSV file."""
    file_exists = os.path.isfile(CSV_FILE)
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header if file is new
            if not file_exists:
                writer.writerow(['Timestamp', 'Query', 'Sys_A_Latency_Sec', 'Sys_A_Answer', 'Sys_B_Latency_Sec', 'Sys_B_Answer', 'Sys_B_Verdict'])
            
            # Write data row
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                query, 
                f"{sys_a_latency:.2f}", 
                sys_a_answer.replace("\n", " "), # Clean newlines for CSV
                f"{sys_b_latency:.2f}", 
                sys_b_answer.replace("\n", " "), 
                sys_b_verdict
            ])
        return True
    except Exception as e:
        print(f"Error logging to CSV: {e}")
        return False

# ==============================================================================
# --- SYSTEM A: "NAIVE RAG" ---
# ==============================================================================

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    # Fallback for manual chain definition
    def format_docs_std(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_stuff_documents_chain(llm, prompt):
        return prompt | llm | StrOutputParser()

    def create_retrieval_chain(retriever, combine_docs_chain):
        def run_rag(input_dict):
            query = input_dict["input"]
            docs = retriever.invoke(query)
            formatted_docs = format_docs_std(docs)
            answer = combine_docs_chain.invoke({
                "context": formatted_docs,
                "input": query
            })
            return {
                "answer": answer,
                "context": docs,
                "input": query
            }
        return RunnableLambda(run_rag)

@st.cache_resource
def get_system_a_chain():
    """Loads and returns the Naive RAG chain (System A)."""
    print("Loading System A...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
    llm = ChatOllama(model=LLM_MODEL, base_url="http://localhost:11434")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a legal assistant. Answer the question based ONLY on the following context. 
        If you don't know the answer, reply "No Context available for this question". [/Instructions] 
        
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]
        """
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# ==============================================================================
# --- SYSTEM B: "SIMPLE AGENT RAG" ---
# ==============================================================================

# --- PROMPT TEMPLATES ---
ANSWER_PROMPT = PromptTemplate.from_template(
    """
    <s> [Instructions] You are a legal assistant. Answer the question based ONLY on the following context. 
    If you don't know the answer, reply "No Context available for this question". [/Instructions] 

    [Instructions] Question: {input} 
    Context: {context} 
    Answer: [/Instructions]
    """
)
EVALUATOR_PROMPT = PromptTemplate.from_template(
    """
    <s> [Instructions] You are an Evaluation Agent. Your goal is to assess the quality of a generated answer
    based on the retrieved context and the user's query.
    
    [Context]
    {context}
    [Query]
    {input}
    [Answer]
    {answer}
    
    [Instructions]
    Read the query, context, and answer carefully. 
    The answer MUST be grounded in the context.
    If the answer is relevant, complete, and factually correct based on the context, return "GOOD".
    If the answer is "No Context available for this question", or if it is vague, irrelevant, or hallucinates, return "BAD".
    
    Return only the single word verdict: "GOOD" or "BAD".
    Verdict: [/Instructions]
    """
)
REFINER_PROMPT = PromptTemplate.from_template(
    """
    <s> [Instructions] You are a Query Refinement Agent. 
    The user's original query produced a "BAD" answer. 
    Your goal is to rephrase the original query to be more specific and relevant, 
    so that the retrieval step can find better documents.
    
    [Original Query]
    {input}
    [Original (BAD) Answer]
    {answer}
    
    [Instructions]
    Output ONLY the new, refined query. Do not add any explanation.
    Refined Query: [/Instructions]
    """
)

# --- LANGGRAPH STATE ---
class GraphState(TypedDict):
    input: str
    context: List[Document]
    answer: str
    verdict: str
    refined_query: str
    loop_count: int
    retriever: Optional[Any]
    llm_chain: Optional[Runnable]
    evaluator_chain: Optional[Runnable]
    refiner_chain: Optional[Runnable]

# --- NODE FUNCTIONS ---
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_docs(state: GraphState) -> GraphState:
    if state.get("refined_query"):
        query = state["refined_query"]
    else:
        query = state["input"]
    retriever = state["retriever"]
    docs = retriever.invoke(query)
    return {"context": docs}

def generate_answer(state: GraphState) -> GraphState:
    llm_chain = state["llm_chain"]
    if state.get("refined_query"):
        query = state["refined_query"]
    else:
        query = state["input"]
    formatted_context = format_docs(state["context"])
    answer = llm_chain.invoke({"input": query, "context": formatted_context})
    return {"answer": answer}

def evaluate_answer(state: GraphState) -> GraphState:
    evaluator_chain = state["evaluator_chain"]
    formatted_context = format_docs(state["context"])
    verdict = evaluator_chain.invoke({
        "input": state["input"],
        "context": formatted_context,
        "answer": state["answer"]
    }).strip().upper()
    loop_count = state.get("loop_count", 0) + 1
    return {"verdict": verdict, "loop_count": loop_count}

def refine_query(state: GraphState) -> GraphState:
    refiner_chain = state["refiner_chain"]
    refined_query = refiner_chain.invoke({
        "input": state["input"],
        "answer": state["answer"]
    }).strip()
    return {"refined_query": refined_query}

def should_refine(state: GraphState) -> str:
    verdict = state["verdict"]
    loop_count = state["loop_count"]
    if verdict == "GOOD" or loop_count >= 2:
        return "end"
    else:
        return "refine"

@st.cache_resource
def get_system_b_chain():
    """Builds and returns the Agentic RAG chain (System B)."""
    print("Loading System B...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
    llm = ChatOllama(model=LLM_MODEL, base_url="http://localhost:11434")
    retriever = Chroma(persist_directory=DB_PATH, embedding_function=embeddings).as_retriever(search_kwargs={"k": 3})

    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    evaluator_chain = EVALUATOR_PROMPT | llm | StrOutputParser()
    refiner_chain = REFINER_PROMPT | llm | StrOutputParser()

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)
    graph.add_node("evaluate", evaluate_answer)
    graph.add_node("refine", refine_query)
    
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", should_refine, {"refine": "refine", "end": END})
    graph.add_edge("refine", "retrieve")

    app = graph.compile()

    def run_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper to inject components and run the graph."""
        inputs["retriever"] = retriever
        inputs["llm_chain"] = answer_chain
        inputs["evaluator_chain"] = evaluator_chain
        inputs["refiner_chain"] = refiner_chain
        inputs["loop_count"] = 0
        
        final_state = inputs.copy()
        for s in app.stream(inputs):
            if "__end__" not in s:
                node_name = list(s.keys())[0]
                node_output = s[node_name]
                final_state.update(node_output)
        return final_state

    return run_agent

# ==============================================================================
# --- MAIN APP UI ---
# ==============================================================================

# Load both systems
try:
    if not os.path.exists(DB_PATH):
        st.error(f"Error: Database not found at '{DB_PATH}'. Please run `python ingest.py` first.")
    else:
        system_a_chain = get_system_a_chain()
        system_b_chain = get_system_b_chain()

        query = st.text_input("Enter your legal query:", placeholder="e.g., What are the rights of a consumer?")

        if st.button("Run Comparison"):
            if query and system_a_chain and system_b_chain:
                st.divider()
                
                col1, col2 = st.columns(2)
                
                # --- RUN SYSTEM A ---
                with col1:
                    st.header("🤖 System A (Naive RAG)")
                    with st.spinner("System A is thinking..."):
                        start_time = time.time()
                        result_a = system_a_chain.invoke({"input": query})
                        end_time = time.time()
                        latency_a = end_time - start_time
                    
                    st.write(f"**Answer:** (Latency: {latency_a:.2f}s)")
                    st.info(result_a["answer"])
                    
                    with st.expander("Show Sources (System A)"):
                        st.write("The following chunks were used as context:")
                        for i, doc in enumerate(result_a["context"]):
                            st.divider()
                            source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                            page = doc.metadata.get('page', 0) + 1
                            st.write(f"**Source {i+1}:** {source} (Page {page})")
                            st.caption(doc.page_content)

                # --- RUN SYSTEM B ---
                with col2:
                    st.header("🧠 System B (Agent RAG)")
                    with st.spinner("System B is thinking..."):
                        start_time = time.time()
                        result_b = system_b_chain({"input": query})
                        end_time = time.time()
                        latency_b = end_time - start_time
                    
                    st.write(f"**Final Answer:** (Latency: {latency_b:.2f}s)")
                    st.success(result_b["answer"])
                    
                    # Display the agent's "thoughts"
                    st.subheader("Agent's Internal Process")
                    st.markdown(f"**Internal Verdict:** `{result_b['verdict']}`")
                    if "refined_query" in result_b:
                        st.markdown(f"**Refined Query:** `{result_b['refined_query']}`")
                    else:
                        st.markdown("_No query refinement was needed._")

                    with st.expander("Show Sources (System B)"):
                        st.write("The following chunks were used as context for the *final* answer:")
                        for i, doc in enumerate(result_b["context"]):
                            st.divider()
                            source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                            page = doc.metadata.get('page', 0) + 1
                            st.write(f"**Source {i+1}:** {source} (Page {page})")
                            st.caption(doc.page_content)

                # --- LOGGING ---
                if log_to_csv(query, latency_a, result_a["answer"], latency_b, result_b["answer"], result_b.get("verdict", "N/A")):
                    st.toast("✅ Results logged to CSV!")
                else:
                    st.error("Failed to log results.")

            else:
                st.warning("Please enter a query.")

except Exception as e:
    st.error(f"An error occurred while loading the systems. Is Ollama running?\n\nError: {e}")