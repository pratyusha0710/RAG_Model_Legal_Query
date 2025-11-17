from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.runnables import Runnable

# --- CONFIGURATION ---
DB_PATH = "legal_chroma_db"
EMBEDDING_MODEL = "all-minilm" 
LLM_MODEL = "llama3"

# --- PROMPT TEMPLATES FOR THE AGENT ---

# 1. Prompt for the Answer Generator
ANSWER_PROMPT = PromptTemplate.from_template(
    """
    <s> [Instructions] You are a legal assistant. Answer the question based ONLY on the following context. 
    If you don't know the answer, reply "No Context available for this question". [/Instructions] 

    [Instructions] Question: {input} 
    Context: {context} 
    Answer: [/Instructions]
    """
)

# 2. Prompt for the Evaluation Agent
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

# 3. Prompt for the Refinement Agent
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


# --- LANGGRAPH STATE DEFINITION ---

# Define the state that will be passed between nodes in the graph
class GraphState(TypedDict):
    input: str                 # The original user query
    context: List[Document]    # The retrieved documents
    answer: str                # The generated answer
    verdict: str               # The verdict from the evaluator ("GOOD" or "BAD")
    refined_query: str         # The new query from the refiner
    loop_count: int            # To prevent infinite loops, we'll allow one retry
    
    # --- Shared components ---
    # We add them to the state so they are accessible by all nodes
    # 'Optional' allows them to be 'None' if not set, but we'll set them.
    retriever: Optional[Any]
    llm_chain: Optional[Runnable]
    evaluator_chain: Optional[Runnable]
    refiner_chain: Optional[Runnable]

# --- NODE FUNCTIONS FOR THE GRAPH ---

def format_docs(docs: List[Document]) -> str:
    """Helper function to format documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_docs(state: GraphState) -> GraphState:
    """Retrieves documents from the vector store."""
    print("--- 1. RETRIEVING DOCUMENTS ---")
    
    # Get the query (either original or refined)
    if state.get("refined_query"):
        query = state["refined_query"]
        print(f"Using refined query: {query}")
    else:
        query = state["input"]
        print(f"Using original query: {query}")
        
    # Retrieve documents
    retriever = state["retriever"]
    docs = retriever.invoke(query)
    
    return {"context": docs}

def generate_answer(state: GraphState) -> GraphState:
    """Generates an answer using the LLM."""
    print("--- 2. GENERATING ANSWER ---")
    
    # Get components from state
    llm_chain = state["llm_chain"]
    
    # Determine which query to use
    if state.get("refined_query"):
        query = state["refined_query"]
    else:
        query = state["input"]

    # Format context
    formatted_context = format_docs(state["context"])
    
    # Generate answer
    answer = llm_chain.invoke({
        "input": query,
        "context": formatted_context
    })
    
    print(f"Generated Answer: {answer}")
    return {"answer": answer}

def evaluate_answer(state: GraphState) -> GraphState:
    """Evaluates the generated answer."""
    print("--- 3. EVALUATING ANSWER ---")
    
    # Get components from state
    evaluator_chain = state["evaluator_chain"]
    
    # Format context
    formatted_context = format_docs(state["context"])
    
    # Get verdict
    verdict = evaluator_chain.invoke({
        "input": state["input"], # Always use the original query for evaluation
        "context": formatted_context,
        "answer": state["answer"]
    }).strip().upper()
    
    print(f"Verdict: {verdict}")
    
    # Increment loop count
    loop_count = state.get("loop_count", 0) + 1
    
    return {"verdict": verdict, "loop_count": loop_count}

def refine_query(state: GraphState) -> GraphState:
    """Refines the query if the answer was "BAD"."""
    print("--- 4. REFINING QUERY ---")
    
    # Get components from state
    refiner_chain = state["refiner_chain"]
    
    # Generate refined query
    refined_query = refiner_chain.invoke({
        "input": state["input"],
        "answer": state["answer"]
    }).strip()
    
    print(f"Refined Query: {refined_query}")
    return {"refined_query": refined_query}

# --- CONDITIONAL EDGE FUNCTION ---

def should_refine(state: GraphState) -> str:
    """
    Determines whether to end the process or refine the query.
    We allow exactly ONE retry (loop_count == 1).
    """
    print("--- 5. CHECKING CONDITIONAL LOGIC ---")
    verdict = state["verdict"]
    loop_count = state["loop_count"]
    
    if verdict == "GOOD" or loop_count >= 2:
        # Either the answer is good, or we've already tried once
        print("Verdict is GOOD or loop limit reached. Ending process.")
        return "end"
    else:
        # The answer is BAD, and we haven't looped yet
        print("Verdict is BAD. Refining query.")
        return "refine"

# --- BUILD THE GRAPH ---

def build_graph():
    """Builds the LangGraph agent."""
    
    # 1. Initialize Models
    print("Loading models...")
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

    # 4. Create the chains for the agents
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    evaluator_chain = EVALUATOR_PROMPT | llm | StrOutputParser()
    refiner_chain = REFINER_PROMPT | llm | StrOutputParser()

    # 5. Define the graph
    graph = StateGraph(GraphState)

    # 6. Add nodes
    # We pass the components to the nodes using .bind()
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)
    graph.add_node("evaluate", evaluate_answer)
    graph.add_node("refine", refine_query)

    # 7. Set the entry point
    graph.set_entry_point("retrieve")

    # 8. Add edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    
    # This is the conditional edge
    graph.add_conditional_edges(
        "evaluate",          # Start node
        should_refine,       # Function to call
        {
            "refine": "refine",  # If it returns "refine", go to "refine"
            "end": END           # If it returns "end", finish
        }
    )
    
    # This creates the loop
    graph.add_edge("refine", "retrieve")

    # 9. Compile the graph
    # We provide the "shared" components that all nodes can access
    app = graph.compile(
        checkpointer=None, # No memory needed for this simple case
        # We pass the components in the `with_config` call
        # but for this structure, let's inject them at compile time
        # This is a bit advanced, let's try a simpler way...
    )
    
    # --- A simpler way to pass components ---
    # We will pass them in the input dictionary instead
    # Re-compiling the graph...
    
    app = graph.compile()

    # Create a "wrapper" function to inject dependencies
    def run_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Inject the shared components into the state
        inputs["retriever"] = retriever
        inputs["llm_chain"] = answer_chain
        inputs["evaluator_chain"] = evaluator_chain
        inputs["refiner_chain"] = refiner_chain
        inputs["loop_count"] = 0 # Initialize loop count
        
        # Use stream to get intermediate steps
        final_state = inputs.copy() 
        for s in app.stream(inputs):
            # s is a dictionary like {'retrieve': {'context': [...]}}
            # We want to un-nest this and update the main state
            
            if "__end__" not in s: # Check if it's the end state
                # Get the name of the node that just ran (e.g., "retrieve")
                node_name = list(s.keys())[0]
                # Get the output of that node (e.g., {'context': [...]})
                node_output = s[node_name]
                
                # Update the final_state with the *contents* of the node output
                final_state.update(node_output)
            
        return final_state

    return run_agent


# --- RUN THE CHAIN (INTERACTIVE MODE) ---
if __name__ == "__main__":
    try:
        # Load the chain only once
        print("Loading models and building agent graph...")
        agent_chain = build_graph()
        print("✅ System B (Agent) ready. Ask a question about the Constitution of India.")

        while True:
            # 1. Get user input
            user_query = input("\n> Ask a question (or type 'exit' to quit): ")
            
            if user_query.lower() == 'exit':
                print("Exiting. Goodbye!")
                break
            
            if not user_query.strip():
                continue

            # 2. Invoke the agent
            print("\n" + "="*50)
            print("Invoking Agent...")
            
            final_state = agent_chain({"input": user_query})
            
            print("="*50)
            print("\n--- AGENT RUN COMPLETE ---")
            
            # 3. Print the results
            print(f"\nOriginal Query: {final_state['input']}")
            if final_state.get("refined_query"):
                print(f"Refined Query: {final_state['refined_query']}")
                
            print("\nFinal Answer:", final_state["answer"])
            
            # 4. Print sources
            if "No Context" not in final_state["answer"] and final_state["context"]:
                print("\n--- Sources Used ---")
                for doc in final_state["context"]:
                    source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
                    page_num = doc.metadata.get('page', 0) + 1 # Assuming 0-based index
                    print(f"- {source_name} (Page {page_num})")
            
            print("\n" + "-"*50)

    except Exception as e:
        print(f"\nAn error occurred: {e}")