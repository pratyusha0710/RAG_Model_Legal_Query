import sys
print("--- Python Path ---")
# This prints where Python looks for files
print(sys.path[0]) 

print("\n--- Attempting Import ---")
try:
    import langchain
    print(f"✅ LangChain found at: {langchain.__file__}")
    
    from langchain.chains import create_retrieval_chain
    print("✅ Success! create_retrieval_chain is imported.")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    print("This means the 'langchain' package is installed, but the 'chains' folder inside it is missing.")