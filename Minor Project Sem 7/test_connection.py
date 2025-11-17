import requests

try:
    print("Testing connection to Ollama at port 11434...")
    response = requests.post(
        "http://127.0.0.1:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": "hello world"}
    )
    if response.status_code == 200:
        print("SUCCESS! Ollama is alive and embedding works.")
    else:
        print(f"FAILED. Status Code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"CRITICAL FAILURE: Could not connect. {e}")