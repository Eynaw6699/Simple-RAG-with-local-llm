from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print("Generating embedding ...")
    return embeddings
