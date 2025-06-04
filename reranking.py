from sentence_transformers import CrossEncoder
import logging
LOGGER = logging.getLogger(__name__)    

# Load the reranker model
reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank_documents(query: str, docs_with_scores: list[tuple]):
    print("Reranking documents...")
    """
    Reranks a list of (doc, score) tuples using cross-encoder relevance scoring.
    
    Parameters:
    - query: The input user question.
    - docs_with_scores: List of (Document, score) tuples from vector similarity search.
    
    Returns:
    - List of (Document, rerank_score) sorted by descending relevance.
    """
    docs = [doc.page_content for doc, _ in docs_with_scores]
    pairs = [(query, doc) for doc in docs]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(zip(docs_with_scores, rerank_scores), key=lambda x: x[1], reverse=True)
    return [(doc, score) for ((doc, _), score) in reranked]
