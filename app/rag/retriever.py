import os
from langchain_chroma import Chroma
from app.rag.ingest import SolarEmbeddingFunction

from langsmith import traceable

CHROMA_DIR = "chroma_db"

_db = None


def _get_db():
    global _db
    if _db is None:
        if not os.path.exists(CHROMA_DIR):
            return None
        # SolarEmbeddingFunction used instead of HuggingFaceEmbeddings
        _db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=SolarEmbeddingFunction()
        )
    return _db

@traceable(name="rag_retrieve_context", run_type="retriever")
def retrieve_context(features, k=3):
    """
    Build a natural language query from image features and retrieve
    the top-k relevant inspection knowledge chunks from Chroma.
    """
    db = _get_db()

    query_parts = []
    if features.get("white_ratio", 0) > 0.5:
        query_parts.append("snow coverage white panel")
    if features.get("edge_density", 0) > 0.2:
        query_parts.append("physical damage cracks edge detection")
    if features.get("brightness", 255) < 100:
        query_parts.append("dust accumulation low brightness")
    if features.get("texture_variance", 0) > 1000:
        query_parts.append("electrical damage texture variance hot spot")
    if not query_parts:
        query_parts.append("clean solar panel normal operation")

    query = " ".join(query_parts)

    if db is None:
        return _rule_based_context(features)

    # Chroma similarity_search is unchanged — still does vector search
    docs = db.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def _rule_based_context(features):
    """Fallback when Chroma not yet initialized."""
    knowledge = {
        "Dusty": "Low brightness usually indicates dust accumulation on panel surface.",
        "Snow-Covered": "High white ratio indicates snow coverage reducing panel output.",
        "Electrical-Damage": "High texture variance indicates possible electrical faults or hot spots.",
        "Physical-Damage": "High edge density suggests cracks, micro-cracks, or physical breaks."
    }
    context = []
    if features.get("white_ratio", 0) > 0.5:
        context.append(knowledge["Snow-Covered"])
    if features.get("edge_density", 0) > 0.2:
        context.append(knowledge["Physical-Damage"])
    if features.get("brightness", 255) < 100:
        context.append(knowledge["Dusty"])
    return context