"""
Builds the Chroma vector store from your docs/ folder.
Run once: python -m app.rag.ingest
Re-run any time you update docs/.
"""

import os
import re
import math
from typing import List

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_chroma import Chroma

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"


# ── SOLAR EMBEDDING FUNCTION ─────────────────────────────────────────────────
#
# Replaces: HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#
# What it does:
# Takes a list of text strings and returns a list of float vectors.
# Each vector has one dimension per term in VOCAB (32 total).
# Each dimension = TF-IDF weight: how often that term appears in this text,
# multiplied by an IDF weight that boosts rare/important terms.
# The vector is L2-normalised so Chroma's cosine similarity works correctly.
#
# Why this works for this domain:
# The inspection vocabulary is small and controlled. Every defect type maps
# directly to a few key terms (edge_density → crack/damage, white_ratio → snow,
# brightness → dust). A 32-dim domain-specific vector captures the same
# discriminative signal as a 384-dim BERT vector for this task.



class SolarEmbeddingFunction(EmbeddingFunction):

    VOCAB = [
        "dust", "dusty", "soiling", "brightness", "dark", "low", "dirty",
        "snow", "white", "coverage", "frozen", "bright",
        "crack", "edge", "damage", "physical", "break", "fracture", "chip",
        "electrical", "texture", "variance", "hotspot", "delamination", "fault",
        "clean", "normal", "healthy", "good", "operational",
        "defect", "inspection"
    ]

    IDF = {
        "crack": 2.0, "fracture": 2.0, "chip": 2.0,
        "electrical": 2.0, "hotspot": 2.0, "delamination": 2.0,
        "fault": 1.8, "damage": 1.8, "break": 1.8,
        "variance": 1.5, "texture": 1.5,
        "soiling": 1.5, "frozen": 1.5,
        "dust": 1.2, "dusty": 1.2, "snow": 1.2,
        "edge": 1.2, "physical": 1.2,
    }
    DEFAULT_IDF = 1.0

    def __init__(self):
        self._clean = re.compile(r"[^a-z0-9\s]")

    # ── Chroma native interface ──
    def __call__(self, input: Documents) -> Embeddings:
        return [self._encode(doc) for doc in input]

    # ── LangChain interface (what langchain_community.Chroma needs) ──
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._encode(text)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = self._clean.sub(" ", text)
        return [t for t in text.split() if len(t) > 1]

    def _encode(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        total = len(tokens) if tokens else 1
        vec = []
        for term in self.VOCAB:
            count = sum(1 for t in tokens if term in t or t in term)
            tf = count / total
            idf = self.IDF.get(term, self.DEFAULT_IDF)
            vec.append(tf * idf)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

# ── INGEST ────────────────────────────────────────────────────────────────────

# ── INGEST ────────────────────────────────────────────────────────────────────

def ingest_documents():
    if not os.path.exists(DOCS_DIR):
        print("[ingest] No docs/ folder found. Creating with sample content.")
        os.makedirs(DOCS_DIR)
        _write_sample_docs()

    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    if not docs:
        print("[ingest] No .txt files found. Writing sample docs.")
        _write_sample_docs()
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # SolarEmbeddingFunction used instead of HuggingFaceEmbeddings
    embedding_fn = SolarEmbeddingFunction()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=CHROMA_DIR,
        collection_name="solar_inspection_docs"   # explicit collection name
    )

    # NO db.persist() needed anymore — Chroma auto-persists

    print(f"[ingest]  Successfully stored {len(chunks)} chunks into Chroma")
    print(f"[ingest] Location: {CHROMA_DIR}/")
    print(f"[ingest] Collection: solar_inspection_docs")


def _write_sample_docs():
    solar_guide = """
Solar Panel Inspection Guide

Defect Types:

1. Physical Damage
Cracks, chips, or breaks in the panel surface. Indicated by high edge density
in aerial imagery. Can reduce output by 5-25%. Requires immediate replacement if
cracks cross busbars.

2. Dust Accumulation
Fine particles on panel surface reducing light absorption. Indicated by low
brightness in captured images. Common in arid environments. Can reduce output
by 5-15%. Resolved by cleaning.

3. Snow Coverage
Panel fully or partially covered by snow. Indicated by high white pixel ratio.
Output drops to zero under full coverage. Usually self-resolving.

4. Electrical Damage
Hot spots, delamination, or internal cell failure. Indicated by high texture
variance due to surface irregularities. Requires thermal imaging for confirmation.
Often caused by bypass diode failure.

5. Clean
No visible defects. Panel operating normally. Brightness within normal range,
no significant edge density anomalies.

Inspection Protocol:
- Capture minimum 3 images per zone
- Flag zones with confidence < 0.6 for human review
- Adjacent damaged zones indicate systemic risk
- High edge density + low brightness = compound damage risk
"""

    defect_types = """
Solar Panel Defect Reference

Physical-Damage: Cracks, micro-cracks, broken cells. High edge density signal.
Dusty: Soiling, bird droppings, dust. Low brightness signal.
Snow-Covered: Snow accumulation. High white ratio signal.
Electrical-Damage: Internal faults, hot spots, delamination. High texture variance.
Clean: No defects. Normal brightness, low edge density.

Risk Propagation:
- Defects in one zone increase risk probability in adjacent zones by 15-30%
- Electrical damage often presents in clusters due to shared inverter circuits
- Physical damage clusters indicate installation quality issues or storm damage
"""

    with open(os.path.join(DOCS_DIR, "solar_inspection_guide.txt"), "w") as f:
        f.write(solar_guide)
    with open(os.path.join(DOCS_DIR, "defect_types.txt"), "w") as f:
        f.write(defect_types)


if __name__ == "__main__":
    ingest_documents()