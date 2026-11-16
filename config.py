
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "mental_health"
CHROMA_DIR = BASE_DIR / "chroma_db"

OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "mental_health_knowledge"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

TOP_K = 5

# Lower vector distance means a closer match.
MAX_RELEVANCE_DISTANCE = 1.25
