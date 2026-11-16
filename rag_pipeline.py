
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_RELEVANCE_DISTANCE,
    TOP_K
)

from document_loader import load_documents, split_documents


_embeddings = None
_vectorstore = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        _embeddings = HuggingFaceEmbeddings(

            model_name=EMBEDDING_MODEL,

            model_kwargs={"device": "cpu"},

            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    return _embeddings


def get_vectorstore():

    global _vectorstore

    if _vectorstore is None:

        _vectorstore = Chroma(

            collection_name=COLLECTION_NAME,

            embedding_function=get_embeddings(),

            persist_directory=str(CHROMA_DIR)

        )

    return _vectorstore


def build_knowledge_base(force_rebuild=False):

    global _vectorstore

    store = get_vectorstore()

    if force_rebuild:

        store.delete_collection()

        _vectorstore = None

        store = get_vectorstore()

    else:

        existing = store.get()

        if existing and existing.get("ids"):

            return len(existing["ids"])

    documents = load_documents()

    if not documents:

        raise RuntimeError(
            "No TXT or PDF documents found."
        )

    chunks = split_documents(documents)

    store.add_documents(chunks)

    return len(chunks)


def retrieve_documents(question, k=TOP_K):

    store = get_vectorstore()

    results = store.similarity_search_with_score(

        question,

        k=k

    )

    return results


def assess_relevance(results):

    if not results:

        return False, "No documents retrieved."

    best_distance = min(
        distance for _, distance in results
    )

    relevant = (
        best_distance <= MAX_RELEVANCE_DISTANCE
    )

    explanation = (
        f"Best distance: {best_distance:.3f}. "
        f"Maximum allowed distance: "
        f"{MAX_RELEVANCE_DISTANCE:.3f}."
    )

    return relevant, explanation
