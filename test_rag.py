
from rag_pipeline import (
    build_knowledge_base,
    get_vectorstore,
    retrieve_documents,
    assess_relevance
)

print("\n========== CHROMADB TEST ==========")

# Check current database
store = get_vectorstore()
existing = store.get()

print("Existing chunks in ChromaDB:",
      len(existing.get("ids", [])))

# Build knowledge base if it is empty
print("\nBuilding knowledge base...")

total = build_knowledge_base()

print("Knowledge base chunks:", total)

# Check database again
store = get_vectorstore()
data = store.get()

print("Stored chunks in ChromaDB:",
      len(data.get("ids", [])))

# Test retrieval
question = "What is anxiety?"

print("\n========== RETRIEVAL TEST ==========")
print("Question:", question)

results = retrieve_documents(question)

print("Retrieved documents:", len(results))

for index, (document, distance) in enumerate(results):

    print(f"\nDocument {index + 1}")
    print("Source:", document.metadata.get("source"))
    print("Distance:", distance)
    print("Retrieved text:")
    print(document.page_content[:300])

# Test relevance
relevant, explanation = assess_relevance(results)

print("\n========== RELEVANCE TEST ==========")
print("Relevant:", relevant)
print("Explanation:", explanation)
