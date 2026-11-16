
from document_loader import load_documents, split_documents

print("\n========== DOCUMENT LOADING TEST ==========")

documents = load_documents()

print("Total documents loaded:", len(documents))

for document in documents:
    print("\nSource:", document.metadata["source"])
    print("Characters:", len(document.page_content))

print("\n========== CHUNKING TEST ==========")

chunks = split_documents(documents)

print("Total chunks created:", len(chunks))

for index, chunk in enumerate(chunks[:5]):
    print(f"\nChunk {index + 1}")
    print("Source:", chunk.metadata["source"])
    print("Text:", chunk.page_content[:200])
