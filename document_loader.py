
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(folder: Path = DATA_DIR) -> List[Document]:

    documents = []

    folder.mkdir(parents=True, exist_ok=True)

    for path in folder.rglob("*"):

        if not path.is_file():
            continue

        try:

            if path.suffix.lower() == ".txt":

                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

            elif path.suffix.lower() == ".pdf":

                reader = PdfReader(str(path))

                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )

            else:
                continue

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "file_path": str(path)
                        }
                    )
                )

        except Exception as error:

            print(f"Error loading {path.name}: {error}")

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=["\n\n", "\n", ". ", " ", ""]

    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

    return chunks
