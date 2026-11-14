from services.ingestion import load_file, chunk_documents
from pathlib import Path
from rag.vectorstore import add_documents


doc = load_file(Path("data/company.md"))
chunks = chunk_documents(doc)
add_documents(chunks)