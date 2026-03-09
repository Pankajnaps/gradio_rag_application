"""
Multi-Year Rulebook RAG System
==============================

Features
--------
✔ Ingest multiple PDFs (2024 + 2025)
✔ Extract text + tables
✔ Semantic chunking
✔ Metadata aware retrieval
✔ Query routing by year
✔ Compare rules across years

"""

# -------------------------------------------------
# Install dependencies
# -------------------------------------------------
# pip install langchain langchain-community langchain-chroma
# pip install chromadb sentence-transformers
# pip install pymupdf pdfplumber tqdm


import os
import re
import uuid
from typing import List

import fitz
import pdfplumber
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# -------------------------------------------------
# Configuration
# -------------------------------------------------

PDF_FILES = {
    "ipl_2024": "./pdf/tata_ipl_2024.pdf",
    "ipl_2025": "./pdf/tata_ipl_2025.pdf"
}

CHROMA_PATH = "./chroma_db"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# -------------------------------------------------
# Section Detection
# -------------------------------------------------

def detect_section_heading(text: str):

    lines = text.split("\n")

    for line in lines[:5]:

        line = line.strip()

        if len(line) < 120 and re.match(r"^(\d+(\.\d+)*)\s+.+", line):
            return line

        if line.lower().startswith("rule"):
            return line

    return "unknown_section"


# -------------------------------------------------
# Extract Text
# -------------------------------------------------

def extract_text(pdf_path: str, rule_year: str) -> List[Document]:

    documents = []

    doc = fitz.open(pdf_path)

    for page_num in tqdm(range(len(doc)), desc=f"Extracting text ({rule_year})"):

        page = doc.load_page(page_num)

        text = page.get_text()

        if not text.strip():
            continue

        section = detect_section_heading(text)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_num + 1,
                    "chunk_type": "text",
                    "section": section,
                    "rule_year": rule_year,
                    "source": "ipl_rule_book"
                }
            )
        )

    return documents


# -------------------------------------------------
# Extract Tables
# -------------------------------------------------

def extract_tables(pdf_path: str, rule_year: str) -> List[Document]:

    table_docs = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(
            tqdm(pdf.pages, desc=f"Extracting tables ({rule_year})")
        ):

            tables = page.extract_tables()

            for table in tables:

                if not table:
                    continue

                header = [str(c) if c else "" for c in table[0]]
                rows = [[str(c) if c else "" for c in row] for row in table[1:]]

                md = "| " + " | ".join(header) + " |\n"
                md += "| " + " | ".join(["---"] * len(header)) + " |\n"

                for row in rows:
                    md += "| " + " | ".join(row) + " |\n"

                table_docs.append(
                    Document(
                        page_content=md,
                        metadata={
                            "page": page_num + 1,
                            "chunk_type": "table",
                            "section": "table_data",
                            "rule_year": rule_year,
                            "source": "ipl_rule_book"
                        }
                    )
                )

    return table_docs


# -------------------------------------------------
# Chunk Text
# -------------------------------------------------

def chunk_text_documents(docs: List[Document]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []

    for doc in docs:

        splits = splitter.split_text(doc.page_content)

        for chunk in splits:

            chunks.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

    return chunks


# -------------------------------------------------
# Embedding Model
# -------------------------------------------------

def load_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return embeddings


# -------------------------------------------------
# Store in ChromaDB
# -------------------------------------------------

def store_in_chroma(documents, embedding_model):

    os.makedirs(CHROMA_PATH, exist_ok=True)

    clean_docs = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    ids = [str(uuid.uuid4()) for _ in clean_docs]

    vectordb = Chroma.from_documents(
        documents=clean_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
        ids=ids
    )

    return vectordb


# -------------------------------------------------
# Detect Year in Query
# -------------------------------------------------

def detect_rule_year(query: str):

    query = query.lower()

    if "2024" in query:
        return "ipl_2024"

    if "2025" in query:
        return "ipl_2025"

    if "latest" in query or "current" in query:
        return "ipl_2025"

    # default → latest rule
    return "ipl_2025"


# -------------------------------------------------
# Detect comparison query
# -------------------------------------------------

def is_comparison_query(query):

    query = query.lower()

    keywords = [
        "compare",
        "difference",
        "change",
        "changed",
        "vs"
    ]

    return any(k in query for k in keywords)


# -------------------------------------------------
# Retrieval Engine
# -------------------------------------------------

def retrieve_documents(query, vectordb):

    if is_comparison_query(query):

        print("Comparison query detected\n")

        docs_2024 = vectordb.similarity_search(
            query,
            k=3,
            filter={"rule_year": "ipl_2024"}
        )

        docs_2025 = vectordb.similarity_search(
            query,
            k=3,
            filter={"rule_year": "ipl_2025"}
        )

        return docs_2024 + docs_2025

    year = detect_rule_year(query)

    print(f"Searching rules for: {year}\n")

    results = vectordb.similarity_search(
        query,
        k=5,
        filter={"rule_year": year}
    )

    return results


# -------------------------------------------------
# Ask Question
# -------------------------------------------------

def ask_question(query, vectordb):

    docs = retrieve_documents(query, vectordb)

    print("\n==============================")
    print("RETRIEVED DOCUMENTS")
    print("==============================\n")

    for i, doc in enumerate(docs):

        print(f"Result {i+1}")
        print("Year:", doc.metadata["rule_year"])
        print("Page:", doc.metadata["page"])
        print("Section:", doc.metadata["section"])
        print("-" * 40)

        print(doc.page_content[:400])
        print("\n")


# -------------------------------------------------
# Ingestion Pipeline
# -------------------------------------------------

def run_ingestion():

    print("\nStarting ingestion pipeline...\n")

    all_docs = []

    for rule_year, pdf_path in PDF_FILES.items():

        print(f"\nProcessing {rule_year}")

        text_docs = extract_text(pdf_path, rule_year)
        table_docs = extract_tables(pdf_path, rule_year)

        text_chunks = chunk_text_documents(text_docs)

        all_docs.extend(text_chunks)
        all_docs.extend(table_docs)

    embeddings = load_embedding_model()

    vectordb = store_in_chroma(all_docs, embeddings)

    print("\nVector DB created successfully\n")

    return vectordb


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    vectordb = run_ingestion()

    while True:

        query = input("\nAsk question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        ask_question(query, vectordb)