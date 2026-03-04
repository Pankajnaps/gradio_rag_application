"""
RAG PDF Ingestion Script
========================

Builds a production-grade RAG ingestion pipeline for a rule book PDF.

Features
--------
✔ Extracts both TEXT and TABLES
✔ Keeps tables intact (never split across chunks)
✔ Semantic-aware recursive chunking
✔ Rich metadata (page, section, chunk type)
✔ Uses high quality embeddings
✔ Persistent ChromaDB vector store

Author: Production RAG Template
"""

# -----------------------------
# Required installs
# -----------------------------
# pip install langchain langchain-community langchain-chroma
# pip install chromadb sentence-transformers
# pip install pymupdf pdfplumber
# pip install tqdm


import os
import re
import uuid
from typing import List, Dict

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ----------------------------------------
# Configuration
# ----------------------------------------

PDF_PATH = "./pdf/tata_ipl_2024.pdf"
CHROMA_PATH = "./chroma_db"

# Chunking parameters tuned for rule documents
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ----------------------------------------
# Section detection helper
# ----------------------------------------

def detect_section_heading(text: str) -> str:
    """
    Detect possible section headings.

    Many rule books contain headings like:
    1. Introduction
    2.3 Compliance Rules
    Rule 4: Eligibility

    This function tries to detect them.
    """

    lines = text.split("\n")

    for line in lines[:5]:
        line = line.strip()

        if len(line) < 120 and re.match(r"^(\d+(\.\d+)*)\s+.+", line):
            return line

        if line.lower().startswith("rule"):
            return line

    return "unknown_section"


# ----------------------------------------
# Extract text using PyMuPDF
# ----------------------------------------

def extract_text(pdf_path: str) -> List[Document]:
    """
    Extract raw text from PDF with page metadata.
    """

    documents = []

    doc = fitz.open(pdf_path)

    for page_num in tqdm(range(len(doc)), desc="Extracting text"):
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
                    "source": "rule_book"
                }
            )
        )

    return documents


# ----------------------------------------
# Extract tables using pdfplumber
# ----------------------------------------

def extract_tables(pdf_path: str) -> List[Document]:

    table_docs = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting tables")):

            tables = page.extract_tables()

            for table in tables:

                if not table:
                    continue

                header = [str(c) if c is not None else "" for c in table[0]]
                rows = [[str(c) if c is not None else "" for c in row] for row in table[1:]]

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
                            "source": "rule_book"
                        }
                    )
                )

    return table_docs


# ----------------------------------------
# Chunk text documents
# ----------------------------------------

def chunk_text_documents(docs: List[Document]) -> List[Document]:
    """
    Split long text into semantic chunks.

    Tables are not passed here (handled separately).
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " "
        ]
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


# ----------------------------------------
# Initialize embedding model
# ----------------------------------------

def load_embedding_model():
    """
    Load high quality embedding model.

    Model: all-mpnet-base-v2
    Reason:
    - Strong semantic retrieval
    - Excellent for long documents
    - Open source
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return embeddings


# ----------------------------------------
# Store vectors in ChromaDB
# ----------------------------------------

def store_in_chroma(documents, embedding_model):

    os.makedirs(CHROMA_PATH, exist_ok=True)

    # Filter empty documents
    clean_docs = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    if len(clean_docs) == 0:
        raise ValueError("No valid documents found after filtering empty chunks.")

    print(f"Storing {len(clean_docs)} valid chunks in ChromaDB")

    ids = [str(uuid.uuid4()) for _ in clean_docs]

    vectordb = Chroma.from_documents(
        documents=clean_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
        ids=ids
    )

    # vectordb.persist()

    return vectordb


# ----------------------------------------
# Retrieval test
# ----------------------------------------

def test_retrieval(vectordb):
    """
    Simple verification query.
    """

    query = "What are the latest rules for eligibility?"

    results = vectordb.similarity_search(query, k=3)

    print("\n=============================")
    print("RETRIEVAL TEST RESULTS")
    print("=============================\n")

    for i, doc in enumerate(results):

        print(f"Result {i+1}")
        print(f"Page:", doc.metadata["page"])
        print(f"Type:", doc.metadata["chunk_type"])
        print(f"Section:", doc.metadata["section"])
        print("-" * 50)
        print(doc.page_content[:600])
        print("\n")


# ----------------------------------------
# Main pipeline
# ----------------------------------------

def main():

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    print("\nStarting PDF ingestion pipeline...\n")

    # Extract
    text_docs = extract_text(PDF_PATH)
    table_docs = extract_tables(PDF_PATH)

    print(f"\nText pages extracted: {len(text_docs)}")
    print(f"Tables extracted: {len(table_docs)}")

    # Chunk text
    text_chunks = chunk_text_documents(text_docs)

    print(f"\nText chunks created: {len(text_chunks)}")

    # Combine
    all_docs = text_chunks + table_docs

    print(f"\nTotal chunks to embed: {len(all_docs)}")

    # Load embedding model
    embeddings = load_embedding_model()

    # Store in Chroma
    vectordb = store_in_chroma(all_docs, embeddings)

    print("\nVector database created successfully.")

    # Run retrieval test
    test_retrieval(vectordb)


# ----------------------------------------
# Entry point
# ----------------------------------------

if __name__ == "__main__":
    main()