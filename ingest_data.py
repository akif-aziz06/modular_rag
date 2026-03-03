"""
ingest_data.py
--------------
Run this script ONCE (or whenever you add new PDFs) to:
1. Scan data/<module_name>/ folders for PDF files.
2. Extract and chunk the text using pypdf.
3. Tag each chunk with its module metadata.
4. Embed and persist everything to a local ./chroma_db.
"""

import os
from pypdf import PdfReader
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR      = "./data"
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "salesforce_rag"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# valid module folder names (must match router.py output)
VALID_MODULES = [
    "admin_mode",
    "dev_mode",
    "consultant_mode",
    "interview_mode",
    "interactive_mode",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pypdf."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"  [ERROR] Could not read {pdf_path}: {e}")
        return ""


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long string into overlapping chunks.
    Mimics RecursiveCharacterTextSplitter behaviour without LangChain.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # slide forward with overlap
    return [c.strip() for c in chunks if c.strip()]


# ── Main Ingestion ─────────────────────────────────────────────────────────────

def ingest():
    print("=" * 55)
    print("  Salesforce RAG — Data Ingestion")
    print("=" * 55)

    # 1. Load embedding model
    print(f"\n[1/4] Loading embedding model: {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    # 2. Connect to (or create) ChromaDB
    print(f"[2/4] Connecting to ChromaDB at: {CHROMA_DIR}")
    client     = PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION)

    # 3. Scan flat data directory
    print(f"[3/4] Scanning data directory: {DATA_DIR}\n")
    total_chunks = 0
    
    if not os.path.isdir(DATA_DIR):
        print(f"  [ERROR] Data directory not found: {DATA_DIR}")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"  [ERROR] No PDFs found in {DATA_DIR}")
        return

    print(f"  Found {len(pdf_files)} PDF(s) in {DATA_DIR}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        
        # Determine the module from the filename prefix
        # e.g., "admin_mode_guide.pdf" -> "admin_mode"
        module = None
        for valid_module in VALID_MODULES:
            if pdf_file.lower().startswith(valid_module):
                module = valid_module
                break
                
        if not module:
            print(f"  [SKIP] Skipping {pdf_file} - Name doesn't start with a valid module (e.g. 'dev_mode_')")
            continue

        print(f"    Reading: {pdf_file} (Assigned to: {module}) ...", end=" ")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print("No text extracted, skipping.")
            continue

        chunks = split_text(raw_text)
        print(f"{len(chunks)} chunks")

        # 4. Embed and upsert chunks into ChromaDB
        for i, chunk in enumerate(chunks):
            doc_id    = f"{module}__{pdf_file}__{i}"
            embedding = embedder.encode(chunk).tolist()

            collection.upsert(
                ids        =[doc_id],
                documents  =[chunk],
                embeddings =[embedding],
                metadatas  =[{"module": module, "source": pdf_file}],
            )
            total_chunks += 1

    # 4. Done
    print(f"\n[4/4] Ingestion complete! Total chunks stored: {total_chunks}")
    print(f"      ChromaDB saved to: {CHROMA_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    ingest()
