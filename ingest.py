"""
Multi-format document ingestion pipeline.

Supported formats: .txt, .md, .docx, .doc, .pdf, .xlsx, .xls, .csv
Run:  python ingest.py  (or python ingest.py --dir /path/to/docs)

Documents are chunked and embedded into ChromaDB using nomic-embed-text via Ollama.
"""
import sys
import argparse
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    CHROMA_DB_PATH, DATA_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


# ── Per-format loaders ─────────────────────────────────────────────────────

def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_docx(path: Path) -> str:
    try:
        import docx
        doc = docx.Document(str(path))
        parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip(" |"):
                    parts.append(row_text)
        return "\n".join(parts)
    except Exception as e:
        print(f"    [docx] Error reading {path.name}: {e}")
        return ""


def load_doc(path: Path) -> str:
    """Legacy .doc — try python-docx first (works on some), fallback to textract."""
    try:
        return load_docx(path)
    except Exception:
        pass
    try:
        import textract
        return textract.process(str(path)).decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"    [doc] Cannot read {path.name}: {e}. Install textract or convert to .docx")
        return ""


def load_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        print(f"    [pdf] Error reading {path.name}: {e}")
        return ""


def load_xlsx(path: Path) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        parts = []
        for sheet in wb.worksheets:
            parts.append(f"Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                row_text = " | ".join(str(c) if c is not None else "" for c in row)
                if row_text.strip(" |"):
                    parts.append(row_text)
        return "\n".join(parts)
    except Exception as e:
        print(f"    [xlsx] Error reading {path.name}: {e}")
        return ""


def load_xls(path: Path) -> str:
    try:
        import xlrd
        wb = xlrd.open_workbook(str(path))
        parts = []
        for sheet in wb.sheets():
            parts.append(f"Sheet: {sheet.name}")
            for row_idx in range(sheet.nrows):
                row_text = " | ".join(str(sheet.cell_value(row_idx, c)) for c in range(sheet.ncols))
                if row_text.strip(" |"):
                    parts.append(row_text)
        return "\n".join(parts)
    except Exception as e:
        print(f"    [xls] Error reading {path.name}: {e}")
        return ""


def load_csv(path: Path) -> str:
    import csv
    try:
        rows = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= 500:    # cap CSV rows to avoid huge embeddings
                    rows.append("... (truncated)")
                    break
                rows.append(" | ".join(row))
        return "\n".join(rows)
    except Exception as e:
        print(f"    [csv] Error reading {path.name}: {e}")
        return ""


LOADERS = {
    ".txt":  load_txt,
    ".md":   load_md,
    ".docx": load_docx,
    ".doc":  load_doc,
    ".pdf":  load_pdf,
    ".xlsx": load_xlsx,
    ".xls":  load_xls,
    ".csv":  load_csv,
}


# ── Ingestion ──────────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_document(path: Path) -> Document | None:
    ext = path.suffix.lower()
    loader_fn = LOADERS.get(ext)
    if not loader_fn:
        return None

    text = loader_fn(path)
    if not text or len(text.strip()) < 50:
        return None

    return Document(
        page_content=text,
        metadata={
            "source":   str(path),
            "filename": path.name,
            "module":   path.parent.name,
            "filetype": ext.lstrip("."),
            "hash":     file_hash(path),
        },
    )


def ingest(data_dir: str = DATA_DIR, reset: bool = False) -> int:
    base = Path(data_dir)
    if not base.exists():
        print(f"[ERROR] Directory '{data_dir}' not found.")
        sys.exit(1)

    supported = list(LOADERS.keys())
    all_files = [
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    ]

    if not all_files:
        print(f"[ERROR] No supported files found in '{data_dir}'.")
        print(f"        Supported formats: {', '.join(supported)}")
        sys.exit(1)

    print(f"\n[1/4] Found {len(all_files)} files in '{data_dir}'")

    docs = []
    for path in all_files:
        print(f"      Loading: {path.name}")
        doc = load_document(path)
        if doc:
            docs.append(doc)
        else:
            print(f"      ⚠  Skipped (empty or unsupported): {path.name}")

    if not docs:
        print("[ERROR] No documents successfully loaded.")
        sys.exit(1)

    print(f"\n[2/4] Splitting {len(docs)} documents into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"      Created {len(chunks)} chunks.")

    print(f"\n[3/4] Generating embeddings with '{OLLAMA_EMBED_MODEL}' ...")
    print("      (Make sure Ollama is running: ollama serve)")
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    print(f"\n[4/4] Storing in ChromaDB at '{CHROMA_DB_PATH}' ...")
    import shutil
    if reset and Path(CHROMA_DB_PATH).exists():
        shutil.rmtree(CHROMA_DB_PATH)
        print("      Wiped existing ChromaDB for fresh re-index.")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
    )
    total = db._collection.count()
    print(f"\n✅  Ingestion complete — {len(chunks)} new chunks | {total} total in ChromaDB")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge base documents into ChromaDB")
    parser.add_argument("--dir",   default=DATA_DIR, help="Directory containing documents")
    parser.add_argument("--reset", action="store_true", help="Wipe ChromaDB before re-indexing")
    args = parser.parse_args()
    ingest(data_dir=args.dir, reset=args.reset)
