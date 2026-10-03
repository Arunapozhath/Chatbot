"""
Scrape public dairy/Milkosoft websites and add them to ChromaDB.

Usage:
    python web_ingest.py                    # scrape all WEB_SOURCES from config
    python web_ingest.py --url https://...  # scrape a single URL
    python web_ingest.py --reset            # wipe ChromaDB and re-ingest
"""
import sys, argparse
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import (
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, WEB_SOURCES,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def scrape_url(url: str) -> Document | None:
    try:
        print(f"  Fetching: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["script","style","nav","footer","header","aside","iframe","noscript","form","button"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else url
        main  = soup.find("main") or soup.find("article") or soup.find("body")
        text  = main.get_text(separator="\n", strip=True) if main else ""

        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
        clean = "\n".join(lines)

        if len(clean) < 100:
            print(f"    ⚠  Too little content ({len(clean)} chars), skipping.")
            return None

        print(f"    ✓  {len(clean):,} chars — \"{title[:60]}\"")
        return Document(
            page_content=clean,
            metadata={"source": url, "title": title, "type": "web", "module": "web"},
        )
    except Exception as e:
        print(f"    ✗  Failed: {e}")
        return None


def web_ingest(urls: list[str], reset: bool = False):
    print(f"\n[1/4] Scraping {len(urls)} URL(s)...")
    docs = [d for url in urls if (d := scrape_url(url)) is not None]

    if not docs:
        print("No documents scraped.")
        return 0

    print(f"\n[2/4] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"      {len(chunks)} chunks from {len(docs)} pages.")

    print(f"\n[3/4] Generating embeddings with '{OLLAMA_EMBED_MODEL}'...")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    print(f"\n[4/4] Storing in ChromaDB at '{CHROMA_DB_PATH}'...")
    if reset and Path(CHROMA_DB_PATH).exists():
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)
        print("      ChromaDB wiped.")

    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    db.add_documents(chunks)
    print(f"\n✅  Web ingest done — {len(chunks)} chunks added | {db._collection.count()} total in ChromaDB")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",   help="Single URL to scrape")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    urls = [args.url] if args.url else WEB_SOURCES
    web_ingest(urls, reset=args.reset)
