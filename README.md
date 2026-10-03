# Milkosoft AI Chatbot

An intelligent RAG (Retrieval-Augmented Generation) + SQL chatbot for the **Milkosoft vBiz** dairy cooperative management platform. Ask questions in plain English — the bot queries your live OFBiz/vBiz database and knowledge-base documents to answer instantly.

---

## Features

- **RAG pipeline** — ingests HR policies, procurement docs, payroll rules, S&D guides into ChromaDB
- **Live SQL queries** — auto-generates safe, whitelisted SQL against your vBiz MySQL database
- **Dual LLM support** — run locally with Ollama (LLaMA, Mistral, Phi-3) or use Groq cloud for faster responses
- **JWT authentication** — login with your vBiz user credentials
- **RLHF feedback loop** — thumbs up/down + corrections → DPO training data
- **Web search fallback** — low-confidence answers trigger DuckDuckGo search
- **Streaming responses** — server-sent events for real-time chat UI

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| RAG / Embeddings | LangChain + ChromaDB + nomic-embed-text |
| Local LLM | Ollama (LLaMA 3.2, Mistral, Phi-3, Gemma 2) |
| Cloud LLM | Groq API (optional) |
| Database | MySQL (OFBiz/vBiz schema) |
| Auth | JWT (PyJWT) |
| Feedback / RLHF | SQLite + TRL DPO training |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project Structure

```
milkosoft-chatbot/
├── app.py                  # FastAPI server — all routes
├── auth.py                 # JWT login + user verification
├── config.example.py       # Config template — copy to config.py
├── db_connector.py         # MySQL connection + safe SQL execution
├── rag_pipeline.py         # RAG chain: embed → retrieve → LLM
├── query_router.py         # Decides: RAG vs SQL vs web search
├── ingest.py               # Loads data/ docs into ChromaDB
├── web_ingest.py           # Fetches web pages into ChromaDB
├── feedback.py             # Save ratings, export DPO pairs
├── dpo_train.py            # Fine-tune model on feedback pairs
├── frontend/
│   └── index.html          # Chat UI
├── data/
│   └── database-understanding.md   # vBiz schema reference
├── requirements.txt
├── setup.bat               # One-time setup (Windows)
└── start.bat               # Start the server (Windows)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- MySQL with your vBiz database
- Git

### 1. Clone the repo

```bash
git clone https://github.com/your-username/milkosoft-chatbot.git
cd milkosoft-chatbot
```

### 2. Set up the virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Or on Windows, just run:
```bash
setup.bat
```

### 3. Configure

```bash
copy config.example.py config.py   # Windows
cp config.example.py config.py     # macOS / Linux
```

Edit `config.py` and fill in:

```python
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "your_db_user"
MYSQL_PASS = "your_db_password"
MYSQL_DB   = "your_database_name"

JWT_SECRET = "replace-with-a-long-random-secret"

# Optional: Groq for faster cloud inference
GROQ_API_KEY = "gsk_xxxx..."
```

### 4. Pull the LLM model

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### 5. Add your knowledge-base documents

Place your `.pdf`, `.docx`, `.xlsx`, or `.txt` files under `data/`:

```
data/
├── HR_Module/
├── Procurement/
├── S&D_Module/
└── ...
```

Then ingest them:

```bash
python ingest.py
```

### 6. Start the server

```bash
python app.py
```

Or on Windows:
```bash
start.bat
```

Open **http://localhost:8000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `GET` | `/health` | Server + DB health check |
| `GET` | `/models` | Available LLM models |
| `POST` | `/auth/login` | Login → returns JWT token |
| `POST` | `/chat/stream` | Streaming chat (SSE) |
| `POST` | `/feedback` | Submit thumbs up/down |
| `GET` | `/feedback/stats` | Feedback statistics |
| `GET` | `/feedback/export` | Export DPO training pairs |
| `POST` | `/ingest` | Re-ingest documents (auth required) |

---

## RLHF / DPO Training (Optional)

After collecting at least 5 feedback pairs with corrections:

```bash
python dpo_train.py
```

This fine-tunes the local model on preferred vs rejected answer pairs using TRL's DPO trainer. Requires:

```bash
pip install trl transformers datasets accelerate bitsandbytes peft
```

---

## Security Notes

- `config.py` is in `.gitignore` — **never commit it**
- The SQL engine only queries tables listed in `ALLOWED_TABLES` — security-sensitive tables (`user_login`, `token_blacklist`, etc.) are never exposed
- Passwords are verified against the vBiz database hash — never stored by the chatbot
- JWT tokens expire after 8 hours (configurable)

---

## Adding a New Data Source

1. Place the file in `data/<module>/`
2. Run `python ingest.py` to re-embed
3. Or hit `POST /ingest` from the UI (admin login required)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
