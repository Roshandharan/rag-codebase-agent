# RAG-Powered Codebase Q&A Agent

Point this at any public GitHub repository and ask natural-language questions
about its code. It clones the repo, chunks the source with language-aware
splitting, embeds the chunks (with a caching layer so re-ingestion is cheap),
and answers questions with a retrieval-augmented LLM that cites the exact
file and chunk it drew each claim from.

## Why this exists

Every engineering team eventually wants a way to ask "how does X work in
this codebase" without grepping blind. This project is an end-to-end,
from-scratch implementation of that idea: not a wrapper around a hosted RAG
product, but the actual pipeline -- chunking strategy, embedding cache,
retrieval strategy, and citation-grounded prompting -- built and reasoned
about explicitly.

## Architecture

```
GitHub repo URL
      │
      ▼
 ingestion/ingest.py      clone repo, walk files, filter to supported
                           source extensions, skip vendored/binary dirs
      │
      ▼
 ingestion/chunking.py    language-aware recursive chunking
                           (function/class-boundary aware where possible)
      │
      ▼
 ingestion/embed.py       OpenAI embeddings, wrapped in a cache
                           (Redis if configured, else local disk)
                           → persisted in a Chroma collection per repo
      │
      ▼
 rag/chain.py             MMR retrieval (diverse, not just top-k similar)
                           → citation-grounded prompt → ChatOpenAI
      │
      ▼
 api/main.py (FastAPI)    /ingest and /ask endpoints
      │
      ▼
 ui/app.py (Streamlit)    chat interface, shows sources per answer
```

## Key design decisions

- **Code-aware chunking, not fixed character windows.** Using
  `RecursiveCharacterTextSplitter.from_language(...)` per file type means
  chunk boundaries tend to fall on function/class edges instead of
  mid-statement, which measurably improves retrieval quality on code.
- **MMR retrieval instead of plain top-k similarity.** Codebases are full
  of near-duplicate chunks (imports, boilerplate). MMR trades a little raw
  similarity for diversity across the retrieved set, so the model doesn't
  get six near-identical chunks from the same file.
- **Cached embeddings.** Re-ingesting a repo (or ingesting a fork that
  shares most of its history) shouldn't re-pay for embeddings on unchanged
  chunks. `CacheBackedEmbeddings` sits in front of the OpenAI embedding
  calls; Redis is used if configured, otherwise a local file store.
- **Citation-required prompting.** The system prompt requires a
  `(file_path, chunk N)` citation for every claim, and the API returns the
  raw source list alongside the answer -- so the UI can show exactly which
  chunks the answer is (and isn't) grounded in.

## Running locally

### 1. With Docker Compose (recommended)

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY

docker compose up --build
```

- API: http://localhost:8000/docs
- UI: http://localhost:8501

### 2. Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY
export $(grep -v '^#' .env | xargs)

uvicorn api.main:app --reload --port 8000
# in a second terminal:
streamlit run ui/app.py
```

## Usage

1. Open the Streamlit UI, paste a GitHub repo URL (e.g.
   `https://github.com/octocat/Hello-World`), click **Ingest repo**.
2. Ask a question in the chat box. Each answer's **Sources** expander shows
   the exact files/chunks the model was given.

Or hit the API directly:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/octocat/Hello-World"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/octocat/Hello-World", "question": "What does this repo do?"}'
```

## Testing

```bash
pytest -v
```

CI runs the same suite plus `ruff` linting on every push/PR (see
`.github/workflows/ci.yml`).

## Roadmap / known limitations

- Single-process in-memory registry of ingested repos (`api/main.py`) --
  fine for a demo, would need a real table (e.g. Postgres) to survive
  restarts or run behind multiple API replicas.
- No incremental re-chunking on `git pull` -- currently a full re-ingest.
  Diffing changed files between commits would make repo updates far cheaper.
- No repo size/rate limiting -- a very large monorepo will take a while
  to embed on first ingest.

## Tech stack

Python · LangChain · ChromaDB · OpenAI Embeddings · FastAPI · Streamlit ·
Redis (optional embedding cache) · Docker Compose · GitHub Actions
