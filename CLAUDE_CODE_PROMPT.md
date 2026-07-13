# Prompt for Claude Code

Paste everything below into Claude Code in this project's root directory
(where this file lives). It's written as a single first message.

---

I'm finishing a project called "RAG-Powered Codebase Q&A Agent." The core
pipeline is already built and tested — read through it before changing
anything:

- `ingestion/ingest.py` — clones a GitHub repo, filters to supported source
  extensions, skips vendored/binary dirs
- `ingestion/chunking.py` — language-aware chunking via LangChain's
  `RecursiveCharacterTextSplitter.from_language`
- `ingestion/embed.py` — cached OpenAI embeddings (Redis or local disk) into
  a per-repo Chroma collection
- `rag/chain.py` — MMR retrieval + citation-grounded prompt + ChatOpenAI
- `api/main.py` — FastAPI `/ingest` and `/ask` endpoints
- `ui/app.py` — Streamlit chat UI that calls the API over HTTP
- `tests/` — pytest suite covering chunking, ingestion, and the API (all
  passing as of this commit; mocks the LLM/embedding calls, doesn't require
  a real API key to run)

Run `pytest -v` first to confirm your environment matches mine before
making changes.

## What I want you to do, in order

1. **Get it running locally end-to-end with a real OpenAI key.** Walk me
   through `.env` setup, `docker compose up --build`, and do a real ingest
   + ask against a small public repo (pick one, e.g.
   `https://github.com/octocat/Hello-World` won't work since it has no
   source files — use something like `https://github.com/pallets/flask`
   or a smaller repo of your choice). Fix anything that breaks in practice
   that the mocked tests didn't catch.

2. **Harden the API layer:**
   - The `_INGESTED_REPOS` registry in `api/main.py` is in-memory and will
     lose state on restart. Replace it with a lightweight SQLite table (no
     need for full Postgres) tracking `repo_url`, `collection_name`,
     `last_ingested_at`, `commit_sha`.
   - Add a request timeout / size guard so a huge monorepo can't hang the
     `/ingest` endpoint indefinitely — cap total files or total bytes and
     return a clear 413-style error if exceeded.
   - Add basic structured logging around retrieval (which chunks were
     retrieved, retrieval latency, generation latency) so performance is
     debuggable.

3. **Add incremental re-ingestion.** Right now every `/ingest` call
   re-chunks and re-embeds every file. Store the last-ingested commit SHA
   per repo, and on a repeat `/ingest`, diff changed files via
   `git diff --name-only <old_sha> <new_sha>` and only re-chunk/re-embed
   those files, deleting stale chunks for removed/changed files from the
   Chroma collection first.

4. **Improve retrieval quality:**
   - Add a small eval script (`scripts/eval_retrieval.py`) with ~10 hand-
     written (question, expected_file) pairs against a fixed test repo,
     and report retrieval hit-rate@k. This is the kind of thing I want to
     be able to talk about concretely in interviews.
   - Try adjusting `k`, `fetch_k`, and `lambda_mult` in
     `rag/chain.py::build_rag_chain` and report what changes.

5. **Deploy it somewhere reachable** (Railway or Render — pick whichever
   is faster to get working with the existing `Dockerfile`/
   `docker-compose.yml`) and give me a live URL.

6. **Polish the README** — add a real architecture diagram (Mermaid is
   fine, rendered inline in the README), and a short GIF or set of
   screenshots of the Streamlit UI actually answering a question with
   citations.

7. **Write 2–3 LinkedIn-ready sentences** describing what the eval script
   found (e.g. "retrieval hit-rate@5 improved from X% to Y% after tuning
   MMR lambda") — I want a number I can actually defend if someone asks
   about it in an interview, not a made-up one.

## Ground rules

- Don't rewrite working, tested code just to restyle it. Extend it.
- Every change you make that affects behavior should have a test.
- Run the full test suite before you tell me something is done.
- If something in the existing code is actually wrong (not just a style
  preference), tell me directly and explain why before changing it.

## Git / commit requirements — read this carefully

This repository's commits must be attributable entirely to me as the
author of the work. Specifically:

- Do **not** add a `Co-Authored-By: Claude <...>` trailer to any commit.
- Do **not** add "🤖 Generated with Claude Code" or any similar tool-
  attribution line to commit messages.
- Write commit messages as plain, first-person-appropriate engineering
  commit messages (e.g. "Add SQLite-backed repo registry", not "Claude
  added SQLite-backed repo registry").
- Confirm `git config user.name` and `git config user.email` are set to
  mine (already configured as "Roshan Dharan" /
  `roshandharan.shashidharan@usc.edu` in this repo) before committing —
  don't override them.
- If your default settings would normally add tool-attribution trailers,
  disable that behavior for this repo before making any commits, and
  confirm you've done so.
