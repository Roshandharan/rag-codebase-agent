"""
ui/app.py

Streamlit chat interface for the RAG Codebase Q&A Agent.

This talks to the FastAPI backend over HTTP (rather than importing the
RAG pipeline directly) so the UI can be deployed and scaled independently
of the API -- and so the same backend could later serve a CLI, a Slack
bot, or a VS Code extension without duplicating logic.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Codebase Q&A Agent", page_icon="🔎", layout="wide")
st.title("🔎 RAG-Powered Codebase Q&A Agent")
st.caption("Point this at any public GitHub repo and ask questions about the code.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_repo" not in st.session_state:
    st.session_state.ingested_repo = None

with st.sidebar:
    st.header("1. Ingest a repository")
    repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/owner/repo")
    ref = st.text_input("Branch/tag (optional)", placeholder="main")

    if st.button("Ingest repo", type="primary", disabled=not repo_url):
        with st.spinner("Cloning, chunking, and embedding..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    json={"repo_url": repo_url, "ref": ref or None},
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.ingested_repo = repo_url
                st.session_state.messages = []
                st.success(
                    f"Ingested {data['files_loaded']} files "
                    f"({data['chunks_created']} chunks) in {data['seconds_elapsed']}s"
                )
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")

    if st.session_state.ingested_repo:
        st.info(f"Active repo:\n{st.session_state.ingested_repo}")

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s['file_path']}` (chunk {s['chunk_index']})")

if prompt := st.chat_input(
    "Ask a question about the codebase...",
    disabled=not st.session_state.ingested_repo,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"repo_url": st.session_state.ingested_repo, "question": prompt},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.markdown(f"- `{s['file_path']}` (chunk {s['chunk_index']})")
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"], "sources": data.get("sources", [])}
                )
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")

if not st.session_state.ingested_repo:
    st.info("👈 Ingest a repository from the sidebar to get started.")
