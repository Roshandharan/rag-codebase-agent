"""
scripts/eval_retrieval.py

Small retrieval-quality eval: ingest a fixed test repo, run a hand-written
set of (question, expected_file) pairs through the MMR retriever, and
report hit-rate@k -- did the expected file show up anywhere in the top-k
retrieved chunks?

This exists to make retrieval-tuning decisions (k, fetch_k, lambda_mult in
rag/chain.py::build_rag_chain) concrete instead of vibes-based, and to have
a real number to point to.

Usage:
    python scripts/eval_retrieval.py                       # default k/fetch_k/lambda_mult
    python scripts/eval_retrieval.py --k 8 --fetch-k 30 --lambda-mult 0.7
    python scripts/eval_retrieval.py --sweep                # compare several configs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunking import chunk_documents  # noqa: E402
from ingestion.embed import get_or_create_vectorstore, open_existing_vectorstore  # noqa: E402
from ingestion.ingest import clone_repo, load_repo_documents  # noqa: E402

FIXED_TEST_REPO = "https://github.com/pallets/flask"
EVAL_COLLECTION = "eval-flask-fixed"

# Hand-written (question, expected_file) pairs. expected_file is a
# repo-relative path; a "hit" means it appears anywhere among the top-k
# retrieved chunks for that question. Grounded in flask's actual source
# layout (verified against a real clone, not guessed):
EVAL_CASES: list[dict] = [
    {
        "question": "Where is the main request dispatch logic that calls the matched view function?",
        "expected_file": "src/flask/app.py",
    },
    {
        "question": "What class implements Flask's default secure cookie-based session storage?",
        "expected_file": "src/flask/sessions.py",
    },
    {
        "question": "Where is the url_for function defined?",
        "expected_file": "src/flask/helpers.py",
    },
    {
        "question": "Where is the Blueprint class defined?",
        "expected_file": "src/flask/blueprints.py",
    },
    {
        "question": "Where is render_template implemented?",
        "expected_file": "src/flask/templating.py",
    },
    {
        "question": "Where is the Config class that loads app configuration defined?",
        "expected_file": "src/flask/config.py",
    },
    {
        "question": "Where are the test client classes like FlaskClient and FlaskCliRunner defined?",
        "expected_file": "src/flask/testing.py",
    },
    {
        "question": "Where are Flask's CLI commands, like the one behind `flask run`, implemented?",
        "expected_file": "src/flask/cli.py",
    },
    {
        "question": "Where is the MethodView class for class-based views defined?",
        "expected_file": "src/flask/views.py",
    },
    {
        "question": "Where are RequestContext and AppContext defined?",
        "expected_file": "src/flask/ctx.py",
    },
]


def ensure_ingested() -> None:
    """Ingest the fixed test repo into a dedicated eval collection, once."""
    existing = open_existing_vectorstore(EVAL_COLLECTION).get()
    if existing["ids"]:
        print(f"Using existing '{EVAL_COLLECTION}' collection ({len(existing['ids'])} chunks).")
        return

    print(f"Ingesting {FIXED_TEST_REPO} into '{EVAL_COLLECTION}'...")
    repo_path = clone_repo(FIXED_TEST_REPO)
    docs = load_repo_documents(repo_path)
    chunks = chunk_documents(docs)
    get_or_create_vectorstore(chunks, collection_name=EVAL_COLLECTION)
    print(f"Ingested {len(docs)} files, {len(chunks)} chunks.")


def run_eval(k: int, fetch_k: int, lambda_mult: float, verbose: bool = True) -> dict:
    vectorstore = open_existing_vectorstore(EVAL_COLLECTION)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )

    hits = 0
    per_question = []
    start = time.perf_counter()

    for case in EVAL_CASES:
        docs = retriever.invoke(case["question"])
        retrieved_files = [d.metadata.get("file_path") for d in docs]
        hit = case["expected_file"] in retrieved_files
        hits += hit
        per_question.append({**case, "retrieved_files": retrieved_files, "hit": hit})

    elapsed = time.perf_counter() - start
    hit_rate = hits / len(EVAL_CASES)

    if verbose:
        for result in per_question:
            mark = "HIT " if result["hit"] else "MISS"
            print(f"  [{mark}] {result['question']}")
            print(f"         expected: {result['expected_file']}")
            if not result["hit"]:
                print(f"         got:      {result['retrieved_files']}")

    print(
        f"\nk={k} fetch_k={fetch_k} lambda_mult={lambda_mult}: "
        f"hit-rate@{k} = {hits}/{len(EVAL_CASES)} ({hit_rate:.0%}) in {elapsed:.2f}s"
    )
    return {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult, "hit_rate": hit_rate, "hits": hits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--lambda-mult", type=float, default=0.75)
    parser.add_argument("--sweep", action="store_true", help="Compare several k/fetch_k/lambda_mult configs")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary line per config")
    args = parser.parse_args()

    ensure_ingested()

    if not args.sweep:
        run_eval(args.k, args.fetch_k, args.lambda_mult, verbose=not args.quiet)
        return

    configs = [
        (4, 15, 0.5),
        (6, 20, 0.5),
        (6, 20, 0.25),
        (6, 20, 0.75),
        (8, 30, 0.5),
        (10, 40, 0.5),
    ]
    print(f"\nSweeping {len(configs)} configs over {len(EVAL_CASES)} eval cases...\n")
    results = [run_eval(k, fetch_k, lm, verbose=not args.quiet) for k, fetch_k, lm in configs]

    print("\n=== Summary ===")
    for r in sorted(results, key=lambda r: -r["hit_rate"]):
        print(f"  k={r['k']:<3} fetch_k={r['fetch_k']:<3} lambda_mult={r['lambda_mult']:<5} "
              f"hit-rate@{r['k']} = {r['hits']}/{len(EVAL_CASES)} ({r['hit_rate']:.0%})")


if __name__ == "__main__":
    main()
