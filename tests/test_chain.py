import logging
from dataclasses import dataclass, field

from langchain_core.runnables import RunnableLambda

import rag.chain as chain_module


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, question):
        return self._docs


class FakeVectorstore:
    def __init__(self, docs):
        self._docs = docs
        self.as_retriever_kwargs = None

    def as_retriever(self, **kwargs):
        self.as_retriever_kwargs = kwargs
        return FakeRetriever(self._docs)


def test_build_rag_chain_returns_answer_and_sources(monkeypatch):
    docs = [
        FakeDoc("def add(a, b): return a + b", {"file_path": "a.py", "chunk_index": 0, "language": "python"}),
        FakeDoc("def sub(a, b): return a - b", {"file_path": "b.py", "chunk_index": 1, "language": "python"}),
    ]
    monkeypatch.setattr(
        chain_module, "ChatAnthropic", lambda model=None: RunnableLambda(lambda prompt_value: "fake answer")
    )

    chain = chain_module.build_rag_chain(FakeVectorstore(docs), k=2, fetch_k=2)
    result = chain.invoke({"question": "What does add do?"})

    assert result["answer"] == "fake answer"
    assert result["sources"] == [
        {"file_path": "a.py", "chunk_index": 0, "language": "python"},
        {"file_path": "b.py", "chunk_index": 1, "language": "python"},
    ]


def test_build_rag_chain_passes_mmr_params_to_retriever(monkeypatch):
    docs = [FakeDoc("content", {"file_path": "a.py", "chunk_index": 0, "language": "python"})]
    monkeypatch.setattr(
        chain_module, "ChatAnthropic", lambda model=None: RunnableLambda(lambda prompt_value: "fake answer")
    )
    vectorstore = FakeVectorstore(docs)

    chain_module.build_rag_chain(vectorstore, k=8, fetch_k=30, lambda_mult=0.9)

    assert vectorstore.as_retriever_kwargs["search_type"] == "mmr"
    assert vectorstore.as_retriever_kwargs["search_kwargs"] == {
        "k": 8, "fetch_k": 30, "lambda_mult": 0.9,
    }


def test_build_rag_chain_logs_retrieval_and_generation_timing(monkeypatch, caplog):
    docs = [FakeDoc("content", {"file_path": "a.py", "chunk_index": 0, "language": "python"})]
    monkeypatch.setattr(
        chain_module, "ChatAnthropic", lambda model=None: RunnableLambda(lambda prompt_value: "fake answer")
    )

    chain = chain_module.build_rag_chain(FakeVectorstore(docs), k=1, fetch_k=1)
    with caplog.at_level(logging.INFO, logger="rag-agent.retrieval"):
        chain.invoke({"question": "anything"})

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.chunks_retrieved == 1
    assert record.retrieved_files == ["a.py"]
    assert record.retrieval_seconds >= 0
    assert record.generation_seconds >= 0
