import json
from pathlib import Path

import pytest

from drishti.bench.synth import generate
from drishti.core.domain import merged_pack
from drishti.ingest.ingestor import DirectoryAdapter, Ingestor
from drishti.pipeline.orchestrator import Pipeline
from drishti.store.db import Store


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    return generate(tmp_path_factory.mktemp("corpus"), seed=42, split="test")


@pytest.fixture(scope="session")
def gold(corpus) -> dict:
    return json.loads((corpus / "gold.json").read_text())


@pytest.fixture(scope="session")
def processed(corpus, tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    st = Store(work / "d.db")
    st.create_case("T", "Test", "fusion")
    Ingestor(st, work / "blobs").ingest_all("T", DirectoryAdapter(corpus / "docs"))
    metrics = Pipeline(st, merged_pack(), workers=2, use_processes=False).run("T")
    return st, "T", metrics
