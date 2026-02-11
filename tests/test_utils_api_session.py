from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from mgnify_methods.utils import api as api_module


def test_retrieve_summary_old(monkeypatch, tmp_path):
    download = SimpleNamespace(
        description=SimpleNamespace(label="Taxonomic assignments SSU"),
        links=SimpleNamespace(self=SimpleNamespace(url="http://example.com/file.tsv")),
        alias="test",
    )

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iterate(self, _):
            return [download]

    monkeypatch.setattr(api_module, "APISession", lambda *_args, **_kwargs: DummySession())

    def fake_urlretrieve(_url, path):
        Path(path).write_text("ok")
        return path, None

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    api_module.retrieve_summary_old("MGYS00000000")
