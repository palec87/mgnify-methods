from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from mgnify_methods.utils import api as api_module


def test_retrieve_summary_downloads(tmp_path, monkeypatch):
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

    out_path = api_module.retrieve_summary("MGYS00000000", out_dir=str(tmp_path))

    assert Path(out_path).exists()


def test_retrieve_summary_no_match(tmp_path, monkeypatch):
    download = SimpleNamespace(
        description=SimpleNamespace(label="Other"),
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

    with pytest.raises(FileNotFoundError):
        api_module.retrieve_summary("MGYS00000000", out_dir=str(tmp_path))


def test_get_mgnify_metadata(monkeypatch):
    sample_json = {
        "id": "S1",
        "attributes": {
            "sample-metadata": [{"key": "env", "value": "marine"}],
            "other": "value",
        },
    }

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iterate(self, _):
            return [SimpleNamespace(json=sample_json)]

    monkeypatch.setattr(api_module, "APISession", lambda *_args, **_kwargs: DummySession())

    df = api_module.get_mgnify_metadata("MGYS00000000")

    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns
    assert "env" in df.columns
    assert "study" in df.columns
