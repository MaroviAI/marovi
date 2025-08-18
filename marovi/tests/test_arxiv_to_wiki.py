import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# ---------------------------------------------------------------------------
# Provide a minimal ``pydantic`` stub if the real library is unavailable
# ---------------------------------------------------------------------------
import sys
import types

try:  # pragma: no cover - executed only when pydantic is available
    import pydantic as pydantic_stub  # type: ignore
except Exception:  # pragma: no cover - executed when pydantic is missing
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:  # minimal BaseModel placeholder
        pass

    def Field(default=None, **_kwargs):  # simple Field stub
        return default

    HttpUrl = str

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    pydantic_stub.HttpUrl = HttpUrl
    def validator(*_args, **_kwargs):  # decorator stub
        def _wrap(func):
            return func
        return _wrap
    pydantic_stub.validator = validator
    sys.modules.setdefault("pydantic", pydantic_stub)

# Stub ``feedparser`` if missing
try:  # pragma: no cover - executed only when feedparser is available
    import feedparser  # type: ignore
except Exception:  # pragma: no cover - executed when feedparser is missing
    feedparser = types.ModuleType("feedparser")
    def _parse(*_args, **_kwargs):
        return types.SimpleNamespace(entries=[])
    feedparser.parse = _parse  # type: ignore
    sys.modules.setdefault("feedparser", feedparser)

# Stub CleanTextStep to avoid importing heavy dependencies
marovi_steps = types.ModuleType("marovi.modules.steps.marovi_api")
class CleanTextStep:  # simple passthrough stub
    def __init__(self, *args, **kwargs):
        pass

    def run_with_retries(self, texts, _context):
        return texts

marovi_steps.CleanTextStep = CleanTextStep
sys.modules.setdefault("marovi.modules.steps.marovi_api", marovi_steps)

from marovi.modules.wiki import WikiClient
from marovi.pipelines.arxiv_to_wiki import (
    ArxivToWikiPipeline,
    CleanWikiStep,
    DownloadArxivStep,
    HTMLToWikiStep,
    UploadWikiStep,
)
from marovi.pipelines.context import PipelineContext
from marovi.storage.document.paper_storage import PaperStorage


def _mock_response(json_data):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json = Mock(return_value=json_data)
    return resp


def test_wiki_client_login_and_upload():
    session = Mock()
    session.get.side_effect = [
        _mock_response({"query": {"tokens": {"logintoken": "LOGIN"}}}),
        _mock_response({"query": {"tokens": {"csrftoken": "CSRF"}}}),
    ]
    session.post.side_effect = [
        _mock_response({"login": {"result": "Success"}}),
        _mock_response({"edit": {"result": "Success"}}),
    ]
    uploader = WikiClient("https://wiki.example/api.php", "user", "pass")
    uploader.session = session
    uploader.upload_page("Title", "Content")
    assert uploader.csrf_token == "CSRF"
    session.post.assert_called_with(
        "https://wiki.example/api.php",
        data={
            "action": "edit",
            "title": "Title",
            "text": "Content",
            "summary": "",
            "token": "CSRF",
            "format": "json",
        },
        timeout=30,
    )


def test_wiki_client_error_on_upload():
    session = Mock()
    session.get.side_effect = [
        _mock_response({"query": {"tokens": {"logintoken": "LOGIN"}}}),
        _mock_response({"query": {"tokens": {"csrftoken": "CSRF"}}}),
    ]
    session.post.side_effect = [
        _mock_response({"login": {"result": "Success"}}),
        _mock_response({"error": {"code": "bad"}}),
    ]
    uploader = WikiClient("https://wiki.example/api.php", "user", "pass")
    uploader.session = session
    with pytest.raises(RuntimeError):
        uploader.upload_page("Title", "Content")


def test_download_arxiv_step_reads_metadata(tmp_path):
    storage = Mock(spec=PaperStorage)
    downloader = Mock()
    downloader.download_document.return_value = str(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps({"title": "Test"}))
    step = DownloadArxivStep(storage)
    step.downloader = downloader
    outputs = step.process(["1234"], PipelineContext())
    assert outputs[0]["metadata"]["title"] == "Test"


def test_html_to_wiki_step_converts(tmp_path):
    step = HTMLToWikiStep()
    arxiv_id = "1234"
    html_path = tmp_path / f"{arxiv_id}.html"
    html_path.write_text("<p>Hello</p>")
    item = {"arxiv_id": arxiv_id, "paper_dir": tmp_path, "metadata": {}}
    with patch("marovi.pipelines.arxiv_to_wiki.PandocParser") as MockParser:
        parser = Mock()
        MockParser.return_value = parser
        def fake_convert(output_file):
            Path(output_file).write_text("converted")
        parser.convert_html_to_wiki.side_effect = fake_convert
        outputs = step.process([item], PipelineContext())
    assert outputs[0]["wiki_text"] == "converted"


def test_clean_wiki_step_runs_and_writes(tmp_path):
    step = CleanWikiStep()
    wiki_file = tmp_path / "file.wiki"
    wiki_file.write_text("raw")
    item = {"wiki_text": "raw", "wiki_path": wiki_file}
    with patch.object(step.clean_step, "run_with_retries", return_value=["cleaned"]) as mock_run:
        outputs = step.process([item], PipelineContext())
    assert outputs[0]["cleaned_text"] == "cleaned"
    assert wiki_file.read_text() == "cleaned"
    mock_run.assert_called_once()


def test_upload_wiki_step_calls_uploader():
    uploader = Mock(spec=WikiClient)
    uploader.upload_page.return_value = None
    step = UploadWikiStep(uploader)
    item = {"arxiv_id": "123", "cleaned_text": "text", "metadata": {"title": "Title"}}
    outputs = step.process([item], PipelineContext())
    uploader.upload_page.assert_called_with("Title", "text")
    assert outputs[0]["uploaded"] is True


def test_arxiv_to_wiki_pipeline_runs(tmp_path):
    storage = Mock(spec=PaperStorage)
    pipeline = ArxivToWikiPipeline(storage, "https://wiki.example/api.php")
    download_result = [{"arxiv_id": "1", "paper_dir": tmp_path, "metadata": {}}]
    html_result = [{"arxiv_id": "1", "paper_dir": tmp_path, "metadata": {}, "wiki_text": "raw"}]
    clean_result = [{"arxiv_id": "1", "paper_dir": tmp_path, "metadata": {}, "cleaned_text": "clean"}]
    upload_result = [{"arxiv_id": "1", "uploaded": True}]
    with patch.object(DownloadArxivStep, "process", return_value=download_result), \
         patch.object(HTMLToWikiStep, "process", return_value=html_result), \
         patch.object(CleanWikiStep, "process", return_value=clean_result), \
         patch.object(UploadWikiStep, "process", return_value=upload_result):
        outputs = pipeline.run(["1"], PipelineContext())
    assert outputs[0]["uploaded"] is True
