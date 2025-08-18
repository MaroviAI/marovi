"""Pipeline for converting ArXiv papers into wiki pages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from marovi.modules.download import ArXivDownloader
from marovi.modules.parsing.pandoc import PandocParser
from marovi.modules.wiki import WikiClient
from marovi.storage.document.paper_storage import PaperStorage
from marovi.pipelines.core import Pipeline, PipelineStep
from marovi.pipelines.context import PipelineContext
from marovi.modules.steps.marovi_api import CleanTextStep

logger = logging.getLogger(__name__)


class DownloadArxivStep(PipelineStep[str, Dict[str, Any]]):
    """Download an ArXiv paper and store its metadata."""

    def __init__(self, storage: PaperStorage, step_id: str | None = None) -> None:
        super().__init__(step_id=step_id or "download_arxiv")
        self.downloader = ArXivDownloader(storage)

    def process(self, inputs: List[str], context: PipelineContext) -> List[Dict[str, Any]]:
        results = []
        for arxiv_id in inputs:
            paper_dir = self.downloader.download_document(arxiv_id)
            if not paper_dir:
                logger.warning("Skipping %s: download failed", arxiv_id)
                continue
            metadata_path = Path(paper_dir) / "metadata.json"
            metadata: Dict[str, Any] = {}
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            results.append({
                "arxiv_id": arxiv_id,
                "paper_dir": Path(paper_dir),
                "metadata": metadata,
            })
        return results


class HTMLToWikiStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
    """Convert downloaded HTML to wiki markup using Pandoc."""

    def process(self, inputs: List[Dict[str, Any]], context: PipelineContext) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for item in inputs:
            arxiv_id = item["arxiv_id"]
            paper_dir: Path = item["paper_dir"]
            html_path = paper_dir / f"{arxiv_id}.html"
            if not html_path.exists():
                logger.warning("HTML file not found for %s", arxiv_id)
                continue
            html_content = html_path.read_text(encoding="utf-8")
            parser = PandocParser(html_content)
            wiki_dir = paper_dir / "wiki"
            wiki_dir.mkdir(exist_ok=True)
            wiki_output = wiki_dir / f"{arxiv_id}.wiki"
            parser.convert_html_to_wiki(output_file=str(wiki_output))
            wiki_text = wiki_output.read_text(encoding="utf-8")
            item["wiki_path"] = wiki_output
            item["wiki_text"] = wiki_text
            outputs.append(item)
        return outputs


class CleanWikiStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
    """Use LLM to clean the wiki markup."""

    def __init__(self, provider: str = "openai", step_id: str | None = None) -> None:
        super().__init__(batch_handling="inherent", step_id=step_id or "clean_wiki")
        self.clean_step = CleanTextStep(format_value="wiki", provider=provider)

    def process(self, inputs: List[Dict[str, Any]], context: PipelineContext) -> List[Dict[str, Any]]:
        texts = [item["wiki_text"] for item in inputs]
        cleaned_texts = self.clean_step.run_with_retries(texts, context)
        outputs: List[Dict[str, Any]] = []
        for item, cleaned in zip(inputs, cleaned_texts):
            item["cleaned_text"] = cleaned
            wiki_path = item.get("wiki_path")
            if wiki_path:
                Path(wiki_path).write_text(cleaned, encoding="utf-8")
            outputs.append(item)
        return outputs


class UploadWikiStep(PipelineStep[Dict[str, Any], Dict[str, Any]]):
    """Upload cleaned wiki text to a remote MediaWiki instance."""

    def __init__(self, uploader: WikiClient, step_id: str | None = None) -> None:
        super().__init__(step_id=step_id or "upload_wiki")
        self.uploader = uploader

    def process(self, inputs: List[Dict[str, Any]], context: PipelineContext) -> List[Dict[str, Any]]:
        outputs = []
        for item in inputs:
            title = item.get("metadata", {}).get("title") or item["arxiv_id"]
            text = item.get("cleaned_text")
            if not text:
                logger.warning("No text to upload for %s", title)
                continue
            try:
                self.uploader.upload_page(title, text)
                item["uploaded"] = True
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to upload %s: %s", title, exc)
                item["uploaded"] = False
                item["error"] = str(exc)
            outputs.append(item)
        return outputs


class ArxivToWikiPipeline(Pipeline):
    """Pipeline that downloads an ArXiv paper and publishes it to a wiki."""

    def __init__(self, storage: PaperStorage, wiki_api_url: str, provider: str = "openai") -> None:
        downloader = DownloadArxivStep(storage)
        html_to_wiki = HTMLToWikiStep()
        clean = CleanWikiStep(provider=provider)
        uploader = UploadWikiStep(WikiClient(wiki_api_url))
        super().__init__(steps=[downloader, html_to_wiki, clean, uploader], name="arxiv_to_wiki")
