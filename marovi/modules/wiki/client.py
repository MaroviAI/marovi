"""Utilities for interacting with a MediaWiki instance.

This client supports downloading pages, editing content and uploading files,
which allows automated workflows (e.g. bots) to manage wiki content.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class WikiClient:
    """Client for downloading, uploading and editing wiki content.

    The client handles the basic login/token workflow required by the
    MediaWiki API. Credentials are expected to be provided either directly or
    through environment variables ``MAROVI_WIKI_USER`` and
    ``MAROVI_WIKI_PASSWORD``.
    """

    def __init__(
        self,
        api_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.api_url = api_url
        self.username = username or os.getenv("MAROVI_WIKI_USER")
        self.password = password or os.getenv("MAROVI_WIKI_PASSWORD")
        self.session = requests.Session()
        self.csrf_token: Optional[str] = None

    def _get_token(self, token_type: str) -> str:
        """Fetch a token of ``token_type`` from the wiki."""
        resp = self.session.get(
            self.api_url,
            params={"action": "query", "meta": "tokens", "type": token_type, "format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["query"]["tokens"][f"{token_type}token"]

    def login(self) -> None:
        """Login to the wiki and store a CSRF token for later edits."""
        if not self.username or not self.password:
            raise ValueError("Wiki credentials are not provided")

        login_token = self._get_token("login")
        resp = self.session.post(
            self.api_url,
            data={
                "action": "login",
                "lgname": self.username,
                "lgpassword": self.password,
                "lgtoken": login_token,
                "format": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        self.csrf_token = self._get_token("csrf")
        logger.info("Authenticated with wiki API")

    def download_page(self, title: str) -> str:
        """Download the raw wiki markup of ``title``.

        Args:
            title: Page title to fetch.

        Returns:
            The raw wiki text of the requested page.  An empty string is
            returned if the page does not exist or has no content.
        """
        resp = self.session.get(
            self.api_url,
            params={
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": title,
                "format": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        return (
            page.get("revisions", [{}])[0]
            .get("slots", {})
            .get("main", {})
            .get("*", "")
        )

    def edit_page(self, title: str, content: str, summary: str = "") -> None:
        """Create or replace a wiki page with ``content``."""
        if not self.csrf_token:
            self.login()

        resp = self.session.post(
            self.api_url,
            data={
                "action": "edit",
                "title": title,
                "text": content,
                "summary": summary,
                "token": self.csrf_token,
                "format": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Wiki edit failed: {data['error']}")
        logger.info("Edited page '%s'", title)

    def upload_page(self, title: str, content: str, summary: str = "") -> None:
        """Backward compatible wrapper for :meth:`edit_page`."""
        self.edit_page(title, content, summary)

    def upload_file(self, filename: str, file_path: str, comment: str = "") -> None:
        """Upload a file to the wiki.

        Args:
            filename: Destination filename on the wiki.
            file_path: Local path to the file to upload.
            comment: Optional upload comment.
        """
        if not self.csrf_token:
            self.login()

        with open(file_path, "rb") as file_obj:
            resp = self.session.post(
                self.api_url,
                data={
                    "action": "upload",
                    "filename": filename,
                    "comment": comment,
                    "ignorewarnings": 1,
                    "token": self.csrf_token,
                    "format": "json",
                },
                files={"file": file_obj},
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Wiki file upload failed: {data['error']}")
        logger.info("Uploaded file '%s'", filename)
