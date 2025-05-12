"""
Download modules for acquiring document data from various sources.
"""

from marovi.modules.download.core import Downloader
from marovi.modules.download.arxiv import ArXivDownloader

__all__ = ['Downloader', 'ArXivDownloader']
