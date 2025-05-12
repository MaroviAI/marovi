from abc import ABC, abstractmethod
from pathlib import Path
import os
import requests
import logging
from typing import Dict, Optional
from marovi.storage.document.paper_storage import PaperStorage

class Downloader(ABC):
    """Base class for document downloaders."""
    
    def __init__(self, storage: PaperStorage):
        """
        Initialize the downloader with a storage backend.
        
        Args:
            storage (PaperStorage): Storage backend to save downloaded documents
        """
        self.storage = storage
    
    def download_file(self, url, save_path):
        """
        Download a file from a given URL and save it locally.
        
        Args:
            url (str): URL to download from
            save_path (str or Path): Path where the file should be saved
            
        Returns:
            Path: Path to the downloaded file, or None if download failed
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logging.info(f"Downloaded: {save_path}")
            return save_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")
            return None
    
    @abstractmethod
    def download_document(self, document_id: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Download a document by its identifier.
        
        Args:
            document_id (str): Unique identifier for the document
            output_dir (Path, optional): Custom output directory. If None, use default storage path.
            
        Returns:
            Path: Path to the downloaded document directory, or None if download failed
        """
        pass
    
    @abstractmethod
    def get_metadata(self, document_id: str) -> Optional[Dict]:
        """
        Get metadata for a document.
        
        Args:
            document_id (str): Unique identifier for the document
            
        Returns:
            dict: Document metadata, or None if unavailable
        """
        pass
