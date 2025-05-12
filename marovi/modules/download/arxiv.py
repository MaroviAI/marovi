import logging
import requests
import tarfile
import io
import feedparser
from pathlib import Path
from typing import Dict, Optional

from marovi.modules.download.core import Downloader
from marovi.storage.document.paper_storage import PaperStorage
from marovi.storage.schemas.document import PaperMetadata

# ArXiv API Base URL
ARXIV_API_BASE = "http://export.arxiv.org/api/query?id_list={}"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ArXivDownloader(Downloader):
    """ArXiv-specific document downloader."""
    
    def __init__(self, storage: PaperStorage):
        """Initialize the ArXiv downloader with storage backend."""
        super().__init__(storage)
        
    def get_metadata(self, document_id: str) -> Optional[Dict]:
        """
        Get metadata for an ArXiv paper.
        
        Args:
            document_id (str): ArXiv paper ID (e.g., "2104.08000")
            
        Returns:
            dict: Paper metadata, or None if unavailable
        """
        try:
            response = feedparser.parse(ARXIV_API_BASE.format(document_id))

            if not response.entries:
                logging.warning(f"⚠️ No metadata found for {document_id}.")
                return None

            entry = response.entries[0]

            # Construct URLs
            pdf_url = f"https://arxiv.org/pdf/{document_id}.pdf"
            html_url = f"https://ar5iv.org/html/{document_id}"  # Better HTML rendering
            latex_url = f"https://arxiv.org/src/{document_id}"  # Triggers .tar.gz download
            source_url = f"https://arxiv.org/abs/{document_id}"  # Main ArXiv page

            return {
                "paper_id": document_id,
                "title": entry.get("title", "Unknown Title").strip(),
                "pdf_url": pdf_url,
                "html_url": html_url,
                "latex_url": latex_url,
                "source_url": source_url,
            }

        except Exception as e:
            logging.error(f"❌ Failed to fetch metadata for {document_id}: {e}")
            return None
        
    def download_document(self, document_id: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Download an ArXiv paper and its associated files.
        
        Args:
            document_id (str): ArXiv paper ID
            output_dir (Path, optional): Custom output directory
            
        Returns:
            Path: Path to the downloaded paper directory, or None if download failed
        """
        # Step 1: Fetch metadata
        metadata = self.get_metadata(document_id)
        if not metadata:
            logging.warning(f"⚠️ Skipping {document_id} - No metadata found.")
            return None
            
        # Step 2: Create PaperMetadata object
        paper_metadata = PaperMetadata(
            paper_id=document_id,
            title=metadata["title"],
            pdf_url=metadata["pdf_url"],
            html_url=metadata["html_url"],
            source_url=metadata["source_url"],
            tex_url=metadata["latex_url"]
        )
        
        # Step 3: Set up directory and download PDF/HTML via storage
        paper_folder = self.storage.setup_document_directory(paper_metadata)
        if not paper_folder:
            logging.warning(f"⚠️ Failed to set up document directory for {document_id}.")
            return None
            
        # Step 4: Download PDF if available
        pdf_path = paper_folder / f"{document_id}.pdf"
        if metadata["pdf_url"]:
            self.download_file(metadata["pdf_url"], pdf_path)
            
        # Step 5: Download HTML version if available
        html_path = paper_folder / f"{document_id}.html"
        if metadata["html_url"]:
            self.download_file(metadata["html_url"], html_path)
            
        # Step 6: Download & Extract LaTeX (if available)
        self.download_and_extract_latex(document_id, paper_folder)
        
        logging.info(f"📄 Successfully downloaded ArXiv paper {document_id} - Stored in {paper_folder}")
        return paper_folder
        
    def download_and_extract_latex(self, arxiv_id: str, output_dir: Path) -> Path:
        """
        Downloads and extracts LaTeX source from ArXiv if available.
        Saves it in a 'latex/' subfolder inside the output directory.

        Args:
            arxiv_id (str): The ArXiv paper ID.
            output_dir (Path): Directory where the paper is stored.

        Returns:
            Path: Path to the extracted LaTeX files, or None if the download fails.
        """
        output_latex_dir = output_dir / "latex"
        output_latex_dir.mkdir(parents=True, exist_ok=True)

        latex_url = f"https://arxiv.org/src/{arxiv_id}"

        try:
            response = requests.get(latex_url, stream=True, timeout=30)
            response.raise_for_status()

            # Extract LaTeX files from .tar.gz
            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
                tar.extractall(path=output_latex_dir)

            logging.info(f"✅ LaTeX source extracted to: {output_latex_dir}")
            return output_latex_dir

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Failed to download LaTeX for {arxiv_id}: {e}")
            return None
        except tarfile.ReadError:
            logging.warning(f"⚠️ Invalid or missing LaTeX archive for {arxiv_id}")
            return None
