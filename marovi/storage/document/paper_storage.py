import os
import json
import logging
from pathlib import Path

from marovi.storage.document.core import StorageBase
from marovi.storage.schemas.document import PaperMetadata
from marovi.config import BASE_PATH

DEFAULT_SUBDIRECTORIES = ["latex", "images", "metadata", "wiki", "csv", "translations", "po", "production"]

class PaperStorage(StorageBase):
    """File-based storage for research papers."""

    def __init__(self, base_path: str):
        """Initialize FileStorage with a base path and ensure structure exists."""
        self.base_path = Path(BASE_PATH) / base_path
        self.global_index_file = self.base_path / "index.json"
        logging.info(f"📂 Ensuring base directory exists: {self.base_path.absolute()}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"📄 Index file path: {self.global_index_file.absolute()}")
        
        super().__init__(base_path)
        self.initialize()

    def get_base_path(self):
        """Return the base path of the storage."""
        return self.base_path

    def ensure_directories_exist(self):
        """Ensure the base paper directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def ensure_index_exists(self):
        """Ensure index.json exists and is valid JSON."""
        if not self.global_index_file.exists():
            with open(self.global_index_file, "w") as index_file:
                json.dump({}, index_file, indent=4)
        else:
            try:
                with open(self.global_index_file, "r") as index_file:
                    data = json.load(index_file)
                    if not isinstance(data, dict):
                        raise ValueError("Invalid index.json format")
            except (json.JSONDecodeError, ValueError):
                print("Warning: index.json was invalid. Resetting...")
                with open(self.global_index_file, "w") as index_file:
                    json.dump({}, index_file, indent=4)

    def generate_document_id(self, existing_id=None):
        """Generate a unique 6-digit paper ID if not provided."""
        if existing_id:
            return existing_id

        with open(self.global_index_file, "r") as index_file:
            index_data = json.load(index_file)

        new_id = f"{len(index_data) + 1:06d}"
        return new_id

    def truncate_title(self, title):
        """Truncate title to a max of 30 characters for filenames."""
        return title[:30].strip().replace(" ", "_")

    def setup_document_directory(self, metadata: PaperMetadata):
        """Set up the directory structure and metadata file."""
        paper_id = self.generate_document_id(metadata.paper_id)
        paper_folder = self.base_path / paper_id
        paper_folder.mkdir(exist_ok=True)

        # Create subdirectories
        for subdir in DEFAULT_SUBDIRECTORIES:
            (paper_folder / subdir).mkdir(exist_ok=True)

        # Save full title inside the truncated title file
        truncated_title = self.truncate_title(metadata.title)
        title_file_path = paper_folder / f"{truncated_title}.txt"
        title_file_path.write_text(metadata.title)

        # Convert Pydantic model to a JSON-serializable dict
        metadata_dict = metadata.model_dump(mode="json") if hasattr(metadata, "model_dump") else metadata.dict()

        # Save metadata.json
        metadata_path = paper_folder / "metadata.json"
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata_dict, metadata_file, indent=4)

        # Update global index
        self.update_global_index(paper_id, metadata.title, str(paper_folder))

        return paper_folder

    def update_global_index(self, paper_id, title, paper_folder):
        """Updates the global index with new paper metadata."""
        try:
            with open(self.global_index_file, "r") as f:
                index_data = json.load(f)
                if not isinstance(index_data, dict):
                    raise ValueError("Corrupt index.json, resetting...")
        except (json.JSONDecodeError, ValueError):
            print("Warning: index.json was invalid. Resetting...")
            index_data = {}

        # Convert Path to string if needed
        if isinstance(paper_folder, Path):
            paper_folder = str(paper_folder)

        index_data[paper_id] = {"title": title, "path": paper_folder}

        with open(self.global_index_file, "w") as f:
            json.dump(index_data, f, indent=4)

        print(f"Updated index with {paper_id} - {title}")

    def get_paper_metadata(self, paper_id):
        """Retrieve metadata for a specific paper."""
        metadata_path = self.base_path / paper_id / "metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r") as f:
            return json.load(f)

    def list_papers(self):
        """Return a list of all stored papers."""
        if not self.global_index_file.exists():
            return {}
        with open(self.global_index_file, "r") as f:
            return json.load(f)
