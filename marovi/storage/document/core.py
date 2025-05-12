import os
import json
from pathlib import Path
from abc import ABC, abstractmethod
from marovi.config import BASE_PATH

class StorageBase(ABC):
    """Abstract base class for storage systems."""

    def __init__(self, sub_path):
        """Initialize storage for a specific sub-path (e.g., papers, books)."""
        self.base_path = Path(BASE_PATH) / sub_path
        self.base_path.mkdir(exist_ok=True)
        self.index_path = self.base_path / "index.json"

    def initialize(self):
        """Explicitly initialize the index file. Called by subclasses after setting up attributes."""
        self.ensure_index_exists()

    @abstractmethod
    def setup_document_directory(self, metadata):
        """Set up the directory structure or database entry."""
        pass

    @abstractmethod
    def get_paper_metadata(self, paper_id):
        """Retrieve metadata for a paper."""
        pass

    @abstractmethod
    def list_papers(self):
        """List all stored papers."""
        pass

    def ensure_index_exists(self):
        """Ensure index.json exists and is valid JSON."""
        if not self.index_path.exists():
            self._write_index({})  # Initialize with empty JSON
        else:
            try:
                data = json.loads(self.index_path.read_text())
                if not isinstance(data, dict):
                    raise ValueError("Invalid index.json format")
            except (json.JSONDecodeError, ValueError):
                print("Warning: index.json was invalid. Resetting...")
                self._write_index({})

    def _write_index(self, data):
        """Safely write index.json, ensuring atomic file updates."""
        temp_path = self.index_path.with_suffix('.tmp')
        temp_path.write_text(json.dumps(data, indent=4))
        temp_path.replace(self.index_path)  # Atomic replace

    def generate_document_id(self, existing_id=None):
        """Generate a unique 6-digit document ID if no ID is provided."""
        if existing_id:
            return existing_id

        index_data = json.loads(self.index_path.read_text())
        return f"{len(index_data) + 1:06d}"
