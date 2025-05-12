from marovi.config import USE_DATABASE
from marovi.storage.document.paper_storage import PaperStorage
# Future: from marovi.storage.db_storage import DatabaseStorage

def get_storage(base_path: str):
    """Factory function to select storage backend."""
    if USE_DATABASE:
        return DatabaseStorage()  # Placeholder for future DB storage
    return FileStorage(base_path=base_path)
