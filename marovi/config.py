import os
from pathlib import Path

# Base path for all storage (default: /data)
BASE_PATH = Path(os.getenv("DATA_STORAGE_PATH", 
    Path(__file__).parent.parent / "data")).resolve()

# Storage backend selection (default: file-based)
USE_DATABASE = os.getenv("USE_DATABASE", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///database/papers.db")

# Languages supported by the system
LANGUAGES = ["en", "es", "fr", "ja", "zh", "ko", "de", "hi"] # ["ar", "pt", "ru"] next languages to add (in order of priority)

# Ensure the base path exists
BASE_PATH.mkdir(exist_ok=True)
