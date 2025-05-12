from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class PaperMetadata(BaseModel):
    """Validates input metadata for processing a research paper."""
    title: str  # Required
    source_url: Optional[HttpUrl] = None  # Base URL (e.g., ArXiv, Journal, etc.)
    pdf_url: Optional[HttpUrl] = None  # Direct PDF link (if available)
    tex_url: Optional[HttpUrl] = None  # Direct LaTeX source link (if available)
    html_url: Optional[HttpUrl] = None  # Optional: HTML version of paper
    paper_id: Optional[str] = None  # Will be auto-generated if not provided

    class Config:
        extra = "forbid"  # Prevents unexpected fields


    def model_dump(self, **kwargs):  
        serialized_data = super().model_dump(**kwargs)
        # Convert all URL fields to strings
        for key in ["source_url", "pdf_url", "tex_url", "html_url"]:
            if serialized_data.get(key):
                serialized_data[key] = str(serialized_data[key])
        return serialized_data