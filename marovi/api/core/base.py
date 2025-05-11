"""
Base definitions for Marovi API core.

This module contains essential enums and types that are used throughout
the Marovi API and must be available without circular import issues.
"""

import enum
from typing import Dict, List, Optional, Any, Type, Union, Callable

class ServiceType(enum.Enum):
    """Types of services supported by the Marovi API."""
    LLM = "llm"
    TRANSLATION = "translation"
    CUSTOM = "custom" 