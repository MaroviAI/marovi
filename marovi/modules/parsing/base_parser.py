from abc import ABC, abstractmethod
from typing import Any, Union, Dict, List, Optional
from pathlib import Path

class BaseParser(ABC):
    """
    A base class for parsers that handle different content types.
    Provides core functionality for loading content and parsing it into structured data.
    """

    def __init__(self, content: str = ""):
        """
        Initializes the parser with optional content to be parsed.
        
        Args:
            content (str): The content to be parsed. If not provided, can be loaded later.
        """
        self.content = content

    def load_content(self, content: str) -> None:
        """
        Loads content into the parser.
        
        Args:
            content (str): The content to be loaded.
        """
        self.content = content

    def load_from_file(self, file_path: str) -> None:
        """
        Loads content from a file and initializes the parser.
        
        Args:
            file_path (str): The path to the file to load content from.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            self.content = file.read()

    @abstractmethod
    def parse(self) -> Union[Dict[str, Any], List[Any], Any]:
        """
        Abstract method to parse the content and return a structured data representation.
        Subclasses must implement this method to provide their specific parsing logic.
        
        Returns:
            Union[Dict[str, Any], List[Any], Any]: The structured data representation of the parsed content.
        """
        pass

    def save_to_file(self, file_path: str, data: Any) -> None:
        """
        Saves the structured data to a file.
        
        Args:
            file_path (str): The path to the file where the data should be saved.
            data (Any): The structured data to save.
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            if isinstance(data, (dict, list)):
                file.write(str(data))
            else:
                file.write(data)

    def get_content(self) -> str:
        """
        Returns the raw content loaded into the parser.
        
        Returns:
            str: The raw content.
        """
        return self.content
    
    def fetch_from_url(self, url: str) -> None:
        """
        Fetches content from a URL and loads it into the parser.
        
        Args:
            url (str): The URL to fetch content from.
        
        Raises:
            NotImplementedError: If the parser does not implement URL fetching.
        """
        raise NotImplementedError("This parser does not support fetching from URLs")
