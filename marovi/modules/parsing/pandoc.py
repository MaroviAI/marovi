import os
import subprocess
import requests
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from marovi.modules.parsing.base_parser import BaseParser

class PandocParser(BaseParser):
    """
    A parser that uses Pandoc to convert between different document formats.
    Supports various input and output formats that Pandoc can handle.
    """

    def __init__(self, content: str = "", pandoc_path: str = "pandoc"):
        """
        Initialize the parser with optional content and path to Pandoc binary.
        
        Args:
            content (str): The content to be parsed. If not provided, can be loaded later.
            pandoc_path (str): Path to the Pandoc binary. Default is 'pandoc'.
        """
        super().__init__(content)
        self.pandoc_path = pandoc_path
        
    def fetch_from_url(self, url: str) -> None:
        """
        Fetches content from a URL and loads it into the parser.
        
        Args:
            url (str): The URL to fetch content from.
            
        Raises:
            Exception: If the fetch operation fails.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.load_content(response.text)
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch content from {url}: {e}")
    
    def parse(self) -> str:
        """
        Base parsing method - in this case, returns the raw content.
        For actual conversion, use the convert() method.
        
        Returns:
            str: The raw content.
        """
        return self.content
    
    def convert(self, 
                from_format: str, 
                to_format: str, 
                output_file: Optional[str] = None,
                bib_file: Optional[str] = None,
                csl_file: Optional[str] = None,
                extra_args: Optional[List[str]] = None) -> str:
        """
        Converts content from one format to another using Pandoc.
        
        Args:
            from_format (str): Input format (e.g., 'html', 'markdown', 'latex')
            to_format (str): Output format (e.g., 'mediawiki', 'docx', 'pdf')
            output_file (str, optional): Path to save the output file. If None, a temporary file is created.
            bib_file (str, optional): Path to bibliography file (.bib or .bbl)
            csl_file (str, optional): Path to CSL file for citation styling
            extra_args (List[str], optional): Additional arguments to pass to Pandoc
            
        Returns:
            str: Path to the output file
        """
        try:
            # Create a temporary file for the input content
            with NamedTemporaryFile(delete=False, suffix=f".{from_format}") as temp_input_file:
                temp_input_path = temp_input_file.name
                temp_input_file.write(self.content.encode("utf-8"))
            
            # Create output path
            if output_file:
                output_path = output_file
            else:
                output_path = f"/tmp/{Path(temp_input_path).stem}.{to_format}"
            
            # Build Pandoc command
            pandoc_command = [
                self.pandoc_path,
                temp_input_path, 
                "-f", from_format, 
                "-t", to_format, 
                "-o", output_path
            ]
            
            # Add bibliography if provided
            if bib_file:
                # Convert .bbl to .bib if necessary
                if bib_file.endswith(".bbl"):
                    bib_file = self._convert_bbl_to_bib(bib_file)
                pandoc_command.extend(["--bibliography", bib_file])
            
            # Add CSL file if provided
            if csl_file:
                pandoc_command.extend(["--csl", csl_file])
            
            # Add any extra arguments
            if extra_args:
                pandoc_command.extend(extra_args)
            
            # Run Pandoc conversion
            subprocess.run(pandoc_command, check=True)
            
            # Clean up
            os.remove(temp_input_path)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Pandoc conversion failed: {e}")
        except Exception as e:
            raise Exception(f"Conversion error: {e}")
    
    def convert_html_to_wiki(self, 
                             bib_file: Optional[str] = None, 
                             csl_file: Optional[str] = None) -> str:
        """
        Convenience method for HTML to MediaWiki conversion.
        
        Args:
            bib_file (str, optional): Path to bibliography file
            csl_file (str, optional): Path to CSL file
            
        Returns:
            str: Path to the output wiki file
        """
        return self.convert(
            from_format="html",
            to_format="mediawiki",
            bib_file=bib_file,
            csl_file=csl_file
        )
    
    def _convert_bbl_to_bib(self, bbl_path: str) -> str:
        """
        Converts a .bbl file to a .bib file for Pandoc compatibility.
        
        Args:
            bbl_path (str): Path to the .bbl file
            
        Returns:
            str: Path to the generated .bib file
        """
        import re
        
        bbl_path = Path(bbl_path)
        bib_output_path = bbl_path.with_suffix('.bib')
        
        try:
            bbl_content = bbl_path.read_text(encoding="utf-8").splitlines()
            
            bib_entries = []
            entry_pattern = re.compile(r"\\bibitem\{(.+?)\}")
            
            for line in bbl_content:
                match = entry_pattern.search(line)
                if match:
                    cite_key = match.group(1)
                    bib_entries.append(f"@misc{{{cite_key},\n  title={{Placeholder Title}},\n  author={{Unknown}},\n  year={{2024}}\n}}\n")
            
            bib_output_path.write_text('\n'.join(bib_entries), encoding="utf-8")
            return str(bib_output_path)
            
        except Exception as e:
            raise Exception(f"Failed to convert .bbl to .bib: {e}")
