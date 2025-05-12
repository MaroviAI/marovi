import polib
from typing import List, Tuple, Dict, Optional, Any, Union
from pathlib import Path

from marovi.modules.parsing.base_parser import BaseParser
from marovi.config import LANGUAGES

class POParser(BaseParser):
    """
    Parser for .po translation files that handles message IDs and strings.
    Compatible with MediaWiki translation format.
    """

    def __init__(self, content: str = ""):
        """
        Initialize the parser.
        
        Args:
            content (str): Optional raw content of a PO file.
        """
        super().__init__(content)
        self.po_file: polib.POFile = None
        self.entries: List[polib.POEntry] = []
        self.source_language: str = "en"
        self.target_language: str = ""
        
        # Initialize from content if provided
        if content:
            self.po_file = polib.pofile(content)
            self.entries = [entry for entry in self.po_file 
                          if not entry.obsolete and entry.msgid.strip()]

    def load_from_file(self, file_path: str, include_translated: bool = False):
        """
        Load PO content from a file.
        
        Args:
            file_path: Path to PO file
            include_translated: If True, include already translated entries
        """
        self.po_file = polib.pofile(file_path)
        if include_translated:
            self.entries = [entry for entry in self.po_file 
                          if not entry.obsolete and entry.msgid.strip()]
        else:
            self.entries = [entry for entry in self.po_file.untranslated_entries() 
                          if not entry.obsolete and entry.msgid.strip()]
        
        super().load_from_file(file_path)

    def parse(self) -> Dict[str, str]:
        """
        Parse the PO file and return a dictionary of message IDs and their content.
        
        Returns:
            Dict[str, str]: Dictionary mapping message IDs to message strings
        """
        if not self.po_file:
            return {}
        
        return {entry.msgid: entry.msgstr for entry in self.entries}

    def get_messages(self) -> List[Tuple[str, str]]:
        """
        Get pairs of message IDs and their content.
        
        Returns:
            List of (msgid, msgstr) tuples for untranslated entries
        """
        if not self.entries:
            return []
        return [(entry.msgid, entry.msgstr) for entry in self.entries]

    def get_messages_with_translations(self) -> List[Tuple[str, str, str]]:
        """
        Get triples of message IDs, their content, and existing translations.
        
        Returns:
            List of (msgid, msgstr, existing_translation) tuples
        """
        if not self.entries:
            return []
        return [(entry.msgid, entry.msgstr, entry.msgstr) for entry in self.entries]

    def save_translated_po(self, output_file: str, translations: List[str]):
        """
        Save PO file with translations added.
        
        Args:
            output_file: Path to output PO file
            translations: List of translated texts
        """
        if len(translations) != len(self.entries):
            raise ValueError(f"Number of translations ({len(translations)}) must match number of entries ({len(self.entries)})")

        # Update translations in the PO file
        for entry, translation in zip(self.entries, translations):
            entry.msgstr = translation

        # Save to file
        self.po_file.save(output_file)

    def compile_translations(self, include_msgids: bool = False) -> str:
        """
        Compile all translations into a single document.
        
        Args:
            include_msgids: If True, include message IDs as headers before translations
            
        Returns:
            str: Combined translations with newlines between entries
        """
        if not self.po_file:
            raise ValueError("No PO file loaded")
            
        translations = []
        for entry in self.po_file:
            if entry.msgstr.strip():  # Only include non-empty translations
                if include_msgids:
                    translations.append(f"# {entry.msgid}")
                translations.append(entry.msgstr)
            
        return "\n\n".join(translations)
    
    def set_languages(self, source_language: str = "en", target_language: str = ""):
        """
        Set source and target languages for translation.
        
        Args:
            source_language: Source language code (e.g., 'en')
            target_language: Target language code (e.g., 'es')
        """
        if source_language not in LANGUAGES:
            raise ValueError(f"Source language {source_language} not supported. Available: {LANGUAGES}")
        
        if target_language and target_language not in LANGUAGES:
            raise ValueError(f"Target language {target_language} not supported. Available: {LANGUAGES}")
        
        self.source_language = source_language
        self.target_language = target_language
    
    def auto_translate(self, translation_service: Any) -> List[str]:
        """
        Automatically translate all untranslated entries using a translation service.
        
        Args:
            translation_service: An object that provides translation functionality
            
        Returns:
            List[str]: List of translated strings
            
        Note:
            translation_service should have a translate method that accepts:
            - text: str or List[str] - text to translate
            - source_language: str - source language code
            - target_language: str - target language code
        """
        if not self.target_language:
            raise ValueError("Target language not set. Call set_languages() first.")
        
        if not self.entries:
            return []
        
        texts_to_translate = [entry.msgid for entry in self.entries]
        
        # Perform batch translation
        try:
            translated_texts = translation_service.translate(
                text=texts_to_translate,
                source_language=self.source_language,
                target_language=self.target_language
            )
            
            # Update entries with translations
            for entry, translation in zip(self.entries, translated_texts):
                entry.msgstr = translation
                
            return translated_texts
            
        except Exception as e:
            raise Exception(f"Translation failed: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the PO file.
        
        Returns:
            Dict[str, int]: Statistics including total entries, translated, and untranslated
        """
        if not self.po_file:
            return {"total": 0, "translated": 0, "untranslated": 0, "fuzzy": 0}
        
        return {
            "total": len(self.po_file),
            "translated": len(self.po_file.translated_entries()),
            "untranslated": len(self.po_file.untranslated_entries()),
            "fuzzy": len(self.po_file.fuzzy_entries())
        } 