"""Tests for the PO translation workflow.

This test includes a minimal stub for the ``polib`` package so that it can
run in environments where the real dependency is unavailable.  The stub
implements just enough functionality for ``POParser`` to operate.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Provide a minimal ``polib`` stub if the real library is unavailable
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only when polib is available
    import polib as polib_stub  # type: ignore
except Exception:  # pragma: no cover - executed when polib is missing
    polib_stub = types.ModuleType("polib")

    class POEntry:
        def __init__(self, msgid: str, msgstr: str = ""):
            self.msgid = msgid
            self.msgstr = msgstr
            self.obsolete = False

    class POFile(list):
        def untranslated_entries(self):
            return [e for e in self if not e.msgstr]

        def translated_entries(self):
            return [e for e in self if e.msgstr]

        def fuzzy_entries(self):
            return []

        def save(self, path: str):
            Path(path).write_text("")

    def _parse_content(content: str) -> POFile:
        entries = []
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        i = 0
        while i < len(lines):
            if lines[i].startswith("msgid"):
                msgid = lines[i][6:].strip().strip('"')
                if msgid == "":  # skip header entry
                    i += 2
                    continue
                if i + 1 < len(lines) and lines[i + 1].startswith("msgstr"):
                    msgstr = lines[i + 1][7:].strip().strip('"')
                    entries.append(POEntry(msgid, msgstr))
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return POFile(entries)

    def pofile(path_or_content: str) -> POFile:
        if Path(path_or_content).exists():
            content = Path(path_or_content).read_text()
        else:
            content = path_or_content
        return _parse_content(content)

    polib_stub.POEntry = POEntry
    polib_stub.POFile = POFile
    polib_stub.pofile = pofile
    sys.modules.setdefault("polib", polib_stub)


# ---------------------------------------------------------------------------
# Import the parser under test
# ---------------------------------------------------------------------------

from marovi.modules.parsing.po_parser import POParser


class _DummyService:
    """Minimal translation service used for testing.

    It simply appends the target language code to each piece of text.
    """

    def translate(self, text, source_language, target_language):
        return [f"{t}_{target_language}" for t in text]


def test_po_translation_workflow(tmp_path):
    """Ensure that POParser can load, translate, and compile a PO file."""

    po_content = (
        'msgid ""\n'
        'msgstr ""\n\n'
        'msgid "Hello"\n'
        'msgstr ""\n\n'
        'msgid "World"\n'
        'msgstr ""\n'
    )

    po_file = tmp_path / "messages.po"
    po_file.write_text(po_content)

    parser = POParser()
    parser.load_from_file(str(po_file))
    parser.set_languages("en", "es")

    service = _DummyService()
    translations = parser.auto_translate(service)

    assert translations == ["Hello_es", "World_es"]
    assert parser.parse() == {"Hello": "Hello_es", "World": "World_es"}

    compiled = parser.compile_translations()
    assert "Hello_es" in compiled
    assert "World_es" in compiled

    stats = parser.get_stats()
    assert stats["translated"] == 2
    assert stats["untranslated"] == 0

