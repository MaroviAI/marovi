import polib
from pathlib import Path

from marovi.modules.parsing.po_parser import POParser

class DummyTranslator:
    def translate(self, text, source_language="en", target_language="es"):
        if isinstance(text, list):
            return [f"{target_language}-{t}" for t in text]
        return f"{target_language}-{text}"

def create_sample_po(path: Path) -> None:
    content = '\n'.join([
        'msgid ""',
        'msgstr ""',
        '"Content-Type: text/plain; charset=UTF-8\\n"',
        '',
        'msgid "Hello"',
        'msgstr ""',
        '',
        'msgid "World"',
        'msgstr ""',
    ])
    path.write_text(content, encoding="utf-8")


def test_po_parser_auto_translate_and_save(tmp_path):
    po_file = tmp_path / "test.po"
    create_sample_po(po_file)

    parser = POParser()
    parser.load_from_file(str(po_file))
    parser.set_languages("en", "es")

    translations = parser.auto_translate(DummyTranslator())
    assert translations == ["es-Hello", "es-World"]

    out_file = tmp_path / "out.po"
    parser.save_translated_po(str(out_file), translations)

    check = polib.pofile(str(out_file))
    assert check[1].msgstr == "es-Hello"
    assert check[2].msgstr == "es-World"

    compiled = parser.compile_translations()
    assert "es-Hello" in compiled and "es-World" in compiled
