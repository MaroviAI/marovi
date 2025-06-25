# Marovi Examples

This directory contains usage examples for the Marovi API and pipeline framework.

## PO Translation Example

The `po_translation_example.py` script demonstrates how to translate PO (gettext) files using both Google Translate and LLM-based translation directly through the Marovi API.

### Usage

```
python po_translation_example.py [options]
```

#### Options:

- `--input PATH`: Input PO file path (if not provided, creates a sample file)
- `--output DIR`: Output directory for translated files (default: same directory as input)
- `--source LANG`: Source language code (default: en)
- `--target LANG`: Target language code (default: es)
- `--translator TYPE`: Translator type to use (choices: google, llm; default: google)
- `--include-translated`: Include already translated entries (default: False)

### Examples

1. Translate a sample file from English to Spanish using Google Translate:
   ```
   python examples/po_translation_example.py
   ```

2. Translate a specific PO file from English to French using LLM-based translation:
   ```
   python examples/po_translation_example.py --input path/to/file.po --target fr --translator llm
   ```

3. Re-translate an already translated PO file:
   ```
   python examples/po_translation_example.py --input path/to/file.po --include-translated
   ```

4. Save output to a specific directory:
   ```
   python examples/po_translation_example.py --input path/to/file.po --output path/to/output/dir
   ```

## Features

- Supports both Google Translate and LLM-based translation
- Handles individual message translation with error recovery
- Works with both untranslated and already translated PO files
- Creates sample PO files for testing
- Detailed logging of the translation process 