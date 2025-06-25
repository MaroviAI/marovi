# Marovi Pipeline Framework

The Marovi Pipeline Framework provides a flexible, modular system for building document processing workflows. It's designed to support various tasks like translation, format conversion, and content summarization through a composable pipeline architecture.

## Key Features

- **Modular design**: Build pipelines from reusable, swappable components
- **Type-safe processing**: Generic typing ensures data consistency
- **Batch processing**: Efficient handling of large datasets
- **Checkpointing**: Save and resume pipeline processing
- **CLI integration**: Run pipelines directly from the command line
- **Extensibility**: Easy to add new pipeline implementations
- **Metrics and observability**: Track performance and execution statistics

## Available Pipelines

- **PO Translation Pipeline**: Translate gettext PO files from one language to another

## Usage

### Command Line Interface

```bash
# List available pipelines
python -m marovi.pipelines.cli list

# Get information about a pipeline
python -m marovi.pipelines.cli info po_translation

# Run a pipeline
python -m marovi.pipelines.cli run po_translation --input data/input_files --target_lang es
```

### Programmatic Usage

```python
# Option 1: Direct Pipeline Usage
from marovi.pipelines.po_translation import POTranslationPipeline

# Create and configure the pipeline
pipeline = POTranslationPipeline()
result = pipeline.run(
    inputs="data/input_files",
    source_lang="en",
    target_lang="es",
    output_dir="data/output_files"
)

# Option 2: Convenience Function
from marovi.pipelines.po_translation import translate_po_files

result = translate_po_files(
    po_files="data/input_files",
    target_lang="es",
    output_dir="data/output_files"
)

# Option 3: PipelineRunner
from marovi.pipelines.utils.runner import PipelineRunner

runner = PipelineRunner()
result = runner.run_pipeline(
    pipeline_name="po_translation",
    inputs="data/input_files",
    target_lang="es"
)
```

## Directory Structure

- **pipelines/**: Root directory for pipeline implementations
  - **core.py**: Core pipeline framework components
  - **context.py**: Pipeline context for state management
  - **po_translation.py**: PO file translation pipeline
  - **cli.py**: Command-line interface entry point
  - **utils/**: Utility modules
    - **base.py**: Base pipeline class definition
    - **runner.py**: Dynamic pipeline discovery and execution
    - **cli.py**: CLI implementation
  - **runs/**: Example scripts for running pipelines
    - **po_translation.py**: Example script for PO translation
    - **test_po_translation.py**: Test script with sample data

## Creating New Pipeline Implementations

To create a new pipeline implementation:

1. Create a new module in the `pipelines` directory
2. Define a pipeline class that inherits from `BasePipeline`
3. Implement the required methods: `run()`, `get_cli_arguments()`
4. Add your pipeline to the `__init__.py` exports

See the `po_translation.py` module for a complete example.

## Documentation

For more information about the Marovi Pipeline Framework, see the tutorials:

- [Pipeline Framework Overview](../tutorials/pipelines.ipynb)
- [MaroviAPI Usage](../tutorials/api.ipynb) 