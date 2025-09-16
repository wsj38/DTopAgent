# Long Generation Script

This script generates long-form answers using a language model pipeline for the DTopAgent project.

## Features

- Processes questions with optional document context
- Generates detailed responses using Meta-Llama-3-8B-Instruct model
- Supports both document-based and knowledge-based answering
- Configurable generation parameters
- Comprehensive error handling and logging

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- CUDA-compatible GPU (recommended)

## Installation

```bash
pip install torch transformers
```

## Usage

### Basic Usage

```bash
python long_generation.py --input input.json --output output.json
```

### Advanced Usage

```bash
python long_generation.py \
    --model_path /path/to/model \
    --input input.json \
    --output output.json \
    --max_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9
```

### Command Line Arguments

- `--model_path`: Path to the language model (default: Meta-Llama-3-8B-Instruct)
- `--input`: Path to input JSON file containing questions and metadata (required)
- `--output`: Path to output JSON file for generated answers (required)
- `--max_tokens`: Maximum number of tokens to generate (default: 256)
- `--temperature`: Sampling temperature for generation (default: 0.6)
- `--top_p`: Top-p sampling parameter (default: 0.9)

## Input Format

The input JSON file should contain an array of objects with the following structure:

```json
[
  {
    "question": "What is the capital of France?",
    "meta": [
      {
        "content": "Paris is the capital and largest city of France..."
      }
    ],
    "ground_truth": ["Paris"]
  }
]
```

### Fields

- `question`: The question to be answered (required)
- `meta`: List of document metadata objects with `content` field (optional)
- `ground_truth`: Expected answer(s) for evaluation (optional)

## Output Format

The output JSON file contains an array of results:

```json
[
  {
    "question": "What is the capital of France?",
    "ground_truth": ["Paris"],
    "answer": "Paris is the capital of France...",
    "meta": [...]
  }
]
```

## Error Handling

The script includes comprehensive error handling:

- Validates input file existence
- Creates output directories if needed
- Handles JSON parsing errors gracefully
- Continues processing even if individual items fail
- Logs errors for debugging

## Examples

### Document-based Question Answering

```bash
python long_generation.py \
    --input questions_with_docs.json \
    --output answers_with_docs.json
```

### Knowledge-based Question Answering

```bash
python long_generation.py \
    --input questions_no_docs.json \
    --output answers_no_docs.json
```

## License

This script is part of the DTopAgent project.
