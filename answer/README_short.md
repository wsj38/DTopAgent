# Short Generation Script

This script generates short, concise answers using a language model pipeline for the AdaComp SPARK project.

## Features

- Generates brief, concise answers from document context
- Optimized for short-form question answering
- Uses Meta-Llama-3-8B-Instruct model
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
python short_generation.py --input input.json --output output.json
```

### Advanced Usage

```bash
python short_generation.py \
    --model_path /path/to/model \
    --input input.json \
    --output output.json \
    --max_tokens 128 \
    --temperature 0.5 \
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
- `meta`: List of document metadata objects with `content` field (required for short generation)
- `ground_truth`: Expected answer(s) for evaluation (optional)

## Output Format

The output JSON file contains an array of results:

```json
[
  {
    "question": "What is the capital of France?",
    "ground_truth": ["Paris"],
    "answer": "Paris",
    "meta": [...]
  }
]
```

## Key Differences from Long Generation

- **Prompt Style**: Uses "as briefly as possible" instruction
- **Answer Length**: Optimized for short, direct answers
- **Document Requirement**: Always requires document context (meta field)
- **Token Limit**: Typically uses fewer tokens for generation

## Error Handling

The script includes comprehensive error handling:

- Validates input file existence
- Creates output directories if needed
- Handles JSON parsing errors gracefully
- Continues processing even if individual items fail
- Logs errors for debugging

## Examples

### Short Answer Generation

```bash
python short_generation.py \
    --input questions_with_docs.json \
    --output short_answers.json \
    --max_tokens 64
```

### Factual Question Answering

```bash
python short_generation.py \
    --input factual_questions.json \
    --output factual_answers.json \
    --temperature 0.3
```

## Performance Tips

- Use lower `max_tokens` values (64-128) for shorter answers
- Lower `temperature` (0.3-0.5) for more deterministic, factual answers
- Ensure documents in `meta` field are relevant and concise

## License

This script is part of the AdaComp SPARK project.
