# Short Answer Generation (short_generation.py)

This module generates concise answers for a list of questions using an instruction-tuned text-generation model. If retrieved documents (`meta`) are provided, they are used as supporting context; otherwise, the model answers from its own knowledge.

## Features
- Uses an instruction-tuned model (e.g., Meta-Llama-3-8B-Instruct)
- Supports context-aware generation with `meta` documents
- Returns brief, direct answers only
- CLI with configurable generation parameters
- Structured logging and basic error handling

## Installation

```bash
pip install torch transformers tqdm
```

## Usage

```bash
python short_generation.py \
  --model-id /path/to/model \
  --input /path/to/input.json \
  --output /path/to/output.json \
  --max-new-tokens 256 \
  --temperature 0.6 \
  --top-p 0.9
```

### Arguments
- `--model-id` (required): Hugging Face model id or local model path
- `--input` (required): Path to input JSON file
- `--output` (required): Path to output JSON file
- `--max-new-tokens` (default: 256): Max tokens to generate
- `--temperature` (default: 0.6): Sampling temperature
- `--top-p` (default: 0.9): Nucleus sampling parameter

## Input Format

Array of objects. Each object may include `meta` documents used as context.

```json
[
  {
    "question": "What is the capital of France?",
    "meta": [
      { "content": "Paris is the capital and largest city of France." }
    ],
    "ground_truth": ["Paris"]
  }
]
```

## Output Format

Array of objects with concise answers.

```json
[
  {
    "question": "What is the capital of France?",
    "ground_truth": ["Paris"],
    "answer": "Paris"
  }
]
```

## Notes
- If `meta` is provided, it is used to guide generation; otherwise the model relies on its knowledge.
- The script is optimized for short, direct answers. For detailed responses, use `long_generation.py`.
- GPU acceleration is recommended for speed.
