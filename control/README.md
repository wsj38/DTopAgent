# Decision Module for DTopAgent

This module provides evaluation and processing capabilities for the DTopAgent project. It consists of two main scripts that work together to evaluate question-answering responses and update context parameters.

## Overview

The decision module implements a feedback loop system that:
1. **Evaluates** generated answers using a language model
2. **Processes** evaluation results to update context parameters (k values)
3. **Refines** the system based on performance feedback

## Files

### `decision.py`
The main evaluation script that assesses question-answering responses using a language model API.

**Key Features:**
- Connects to Ollama API for response evaluation
- Evaluates answers on a 1-10 scale
- Provides context adjustment recommendations
- Handles both document-based and knowledge-based questions

### `process.py`
The processing script that handles evaluation results and updates system parameters.

**Key Features:**
- Classifies results by evaluation scores
- Updates k values based on performance feedback
- Supports both individual step execution and batch processing
- Modular design for easy integration

## Workflow

The typical workflow follows this sequence:

```
1. Run decision.py → Generate evaluation results
2. Run process.py → Process results and update parameters
3. Iterate → Use updated parameters for next evaluation cycle
```

## Installation

### Requirements
- Python 3.7+
- requests
- tqdm
- Access to Ollama API endpoint

### Setup
```bash
pip install requests tqdm
```

## Usage

### Step 1: Evaluation with decision.py

```bash
python decision.py --meta_file metadata.json --answer_file answers.json --output results.json
```

**Parameters:**
- `--meta_file`: Path to metadata JSON file containing questions and context
- `--answer_file`: Path to answer JSON file containing generated responses
- `--output`: Path to save evaluation results (optional)
- `--api_url`: API endpoint URL (default: Ollama endpoint)
- `--model_name`: Model name to use (default: Qwen2.5-72B-Instruct-GPTQ-Int4)
- `--temperature`: Sampling temperature (default: 0.8)

### Step 2: Processing with process.py

#### Option A: Run All Steps
```bash
python process.py all --evaluation results.json --original original.json --output-dir output/
```

#### Option B: Run Individual Steps
```bash
# Step 1: Classify by score
python process.py classify --input results.json --output-high high.json --output-low low.json

# Step 2: Reflect and update k values
python process.py reflect --input low.json --output updated.json

# Step 3: Update original dataset
python process.py update --original original.json --k-values updated.json --output final.json
```

## Input/Output Formats

### Input Format (decision.py)
```json
[
  {
    "question": "What is the capital of France?",
    "meta": [
      {
        "content": "Paris is the capital and largest city of France..."
      }
    ],
    "k": "1"
  }
]
```

### Evaluation Output Format
```json
[
  {
    "question": "What is the capital of France?",
    "meta": [...],
    "answer": "Paris",
    "response": "Evaluation Score: 9\nContext Adjustment: 0",
    "k": "1"
  }
]
```

### Processing Output Format
The processing script generates multiple files:
- `score_9_or_10.json`: High-scoring items (9-10)
- `score_not_9_or_10.json`: Low-scoring items (<9)
- `output_adjusted.json`: Updated k values
- `updated_with_k.json`: Final dataset with updated parameters

## Configuration

### API Configuration
The decision script connects to an Ollama API endpoint. Default configuration:
- URL: `http://192.168.200.215:端口/v1/chat/completions`
- Model: `Qwen2.5-72B-Instruct-GPTQ-Int4`
- Temperature: `0.8`

### K Value Logic
The system uses k values to decision context size:
- `k=0`: No context (knowledge-based answering)
- `k=1-5`: Increasing context size
- Dynamic adjustment based on evaluation scores

## Error Handling

Both scripts include comprehensive error handling:
- File validation and existence checks
- JSON parsing error recovery
- API request failure handling
- Graceful degradation for missing data

## Examples

### Complete Evaluation Cycle
```bash
# 1. Evaluate responses
python decision.py \
    --meta_file questions_with_context.json \
    --answer_file generated_answers.json \
    --output evaluation_results.json

# 2. Process results and update parameters
python process.py all \
    --evaluation evaluation_results.json \
    --original original_dataset.json \
    --output-dir processed_results/
```

### Custom API Configuration
```bash
python decision.py \
    --meta_file data.json \
    --answer_file answers.json \
    --api_url http://your-api-endpoint:port/v1/chat/completions \
    --model_name your-model-name \
    --temperature 0.7
```

## Integration

The decision module is designed to integrate seamlessly with the broader AdaComp SPARK system:

1. **Input**: Receives questions and generated answers from the answer generation module
2. **Processing**: Evaluates and processes results using language model APIs
3. **Output**: Provides updated parameters for the next iteration cycle

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API endpoint URL is correct
   - Check network connectivity
   - Ensure API service is running

2. **File Not Found Errors**
   - Verify input file paths are correct
   - Check file permissions
   - Ensure files exist before running scripts

3. **JSON Parsing Errors**
   - Validate JSON file format
   - Check for encoding issues (use UTF-8)
   - Verify required fields are present

### Debug Mode
Add verbose logging by modifying the scripts to include more detailed output for troubleshooting.

## License

This module is part of the AdaComp SPARK project.
