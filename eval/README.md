# Evaluation Module for AdaComp SPARK

This module provides comprehensive evaluation tools for the AdaComp SPARK project. It includes various metrics for assessing question-answering performance, text quality, and data analysis.

## Overview

The evaluation module offers five main evaluation scripts:
1. **Exact Match (EM)** - Binary matching evaluation
2. **F1 Score** - Unigram F1 score calculation
3. **ROUGE & BERTScore** - Text generation quality metrics
4. **Token Counter** - Token analysis and statistics
5. **Content Counter** - Data structure analysis

## Files

### `cal_EM.py`
Calculates Exact Match scores for question-answering evaluation.

**Features:**
- Normalizes text for fair comparison
- Handles multiple predictions per question
- Supports "and"/"or" separated answers
- Binary scoring (0 or 1)

### `F1.py`
Calculates F1 scores using unigram token matching.

**Features:**
- Supports F1, precision, and recall metrics
- Text normalization with article removal
- Handles multiple ground truth answers
- Returns maximum score across references

### `eval_rouge.py`
Evaluates text generation quality using ROUGE and BERTScore metrics.

**Features:**
- ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum
- BERTScore precision, recall, and F1
- Optional newline removal
- Multi-language support

### `eval_token.py`
Counts tokens in JSON data using tiktoken encoding.

**Features:**
- GPT-3.5/GPT-4 compatible tokenization
- Supports standard and CRAG data formats
- Text normalization and preprocessing
- Comprehensive token statistics

### `eval_number.py`
Analyzes content distribution in JSON data files.

**Features:**
- Counts content fields in metadata
- Distribution analysis
- Statistical summaries
- Detailed reporting options

## Installation

### Requirements
- Python 3.7+
- tiktoken
- evaluate
- transformers (for BERTScore)

### Setup
```bash
pip install tiktoken evaluate transformers
```

## Usage

### Exact Match Evaluation
```bash
python cal_EM.py --input answers.json --output em_results.json
```

### F1 Score Calculation
```bash
python F1.py --input answers.json --output f1_results.json --metric f1
```

### ROUGE and BERTScore Evaluation
```bash
python eval_rouge.py --input answers.json --output rouge_results.json --lang en
```

### Token Counting
```bash
# Standard format (items with meta fields)
python eval_token.py --input data.json --output token_stats.json

# CRAG format (list of strings)
python eval_token.py --input data.json --output token_stats.json --mode crag
```

### Content Analysis
```bash
python eval_number.py --input data.json --output content_stats.json --detailed
```

## Input/Output Formats

### Input Format
All scripts expect JSON files with the following structure:

```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "ground_truth": ["Paris"],
    "meta": [
      {
        "content": "Paris is the capital of France..."
      }
    ]
  }
]
```

### Output Formats

#### EM Results
```json
{
  "average_em": 85.5,
  "total_samples": 100,
  "em_scores": [1, 0, 1, ...],
  "input_file": "answers.json"
}
```

#### F1 Results
```json
{
  "metric": "f1",
  "average_score": 78.3,
  "total_samples": 100,
  "scores": [0.8, 0.6, 0.9, ...],
  "input_file": "answers.json"
}
```

#### ROUGE Results
```json
{
  "input_file": "answers.json",
  "total_samples": 100,
  "remove_newlines": false,
  "language": "en",
  "metrics": {
    "rouge1": 0.45,
    "rouge2": 0.32,
    "rougeL": 0.41,
    "rougeLsum": 0.42,
    "bertscore_precision": 0.78,
    "bertscore_recall": 0.76,
    "bertscore_f1": 0.77,
    "bertscore_avg": 0.77
  }
}
```

#### Token Statistics
```json
{
  "input_file": "data.json",
  "mode": "standard",
  "encoding": "cl100k_base",
  "statistics": {
    "total_items": 100,
    "total_tokens": 15000,
    "average_tokens": 150.0,
    "min_tokens": 50,
    "max_tokens": 300,
    "token_counts": [150, 200, 100, ...]
  }
}
```

## Command Line Options

### Common Options
All scripts support these common options:
- `--input`: Path to input JSON file (required)
- `--output`: Path to save results (optional)

### Specific Options

#### cal_EM.py
- No additional options

#### F1.py
- `--metric`: Choose metric (f1, precision, recall)

#### eval_rouge.py
- `--remove-newlines`: Remove newlines before evaluation
- `--lang`: Language code for BERTScore (default: en)

#### eval_token.py
- `--mode`: Processing mode (standard, crag)
- `--encoding`: tiktoken encoding name (default: cl100k_base)

#### eval_number.py
- `--detailed`: Show detailed analysis

## Examples

### Complete Evaluation Pipeline
```bash
# 1. Calculate Exact Match
python cal_EM.py --input answers.json --output em_results.json

# 2. Calculate F1 Score
python F1.py --input answers.json --output f1_results.json

# 3. Calculate ROUGE and BERTScore
python eval_rouge.py --input answers.json --output rouge_results.json

# 4. Analyze token usage
python eval_token.py --input data.json --output token_stats.json

# 5. Count content fields
python eval_number.py --input data.json --output content_stats.json --detailed
```

### Batch Evaluation
```bash
# Evaluate multiple files
for file in *.json; do
    python cal_EM.py --input "$file" --output "${file%.json}_em.json"
    python F1.py --input "$file" --output "${file%.json}_f1.json"
done
```

## Configuration

### Token Encoding
The token counter uses tiktoken with cl100k_base encoding by default, which is compatible with GPT-3.5 and GPT-4 models.

### Text Normalization
All evaluation scripts normalize text by:
- Converting to lowercase
- Removing articles (a, an, the)
- Removing punctuation
- Fixing whitespace

### Language Support
BERTScore supports multiple languages. Use the `--lang` parameter to specify:
- `en`: English (default)
- `zh`: Chinese
- `de`: German
- `fr`: French
- And many more...

## Performance Tips

1. **Large Datasets**: For very large datasets, consider processing in batches
2. **Memory Usage**: BERTScore can be memory-intensive for large texts
3. **Token Counting**: Use appropriate encoding for your model
4. **Parallel Processing**: Run different metrics in parallel for faster evaluation

## Integration

The evaluation module integrates with the broader AdaComp SPARK system:

1. **Input**: Receives generated answers from answer generation modules
2. **Processing**: Applies various evaluation metrics
3. **Output**: Provides detailed evaluation results for analysis

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all required packages are installed
   - Check Python version compatibility

2. **File Format Errors**
   - Verify JSON file structure
   - Check for required fields (answer, ground_truth)

3. **Memory Issues**
   - Reduce batch size for large datasets
   - Use appropriate token encoding

4. **Encoding Issues**
   - Ensure files are UTF-8 encoded
   - Check for special characters

### Debug Mode
Add verbose logging by modifying scripts to include more detailed output for troubleshooting.

## License

This module is part of the AdaComp SPARK project.
