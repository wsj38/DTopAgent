<img width="1508" height="660" alt="image" src="https://github.com/user-attachments/assets/c4da60d3-e164-4ff0-b902-c1af38af5b77" /># DTopAgent: A Multi-agent Framework for Dynamic Top-k Chunk Retrieval in RAG Pipeline

DTopAgent is a comprehensive framework for adaptive retrieval-augmented generation that dynamically adjusts chunk size based on question complexity and document quality. The system implements a four-stage pipeline: **Predictor** → **Answer** → **Decision** → **Evaluation**.

## 🏗️ System Architecture

```
SPARK/
├── predictor/     # Stage 1: Predict optimal chunk size (K)
├── answer/        # Stage 2: Generate answers using predicted K
├── decision/       # Stage 3: Evaluate and adjust K values
├── eval/          # Stage 4: Final evaluation and metrics
└── data/          # Datasets and training data
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch transformers requests tqdm

# Ensure you have access to:
# - Language model API (Ollama or similar)
# - GPU with CUDA support (recommended)
# - Sufficient disk space for datasets (~10GB)
```

### Complete Workflow

```bash
# 1. Predict optimal K values
cd predictor
python use_predictor.py --model_id "your-model" --input questions.json --output k_predictions.json

# 2. Generate answers (short or long)
cd ../answer
python short_generation.py --input k_predictions.json --output answers.json
# OR
python long_generation.py --input k_predictions.json --output answers.json

# 3. Evaluate and control
cd ../decision
python decision.py --meta_file k_predictions.json --answer_file answers.json --output evaluation.json
python process.py all --evaluation evaluation.json --original k_predictions.json --output-dir processed/

# 4. Final evaluation (run your evaluation scripts in eval/)
cd ../eval
# Add your evaluation metrics here
```

## 📋 Detailed Workflow

### Stage 1: Predictor (`predictor/`)

**Purpose**: Predict the optimal number of documents (K) needed to answer each question.

**Key Features**:
- Analyzes question complexity and document relevance
- Outputs K values: 0 (no context), 1-5 (increasing context), or null (unanswerable)
- Uses language model to make intelligent predictions

**Usage**:
```bash
python use_predictor.py \
    --model_id "Meta-Llama-3-8B-Instruct" \
    --input questions_with_docs.json \
    --output k_predictions.json
```

**Input Format**:
```json
[
  {
    "question": "What is the capital of France?",
    "meta": [
      {"content": "Paris is the capital and largest city of France..."},
      {"content": "France is a country in Western Europe..."}
    ],
    "ground_truth": ["Paris"]
  }
]
```

**Output Format**:
```json
[
  {
    "question": "What is the capital of France?",
    "meta": [...],
    "ground_truth": ["Paris"],
    "k": "1"
  }
]
```

### Stage 2: Answer Generation (`answer/`)

**Purpose**: Generate answers using the predicted K values and document context.

**Two Generation Modes**:

#### Short Generation (`short_generation.py`)
- Optimized for concise, direct answers
- Always requires document context
- Uses "as briefly as possible" instruction

```bash
python short_generation.py \
    --input k_predictions.json \
    --output short_answers.json \
    --max_tokens 64 \
    --temperature 0.3
```

#### Long Generation (`long_generation.py`)
- Generates detailed, comprehensive answers
- Supports both document-based and knowledge-based answering
- More flexible context handling

```bash
python long_generation.py \
    --input k_predictions.json \
    --output long_answers.json \
    --max_tokens 512 \
    --temperature 0.7
```

**Output Format**:
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

### Stage 3: Decision (`decision/`)

**Purpose**: Evaluate generated answers and dynamically adjust K values based on performance.

#### Evaluation (`decision.py`)
- Assesses answer quality on a 1-10 scale
- Provides context adjustment recommendations
- Connects to Ollama API for evaluation

```bash
python decision.py \
    --meta_file k_predictions.json \
    --answer_file answers.json \
    --output evaluation_results.json \
    --api_url "http://192.168.200.215:21004/v1/chat/completions" \
    --model_name "Qwen2.5-72B-Instruct-GPTQ-Int4"
```

#### Processing (`process.py`)
- Classifies results by evaluation scores
- Updates K values based on performance feedback
- Supports both individual steps and batch processing

```bash
# Run all processing steps
python process.py all \
    --evaluation evaluation_results.json \
    --original k_predictions.json \
    --output-dir processed_results/

# Or run individual steps
python process.py classify --input evaluation_results.json --output-high high.json --output-low low.json
python process.py reflect --input low.json --output updated.json
python process.py update --original k_predictions.json --k-values updated.json --output final.json
```

**Evaluation Output**:
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

### Stage 4: Evaluation (`eval/`)

**Purpose**: Final evaluation and metrics calculation.

**Note**: The eval directory is currently empty. Add your evaluation scripts here to:
- Calculate accuracy metrics
- Compare different K values
- Analyze performance across datasets
- Generate final reports

**Suggested Evaluation Scripts**:
```bash
# Example evaluation structure
eval/
├── calculate_metrics.py    # Calculate accuracy, F1, etc.
├── compare_k_values.py     # Compare performance across K values
├── dataset_analysis.py     # Analyze performance by dataset
└── generate_report.py      # Generate final evaluation report
```

## 📊 Data Management (`data/`)

The data directory contains datasets for training and evaluation:

### Supported Datasets
- **Natural Questions (NQ)**: Real user questions from Google
- **TriviaQA**: Trivia-style reading comprehension
- **SQuAD**: Wikipedia-based reading comprehension
- **HotpotQA**: Multi-hop reasoning questions
- **Bio Dataset**: Biomedical questions (from CRAG)

### Data Download
```bash
# Download all datasets
cd data
bash download_datasets.sh  # Create this script based on data/README.md

# Or download individually
mkdir -p nq && cd nq
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
gzip -d biencoder-nq-dev.json.gz
```

## ⚙️ Configuration

### Model Configuration
- **Predictor Model**: Meta-Llama-3-8B-Instruct (default)
- **Answer Generation**: Meta-Llama-3-8B-Instruct (default)
- **Evaluation Model**: Qwen2.5-72B-Instruct-GPTQ-Int4 (default)

### API Configuration
- **Ollama Endpoint**: `http://192.168.200.215:21004/v1/chat/completions`
- **Temperature**: 0.6-0.8 (adjustable)
- **Max Tokens**: 256-512 (adjustable)

### K Value Logic
- `k=0`: No context (knowledge-based answering)
- `k=1-5`: Increasing context size
- `k=null`: Question unanswerable
- Dynamic adjustment based on evaluation scores

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Errors**
   ```bash
   # Check API endpoint
   curl http://192.168.200.215:21004/v1/models
   
   # Verify model availability
   curl http://192.168.200.215:21004/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "Qwen2.5-72B-Instruct-GPTQ-Int4", "messages": [{"role": "user", "content": "test"}]}'
   ```

2. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or use CPU
   export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
   ```

3. **File Not Found Errors**
   ```bash
   # Verify file paths and permissions
   ls -la input_file.json
   chmod 644 input_file.json
   ```

### Performance Optimization

- Use GPU acceleration when available
- Adjust `max_tokens` based on answer length requirements
- Lower `temperature` for more deterministic results
- Batch process multiple questions for efficiency

## 📈 Expected Results

### Performance Metrics
- **Accuracy**: Question-answering accuracy
- **Efficiency**: Optimal K value prediction
- **Adaptability**: Dynamic context adjustment
- **Robustness**: Performance across different question types

### Typical K Value Distribution
- Simple factual questions: K=1-2
- Complex reasoning questions: K=3-5
- Knowledge-based questions: K=0
- Unanswerable questions: K=null

## 🤝 Contributing

1. Follow the four-stage pipeline structure
2. Add comprehensive error handling
3. Include detailed logging
4. Update documentation for new features
5. Test with multiple datasets

## 📄 License

This project is part of the DTopAgent system. Please refer to the main project license for usage terms.

## 🔗 References

- [AdaComp Project](https://anonymous.4open.science/r/AdaComp-8C0C/)
- [CRAG Repository](https://github.com/HuskyInSalt/CRAG)
- [Self-RAG Project](https://selfrag.github.io/)
- [DPR Repository](https://github.com/facebookresearch/DPR)
