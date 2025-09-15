# Data Directory for AdaComp SPARK

This directory contains the raw datasets used for training and evaluation in the AdaComp SPARK project. The datasets are primarily sourced from DPR (Dense Passage Retrieval) and other question-answering benchmarks.

## Dataset Overview

The data directory is organized into the following subdirectories:
- `nq/` - Natural Questions dataset
- `trivia/` - TriviaQA dataset  
- `squad/` - SQuAD dataset

## Data Download Instructions

### Download Natural Questions (NQ)

```bash
# Create directory for Natural Questions
$ mkdir -p raw_data/nq
$ cd raw_data/nq

# Download development set
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
$ gzip -d biencoder-nq-dev.json.gz

# Download training set
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
$ gzip -d biencoder-nq-train.json.gz
```

### Download TriviaQA

```bash
# Navigate to parent directory
$ cd ..

# Create directory for TriviaQA
$ mkdir -p trivia
$ cd trivia

# Download development set
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz
$ gzip -d biencoder-trivia-dev.json.gz

# Download training set
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz
$ gzip -d biencoder-trivia-train.json.gz
```

### Download SQuAD

```bash
# Navigate to parent directory
$ cd ..

# Create directory for SQuAD
$ mkdir -p squad
$ cd squad

# Download development set
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz
$ gzip -d biencoder-squad1-dev.json.gz

# Download training set
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz
$ gzip -d biencoder-squad1-train.json.gz
```

### Download Bio Dataset (from CRAG)

```bash
# Navigate to parent directory
$ cd ..

# Clone CRAG repository for Bio dataset
$ git clone https://github.com/HuskyInSalt/CRAG.git
$ cd CRAG

# Download evaluation data created by Self-RAG on Bio dataset
# Note: You need to follow the CRAG repository instructions for downloading eval_data
$ mkdir -p eval_data
$ cd eval_data

# Download Bio dataset evaluation files
# (Specific download commands depend on CRAG repository setup)
# Please refer to CRAG documentation for exact download instructions
```

### Download HotpotQA Dataset (from DPR)

```bash
# Navigate to parent directory
$ cd ../..

# Clone DPR repository for HotpotQA dataset
$ git clone https://github.com/facebookresearch/DPR.git
$ cd DPR

# Use DPR's download script to get HotpotQA data
$ python dpr/data/download_data.py \
  --resource data.wikipedia_split.psgs_w100 \
  --output_dir ./data

# Download HotpotQA questions and passages
$ python dpr/data/download_data.py \
  --resource data.retriever.qas.hotpotqa-train \
  --output_dir ./data

$ python dpr/data/download_data.py \
  --resource data.retriever.qas.hotpotqa-dev \
  --output_dir ./data

# Alternatively, download directly from DPR's data URLs
$ mkdir -p hotpot
$ cd hotpot

# Download HotpotQA training data
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-hotpot-train.json.gz
$ gzip -d biencoder-hotpot-train.json.gz

# Download HotpotQA development data
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-hotpot-dev.json.gz
$ gzip -d biencoder-hotpot-dev.json.gz
```

## Dataset Descriptions

### Natural Questions (NQ)
- **Source**: Google AI Language
- **Purpose**: Question-answering evaluation with real user questions
- **Format**: JSON files with questions, passages, and answers
- **Size**: ~300k questions in training set, ~8k in development set

### TriviaQA
- **Source**: University of Washington
- **Purpose**: Reading comprehension with trivia-style questions
- **Format**: JSON files with questions, supporting documents, and answers
- **Size**: ~95k question-answer pairs in training set, ~11k in development set

### SQuAD
- **Source**: Stanford University
- **Purpose**: Reading comprehension from Wikipedia articles
- **Format**: JSON files with questions, passages, and answers
- **Size**: ~100k questions in training set, ~10k in development set

### Bio Dataset
- **Source**: CRAG project (Corrective Retrieval Augmented Generation)
- **Purpose**: Biomedical question-answering evaluation
- **Format**: JSON files with biomedical questions and factual answers
- **Size**: Biomedical domain questions for evaluation
- **Evaluation**: Uses FactScore for factual accuracy assessment

### HotpotQA Dataset
- **Source**: Facebook Research DPR
- **Purpose**: Multi-hop reasoning question answering
- **Format**: JSON files with complex questions requiring reasoning across multiple passages
- **Size**: ~90k questions in training set, ~7k in development set
- **Features**: Requires reasoning over multiple Wikipedia passages

## Data Format

All datasets follow the DPR format with the following structure:

```json
[
  {
    "question": "What is the capital of France?",
    "answers": ["Paris"],
    "positive_ctxs": [
      {
        "title": "France",
        "text": "Paris is the capital and largest city of France..."
      }
    ],
    "negative_ctxs": [
      {
        "title": "Germany",
        "text": "Berlin is the capital of Germany..."
      }
    ]
  }
]
```

## Usage in AdaComp SPARK

These datasets are used for:
1. **Training**: Fine-tuning DPR models for passage retrieval
2. **Evaluation**: Assessing retrieval and generation performance
3. **Analysis**: Studying question-answering patterns and performance

## File Organization

```
data/
├── nq/
│   ├── biencoder-nq-dev.json
│   └── biencoder-nq-train.json
├── trivia/
│   ├── biencoder-trivia-dev.json
│   └── biencoder-trivia-train.json
├── squad/
│   ├── biencoder-squad1-dev.json
│   └── biencoder-squad1-train.json
├── hotpot/
│   ├── biencoder-hotpot-dev.json
│   └── biencoder-hotpot-train.json
├── bio/
│   └── (Bio evaluation files from CRAG)
└── README.md
```

## Notes

- All datasets are in JSON format after decompression
- Files are sourced from Facebook AI Research's DPR repository
- Ensure sufficient disk space (~10GB total for all datasets)
- Download times may vary depending on network speed

## License and Attribution

Please refer to the original dataset licenses:
- [Natural Questions](https://ai.google.com/research/NaturalQuestions/)
- [TriviaQA](https://www.cs.washington.edu/triviaqa/)
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [HotpotQA](https://hotpotqa.github.io/) - Multi-hop reasoning dataset
- [CRAG](https://github.com/HuskyInSalt/CRAG) - Corrective Retrieval Augmented Generation

For DPR-specific formatting and download scripts, see: [DPR Repository](https://github.com/facebookresearch/DPR)

## Bio Dataset Evaluation

The Bio dataset from CRAG uses FactScore for evaluation. To run evaluation:

```bash
python -m factscore.factscorer \
  --data_path YOUR_OUTPUT_FILE \
  --model_name retrieval+ChatGPT \
  --cache_dir YOUR_CACHE_DIR \
  --openai_key YOUR_OPEN_AI_KEY \
  --verbose
```

**Note**: FactScore previously used `text-davinci-003` by default, which has been deprecated since 2024-01-04 and replaced by `gpt-3.5-turbo-instruct`. Results may differ between these models.
