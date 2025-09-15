#!/usr/bin/env python3
"""
F1 Score Calculator for AdaComp SPARK

This script calculates F1 scores for question-answering evaluation.
It computes unigram F1 scores between predicted answers and ground truth answers.

Author: AdaComp Team
Date: 2024
"""

import argparse
import json
import re
import string
from typing import List, Dict, Any, Union
from collections import Counter


def normalize_text(text: str) -> str:
    """
    Normalize text with lowercasing, removing articles, and punctuation.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    def remove_articles(text: str) -> str:
        """Remove articles (a, an, the) from text."""
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        """Fix whitespace by joining split words."""
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        """Remove punctuation from text."""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def calc_unigram_f1(text: str, answers: List[str], field: str = "f1") -> float:
    """
    Calculate unigram F1 score between the text and reference answers.
    
    Args:
        text: Predicted text
        answers: List of reference answers
        field: Metric to return ("f1", "precision", "recall")
        
    Returns:
        F1 score, precision, or recall
    """
    norm_pred = normalize_text(text)
    norm_answers = [normalize_text(ans) for ans in answers]
    
    # Calculate common tokens with each reference answer
    common_tokens = [
        Counter(norm_pred) & Counter(norm_ans) for norm_ans in norm_answers
    ]
    num_same = [sum(common.values()) for common in common_tokens]

    score_list = []
    for i, num in enumerate(num_same):
        if num == 0:
            score_list.append(0.0)
        else:
            p = 1.0 * num / len(norm_pred)
            r = 1.0 * num / len(norm_answers[i])
            f1 = 2 * p * r / (p + r)
            
            if field == "precision":
                score_list.append(p)
            elif field == "recall":
                score_list.append(r)
            elif field == "f1":
                score_list.append(f1)
            else:
                raise ValueError(f"Unknown field: {field}")
    
    return max(score_list)


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of data items
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")


def calculate_f1_scores(data: List[Dict[str, Any]], metric: str = "f1") -> List[float]:
    """
    Calculate F1 scores for all items in the dataset.
    
    Args:
        data: List of data items with 'answer' and 'ground_truth' fields
        metric: Metric to calculate ("f1", "precision", "recall")
        
    Returns:
        List of F1 scores for each item
    """
    scores = []
    
    for item in data:
        text = item.get("answer", "").strip()
        answers = item.get("ground_truth", [])
        
        # Handle different ground truth formats
        if isinstance(answers, str):
            answers = [answers]
        
        score = calc_unigram_f1(text, answers, field=metric)
        scores.append(score)
    
    return scores


def main(input_file: str, output_file: str = None, metric: str = "f1") -> None:
    """
    Main function to calculate and display F1 scores.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save results
        metric: Metric to calculate ("f1", "precision", "recall")
    """
    print(f"Loading data from: {input_file}")
    data = load_data(input_file)
    
    print(f"Calculating {metric.upper()} scores for {len(data)} samples...")
    scores = calculate_f1_scores(data, metric)
    
    # Calculate average score
    average_score = sum(scores) / len(scores) * 100
    
    print(f"\n🔎 Average {metric.upper()} over {len(scores)} samples: {average_score:.2f}%")
    
    # Save results if output file specified
    if output_file:
        results = {
            "metric": metric,
            "average_score": average_score,
            "total_samples": len(scores),
            "scores": scores,
            "input_file": input_file
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_file}")


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Calculate F1 scores for question-answering evaluation"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSON file containing answers and ground truth'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save results (optional)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['f1', 'precision', 'recall'],
        default='f1',
        help='Metric to calculate (default: f1)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args.input, args.output, args.metric)
