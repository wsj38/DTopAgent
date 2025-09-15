
#!/usr/bin/env python3
"""
Exact Match (EM) Calculator for AdaComp SPARK

This script calculates Exact Match scores for question-answering evaluation.
It compares predicted answers with ground truth answers using normalized text matching.

Author: AdaComp Team
Date: 2024
"""

import argparse
import json
import re
import string
from typing import List, Dict, Any, Union


def normalize_answer(s: Union[str, None]) -> str:
    """
    Normalize answer text for comparison.
    
    Args:
        s: Input text to normalize
        
    Returns:
        Normalized text string
    """
    # Check if None or NaN or non-string, convert to empty string if so
    if not isinstance(s, str):
        s = str(s) if s is not None else ''
    
    def remove_articles(text: str) -> str:
        """Remove articles (a, an, the) from text."""
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text: str) -> str:
        """Fix whitespace by joining split words."""
        return ' '.join(text.split())
    
    def remove_punc(text: str) -> str:
        """Remove punctuation from text."""
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(predictions: List[str], references: List[str]) -> int:
    """
    Compute exact match score: prediction matches any reference.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        1 if any prediction matches any reference, 0 otherwise
    """
    predictions = [normalize_answer(p) for p in predictions]
    references = [normalize_answer(r) for r in references]
    
    for pred in predictions:
        if pred in references:
            return 1
    return 0


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


def calculate_em_scores(data: List[Dict[str, Any]]) -> List[int]:
    """
    Calculate EM scores for all items in the dataset.
    
    Args:
        data: List of data items with 'answer' and 'ground_truth' fields
        
    Returns:
        List of EM scores (0 or 1 for each item)
    """
    em_scores = []
    
    for item in data:
        raw_prediction = item.get("answer", "").strip()
        predictions = [p.strip() for p in re.split(r"\band\b|\bor\b", raw_prediction) if p.strip()]
        references = item.get("ground_truth", [])
        
        score = compute_exact_match(predictions, references)
        em_scores.append(score)
    
    return em_scores


def main(input_file: str, output_file: str = None) -> None:
    """
    Main function to calculate and display EM scores.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save results
    """
    print(f"Loading data from: {input_file}")
    data = load_data(input_file)
    
    print(f"Calculating EM scores for {len(data)} samples...")
    em_scores = calculate_em_scores(data)
    
    # Calculate average EM
    average_em = sum(em_scores) / len(em_scores) * 100
    
    print(f"\n🔎 Average EM over {len(em_scores)} samples: {average_em:.2f}%")
    
    # Save results if output file specified
    if output_file:
        results = {
            "average_em": average_em,
            "total_samples": len(em_scores),
            "em_scores": em_scores,
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
        description="Calculate Exact Match scores for question-answering evaluation"
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
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args.input, args.output)
