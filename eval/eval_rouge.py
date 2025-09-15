#!/usr/bin/env python3
"""
ROUGE and BERTScore Evaluator for AdaComp SPARK

This script evaluates text generation quality using ROUGE metrics and BERTScore.
It compares predicted answers with ground truth answers using multiple evaluation metrics.

Author: AdaComp Team
Date: 2024
"""

import argparse
import json
from typing import List, Dict, Any, Optional
import evaluate


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


def extract_predictions_and_references(data: List[Dict[str, Any]], remove_newlines: bool = False) -> tuple[List[str], List[str]]:
    """
    Extract predictions and references from data.
    
    Args:
        data: List of data items
        remove_newlines: Whether to remove newlines from text
        
    Returns:
        Tuple of (predictions, references)
    """
    pred_list = []
    gt_list = []
    
    for item in data:
        prediction = item.get('answer', '')
        ground_truth = item.get('ground_truth', '')
        
        # Handle different ground truth formats
        if isinstance(ground_truth, list):
            # Take the first ground truth if multiple exist
            ground_truth = ground_truth[0] if ground_truth else ''
        
        if remove_newlines:
            prediction = prediction.replace("\n", " ")
            ground_truth = ground_truth.replace("\n", " ")
        
        pred_list.append(prediction)
        gt_list.append(ground_truth)
    
    return pred_list, gt_list


def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE scores.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary of ROUGE scores
    """
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=references)
    
    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "rougeLsum": rouge_results["rougeLsum"]
    }


def calculate_bertscore(predictions: List[str], references: List[str], lang: str = "en") -> Dict[str, float]:
    """
    Calculate BERTScore.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        lang: Language code for BERTScore
        
    Returns:
        Dictionary of BERTScore metrics
    """
    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(predictions=predictions, references=references, lang=lang)
    
    p = bertscore_results["precision"]
    r = bertscore_results["recall"]
    f1 = bertscore_results["f1"]
    
    # Calculate average scores
    avg_precision = sum(p) / len(p)
    avg_recall = sum(r) / len(r)
    avg_f1 = sum(f1) / len(f1)
    
    return {
        "bertscore_precision": avg_precision,
        "bertscore_recall": avg_recall,
        "bertscore_f1": avg_f1,
        "bertscore_avg": (avg_precision + avg_recall + avg_f1) / 3
    }


def evaluate_text_quality(data: List[Dict[str, Any]], remove_newlines: bool = False, lang: str = "en") -> Dict[str, float]:
    """
    Evaluate text quality using ROUGE and BERTScore.
    
    Args:
        data: List of data items
        remove_newlines: Whether to remove newlines from text
        lang: Language code for BERTScore
        
    Returns:
        Dictionary of evaluation metrics
    """
    predictions, references = extract_predictions_and_references(data, remove_newlines)
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(predictions, references)
    
    # Calculate BERTScore
    bertscore_scores = calculate_bertscore(predictions, references, lang)
    
    # Combine all metrics
    metrics = {**rouge_scores, **bertscore_scores}
    
    return metrics


def main(input_file: str, output_file: str = None, remove_newlines: bool = False, lang: str = "en") -> None:
    """
    Main function to evaluate text quality.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save results
        remove_newlines: Whether to remove newlines from text
        lang: Language code for BERTScore
    """
    print(f"Loading data from: {input_file}")
    data = load_data(input_file)
    
    print(f"Evaluating {len(data)} samples...")
    metrics = evaluate_text_quality(data, remove_newlines, lang)
    
    print("\n最终评估结果:")
    for metric_name, score in metrics.items():
        print(f"{metric_name}: {score * 100:.2f}")
    
    # Save results if output file specified
    if output_file:
        results = {
            "input_file": input_file,
            "total_samples": len(data),
            "remove_newlines": remove_newlines,
            "language": lang,
            "metrics": metrics
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
        description="Evaluate text quality using ROUGE and BERTScore metrics"
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
        '--remove-newlines',
        action='store_true',
        help='Remove newlines from text before evaluation'
    )
    parser.add_argument(
        '--lang',
        type=str,
        default='en',
        help='Language code for BERTScore (default: en)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args.input, args.output, args.remove_newlines, args.lang)
