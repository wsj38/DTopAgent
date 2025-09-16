#!/usr/bin/env python3
"""
Process Script for DTopAgent

This script processes evaluation results from the control.py script.
It performs three main operations:
1. Score Classification: Separates high-scoring (9-10) from low-scoring items
2. Reflection: Updates k values based on evaluation scores and context adjustments
3. K-value Update: Updates k values in the original dataset

"""

import argparse
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries from the JSON file
        
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


def save_json_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def classify_by_score(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Classify data into high-scoring (9-10) and low-scoring items.
    
    Args:
        data: List of evaluation results
        
    Returns:
        Tuple of (high_score_items, low_score_items)
    """
    score_9_or_10 = []
    score_not_9_or_10 = []
    
    for item in data:
        response = item.get("response", "")
        if "Evaluation Score: 9" in response or "Evaluation Score: 10" in response:
            score_9_or_10.append(item)
        else:
            score_not_9_or_10.append(item)
    
    return score_9_or_10, score_not_9_or_10


def extract_evaluation_score(response: str) -> int:
    """
    Extract evaluation score from response text.
    
    Args:
        response: The evaluation response text
        
    Returns:
        Extracted score (defaults to 0 if not found)
    """
    match = re.search(r"Evaluation Score:\s*(\d+)", response)
    return int(match.group(1)) if match else 0


def update_k_values(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Update k values based on evaluation scores and context adjustments.
    
    Args:
        data: List of evaluation results
        
    Returns:
        Updated data with modified k values
    """
    for item in data:
        k = item.get("k", "0")
        response = item.get("response", "")
        score = extract_evaluation_score(response)
        
        # Handle different k value scenarios
        if k == "0":
            if "Context Adjustment: 1" in response:
                item["k"] = str(int(k) + 1)
            elif "Context Adjustment: 0" in response:
                item["k"] = k
            # If -1, keep k as 0 (no change)
            
        elif k == "5" or k == "null":
            if score < 3:
                item["k"] = "0"
            elif "Context Adjustment: -1" in response and score > 2:
                item["k"] = str(5 - 1)
            elif "Context Adjustment: 1" in response and score > 2:
                item["k"] = k
            elif "Context Adjustment: 0" in response and score > 2:
                item["k"] = k
                
        elif k in ["1", "2", "3", "4"]:
            if "Context Adjustment: -1" in response:
                item["k"] = str(int(k) - 1)
            elif "Context Adjustment: 1" in response:
                item["k"] = str(int(k) + 1)
            elif "Context Adjustment: 0" in response:
                item["k"] = k
    
    return data


def update_k_values_from_mapping(
    original_data: List[Dict[str, Any]], 
    k_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Update k values in original data using a mapping from evaluation results.
    
    Args:
        original_data: Original dataset
        k_mapping: Mapping of questions to new k values
        
    Returns:
        Updated original data
    """
    updated_data = []
    
    for item in original_data:
        question = item.get("question")
        if question in k_mapping:
            item["k"] = k_mapping[question]
            updated_data.append(item)
    
    return updated_data


def step1_classify_scores(input_file: str, output_high: str, output_low: str) -> None:
    """
    Step 1: Classify evaluation results by score.
    
    Args:
        input_file: Path to evaluation results file
        output_high: Path to save high-scoring items
        output_low: Path to save low-scoring items
    """
    print("Step 1: Classifying evaluation results by score...")
    
    data = load_json_file(input_file)
    score_9_or_10, score_not_9_or_10 = classify_by_score(data)
    
    save_json_file(score_9_or_10, output_high)
    save_json_file(score_not_9_or_10, output_low)
    
    print(f"Processed {len(data)} items:")
    print(f"  - High scores (9-10): {len(score_9_or_10)}")
    print(f"  - Low scores (<9): {len(score_not_9_or_10)}")


def step2_reflect_and_update_k(input_file: str, output_file: str) -> None:
    """
    Step 2: Reflect on scores and update k values.
    
    Args:
        input_file: Path to low-scoring items file
        output_file: Path to save updated items
    """
    print("Step 2: Reflecting on scores and updating k values...")
    
    data = load_json_file(input_file)
    updated_data = update_k_values(data)
    
    save_json_file(updated_data, output_file)
    print(f"Updated k values for {len(updated_data)} items")


def step3_update_original_k_values(
    original_file: str, 
    k_values_file: str, 
    output_file: str
) -> None:
    """
    Step 3: Update k values in original dataset.
    
    Args:
        original_file: Path to original dataset
        k_values_file: Path to file with updated k values
        output_file: Path to save updated original dataset
    """
    print("Step 3: Updating k values in original dataset...")
    
    original_data = load_json_file(original_file)
    k_data = load_json_file(k_values_file)
    
    # Create question to k mapping
    question_to_k = {item["question"]: item.get("k") for item in k_data}
    
    updated_data = update_k_values_from_mapping(original_data, question_to_k)
    
    save_json_file(updated_data, output_file)
    print(f"Updated {len(updated_data)} records in original dataset")


def run_all_steps(
    evaluation_file: str,
    original_file: str,
    output_dir: str
) -> None:
    """
    Run all three processing steps in sequence.
    
    Args:
        evaluation_file: Path to evaluation results
        original_file: Path to original dataset
        output_dir: Directory to save all outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Classify scores
    high_scores_file = output_dir / "score_9_or_10.json"
    low_scores_file = output_dir / "score_not_9_or_10.json"
    step1_classify_scores(evaluation_file, str(high_scores_file), str(low_scores_file))
    
    # Step 2: Reflect and update k values
    updated_k_file = output_dir / "output_adjusted.json"
    step2_reflect_and_update_k(str(low_scores_file), str(updated_k_file))
    
    # Step 3: Update original dataset
    final_output_file = output_dir / "updated_with_k.json"
    step3_update_original_k_values(original_file, str(updated_k_file), str(final_output_file))
    
    print("All processing steps completed successfully!")


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process evaluation results and update k values"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Step 1: Classify scores
    classify_parser = subparsers.add_parser('classify', help='Classify results by score')
    classify_parser.add_argument('--input', required=True, help='Input evaluation file')
    classify_parser.add_argument('--output-high', required=True, help='Output file for high scores')
    classify_parser.add_argument('--output-low', required=True, help='Output file for low scores')
    
    # Step 2: Reflect and update
    reflect_parser = subparsers.add_parser('reflect', help='Reflect and update k values')
    reflect_parser.add_argument('--input', required=True, help='Input low scores file')
    reflect_parser.add_argument('--output', required=True, help='Output file for updated k values')
    
    # Step 3: Update original
    update_parser = subparsers.add_parser('update', help='Update original dataset k values')
    update_parser.add_argument('--original', required=True, help='Original dataset file')
    update_parser.add_argument('--k-values', required=True, help='File with updated k values')
    update_parser.add_argument('--output', required=True, help='Output file for updated dataset')
    
    # Run all steps
    all_parser = subparsers.add_parser('all', help='Run all processing steps')
    all_parser.add_argument('--evaluation', required=True, help='Evaluation results file')
    all_parser.add_argument('--original', required=True, help='Original dataset file')
    all_parser.add_argument('--output-dir', required=True, help='Output directory')
    
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the processing pipeline.
    """
    args = get_args()
    
    if args.command == 'classify':
        step1_classify_scores(args.input, args.output_high, args.output_low)
    elif args.command == 'reflect':
        step2_reflect_and_update_k(args.input, args.output)
    elif args.command == 'update':
        step3_update_original_k_values(args.original, args.k_values, args.output)
    elif args.command == 'all':
        run_all_steps(args.evaluation, args.original, args.output_dir)
    else:
        print("Please specify a command: classify, reflect, update, or all")
        print("Use --help for more information")


if __name__ == "__main__":
    main()









