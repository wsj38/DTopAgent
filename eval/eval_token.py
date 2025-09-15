#!/usr/bin/env python3
"""
Token Counter for AdaComp SPARK

This script counts tokens in JSON data files using tiktoken encoding.
It analyzes content fields and provides token statistics for text processing.

Author: AdaComp Team
Date: 2024
"""

import argparse
import json
import re
from typing import List, Dict, Any, Tuple
import tiktoken


def get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """
    Get tiktoken tokenizer for token counting.
    
    Args:
        encoding_name: Name of the encoding to use
        
    Returns:
        tiktoken.Encoding object
    """
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """
    Count tokens in text using tiktoken encoding.
    
    Args:
        text: Input text to count tokens for
        encoding: tiktoken encoding object
        
    Returns:
        Number of tokens
    """
    return len(encoding.encode(text))


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra spaces and punctuation spacing.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:，。！？；：])", r"\1", text)
    # Replace multiple spaces with single space
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_content_from_meta(meta_list: List[Dict[str, Any]]) -> str:
    """
    Extract and combine content from metadata list.
    
    Args:
        meta_list: List of metadata dictionaries
        
    Returns:
        Combined content string
    """
    contents = []
    for meta in meta_list:
        content = meta.get("content", "")
        if content:
            contents.append(content)
    
    return " ".join(contents)


def compute_average_tokens(file_path: str, encoding_name: str = "cl100k_base") -> Tuple[float, Dict[str, Any]]:
    """
    Compute average tokens per item in JSON file.
    
    Args:
        file_path: Path to JSON file
        encoding_name: Name of tiktoken encoding to use
        
    Returns:
        Tuple of (average_tokens, statistics)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")
    
    encoding = get_tokenizer(encoding_name)
    total_tokens = 0
    total_items = 0
    token_counts = []
    
    for item in data:
        meta_list = item.get("meta", [])
        combined_content = extract_content_from_meta(meta_list)
        
        # Normalize text
        normalized_content = normalize_text(combined_content)
        
        # Count tokens
        tokens = count_tokens(normalized_content, encoding)
        total_tokens += tokens
        total_items += 1
        token_counts.append(tokens)
    
    if total_items == 0:
        return 0.0, {"total_items": 0, "total_tokens": 0, "token_counts": []}
    
    average_tokens = total_tokens / total_items
    
    statistics = {
        "total_items": total_items,
        "total_tokens": total_tokens,
        "average_tokens": average_tokens,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "token_counts": token_counts
    }
    
    return average_tokens, statistics


def compute_crag_tokens(file_path: str, encoding_name: str = "cl100k_base") -> Tuple[float, Dict[str, Any]]:
    """
    Compute average tokens for CRAG-style data (list of strings).
    
    Args:
        file_path: Path to JSON file containing list of strings
        encoding_name: Name of tiktoken encoding to use
        
    Returns:
        Tuple of (average_tokens, statistics)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")
    
    encoding = get_tokenizer(encoding_name)
    total_tokens = 0
    total_items = 0
    token_counts = []
    
    for item in data:
        if isinstance(item, str) and item.strip():
            # Normalize text
            normalized_content = normalize_text(item)
            
            # Count tokens
            tokens = count_tokens(normalized_content, encoding)
            total_tokens += tokens
            total_items += 1
            token_counts.append(tokens)
    
    if total_items == 0:
        return 0.0, {"total_items": 0, "total_tokens": 0, "token_counts": []}
    
    average_tokens = total_tokens / total_items
    
    statistics = {
        "total_items": total_items,
        "total_tokens": total_tokens,
        "average_tokens": average_tokens,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "token_counts": token_counts
    }
    
    return average_tokens, statistics


def main(input_file: str, output_file: str = None, mode: str = "standard", encoding_name: str = "cl100k_base") -> None:
    """
    Main function to compute token statistics.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save results
        mode: Processing mode ("standard" or "crag")
        encoding_name: Name of tiktoken encoding to use
    """
    print(f"Loading data from: {input_file}")
    
    if mode == "crag":
        average_tokens, statistics = compute_crag_tokens(input_file, encoding_name)
        print("Processing CRAG-style data (list of strings)")
    else:
        average_tokens, statistics = compute_average_tokens(input_file, encoding_name)
        print("Processing standard data (items with meta fields)")
    
    print(f"总条数（item数）: {statistics['total_items']}")
    print(f"总tokens数: {statistics['total_tokens']}")
    print(f"所有content合并后，每条数据的平均token数: {average_tokens:.2f}")
    
    if statistics['total_items'] > 0:
        print(f"最小token数: {statistics['min_tokens']}")
        print(f"最大token数: {statistics['max_tokens']}")
    
    # Save results if output file specified
    if output_file:
        results = {
            "input_file": input_file,
            "mode": mode,
            "encoding": encoding_name,
            "statistics": statistics
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
        description="Count tokens in JSON data files using tiktoken encoding"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save results (optional)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['standard', 'crag'],
        default='standard',
        help='Processing mode: standard (items with meta) or crag (list of strings)'
    )
    parser.add_argument(
        '--encoding',
        type=str,
        default='cl100k_base',
        help='tiktoken encoding name (default: cl100k_base)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args.input, args.output, args.mode, args.encoding)