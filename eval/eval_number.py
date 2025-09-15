#!/usr/bin/env python3
"""
Content Counter for AdaComp SPARK

This script counts the number of content fields in JSON data files.
It analyzes metadata structures and provides statistics about content distribution.

Author: AdaComp Team
Date: 2024
"""

import argparse
import json
from typing import List, Dict, Any, Tuple


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
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


def count_contents(data: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Count total items and content fields in the dataset.
    
    Args:
        data: List of data items
        
    Returns:
        Tuple of (total_items, total_contents)
    """
    total_contents = 0
    
    for item in data:
        if "meta" in item:
            total_contents += sum(1 for m in item["meta"] if "content" in m)
    
    return len(data), total_contents


def analyze_content_distribution(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the distribution of content fields across items.
    
    Args:
        data: List of data items
        
    Returns:
        Dictionary with distribution statistics
    """
    content_counts = []
    items_with_content = 0
    items_without_content = 0
    
    for item in data:
        if "meta" in item:
            content_count = sum(1 for m in item["meta"] if "content" in m)
            content_counts.append(content_count)
            
            if content_count > 0:
                items_with_content += 1
            else:
                items_without_content += 1
        else:
            content_counts.append(0)
            items_without_content += 1
    
    if content_counts:
        avg_content_per_item = sum(content_counts) / len(content_counts)
        max_content_per_item = max(content_counts)
        min_content_per_item = min(content_counts)
    else:
        avg_content_per_item = max_content_per_item = min_content_per_item = 0
    
    return {
        "total_items": len(data),
        "items_with_content": items_with_content,
        "items_without_content": items_without_content,
        "total_contents": sum(content_counts),
        "avg_content_per_item": avg_content_per_item,
        "max_content_per_item": max_content_per_item,
        "min_content_per_item": min_content_per_item,
        "content_distribution": content_counts
    }


def main(input_file: str, output_file: str = None, detailed: bool = False) -> None:
    """
    Main function to count and analyze content fields.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save results
        detailed: Whether to show detailed analysis
    """
    print(f"Loading data from: {input_file}")
    data = load_json_data(input_file)
    
    total_items, total_contents = count_contents(data)
    
    print(f"一共有 {total_items} 条数据")
    print(f"总共有 {total_contents} 个 content 字段")
    
    if detailed:
        print("\n详细分析:")
        analysis = analyze_content_distribution(data)
        
        print(f"包含内容的条目: {analysis['items_with_content']}")
        print(f"不包含内容的条目: {analysis['items_without_content']}")
        print(f"平均每条数据的content数量: {analysis['avg_content_per_item']:.2f}")
        print(f"最大content数量: {analysis['max_content_per_item']}")
        print(f"最小content数量: {analysis['min_content_per_item']}")
    
    # Save results if output file specified
    if output_file:
        results = {
            "input_file": input_file,
            "total_items": total_items,
            "total_contents": total_contents,
            "detailed_analysis": analyze_content_distribution(data) if detailed else None
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
        description="Count content fields in JSON data files"
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
        '--detailed',
        action='store_true',
        help='Show detailed analysis'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args.input, args.output, args.detailed)

