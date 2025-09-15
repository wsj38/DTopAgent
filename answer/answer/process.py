#!/usr/bin/env python3
"""
Answer Meta Processor

Trim or clear the `meta` documents for each record based on predicted `k`:
- k == "0": remove all meta (knowledge-only answering)
- k == "null": keep meta unchanged
- k in {"1","2","3","4","5",...}: keep only top-k documents

Reads an input JSON array and writes the processed array to output JSON.
"""

import argparse
import json
import logging
from typing import Any, Dict, List


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_meta_by_k(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in data:
        k_value = item.get("k")
        meta_list = item.get("meta", []) or []

        if k_value == "0":
            item["meta"] = []
        elif k_value == "null":
            item["meta"] = meta_list
        else:
            try:
                k_int = int(k_value)  # may raise ValueError
                if k_int < 0:
                    k_int = 0
                item["meta"] = meta_list[:k_int]
            except (TypeError, ValueError):
                # Unknown k → keep meta unchanged
                logger.warning(f"Invalid k '{k_value}' encountered; keeping meta unchanged.")
                item["meta"] = meta_list
    return data


def process_file(input_file: str, output_file: str) -> None:
    with open(input_file, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)

    processed_data = process_meta_by_k(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Processed data written to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim meta documents based on predicted k values.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_file(args.input, args.output)
