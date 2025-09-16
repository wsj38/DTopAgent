#!/usr/bin/env python3
"""
URL Content Processor

This script fetches and cleans web page content for each URL contained in a JSON
input file and writes the enriched records to a JSONL output file. It is used in
the DTopAgent pipeline to enrich web search results with cleaned page text.

Features:
- Robust HTTP fetching with retries and timeouts
- HTML parsing via BeautifulSoup
- Text normalization (whitespace, special spaces)
- CLI with configurable parameters

Input JSON format (array of records):
{
  "results": [
    { "url": "https://example.com", "snippet": "..." },
    ...
  ],
  ...
}

Output JSONL format (one JSON per line), where each record has each result enriched with
`context` from the fetched page (fallback to `snippet` if fetching/parsing fails).
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clean_page_text(url: str, min_length: int = 10, timeout_sec: int = 10, max_retries: int = 1, delay_sec: float = 1.0) -> str:
    """
    Fetch and clean visible text from a web page.

    Args:
        url: Webpage URL to fetch
        min_length: Minimum paragraph length to keep
        timeout_sec: Per-request timeout in seconds
        max_retries: Number of retries on request failure
        delay_sec: Delay between retries (seconds)

    Returns:
        Cleaned text content joined by newlines; empty string if failed
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }

    soup = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_sec)
            resp.raise_for_status()
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            break
        except (requests.RequestException, ValueError) as exc:
            logger.warning(f"[{attempt}/{max_retries}] Failed to fetch {url}: {exc}")
            if attempt < max_retries:
                time.sleep(delay_sec)
            else:
                return ""

    assert soup is not None
    paragraphs = soup.find_all("p")
    cleaned_blocks: List[str] = []

    for p in paragraphs:
        text = p.get_text(separator=" ", strip=True)
        # Normalize special spaces
        text = re.sub(r"[\xa0\u200b\u202f]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= min_length:
            cleaned_blocks.append(text)

    return "\n".join(cleaned_blocks)


def process_urls(input_path: str, output_path: str, min_length: int = 10, timeout_sec: int = 10, max_retries: int = 1, delay_sec: float = 1.0, append: bool = False) -> None:
    """
    Process input JSON, fetch and clean content for each URL, and write JSONL.

    Args:
        input_path: Path to input JSON file (array of records)
        output_path: Path to output JSONL file
        min_length: Minimum paragraph length to keep
        timeout_sec: Per-request timeout in seconds
        max_retries: Number of retries on request failure
        delay_sec: Delay between retries (seconds)
        append: If True, append to output; otherwise overwrite
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure output directory exists
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading input JSON from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    mode = "a" if append else "w"
    processed = 0
    failed = 0

    logger.info(f"Writing JSONL to: {output_path} (append={append})")
    with open(output_path, mode, encoding="utf-8") as out_f:
        for item in data:
            results = item.get("results", [])
            for result in results:
                url = result.get("url", "")
                if not url:
                    continue
                logger.info(f"Fetching content from: {url}")
                try:
                    context = clean_page_text(
                        url,
                        min_length=min_length,
                        timeout_sec=timeout_sec,
                        max_retries=max_retries,
                        delay_sec=delay_sec,
                    )
                    result["context"] = context if context else result.get("snippet", "")
                    processed += 1
                except Exception as exc:
                    logger.error(f"Failed to process {url}: {exc}")
                    result["context"] = result.get("snippet", "")
                    failed += 1

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Done. Processed results: {processed}, failed: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and clean web page content for URLs in JSON and write JSONL.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum paragraph length to keep")
    parser.add_argument("--timeout", type=int, default=10, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=1, help="Max retries on request failure")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay (seconds) between retries")
    parser.add_argument("--append", action="store_true", help="Append to output file instead of overwrite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        process_urls(
            input_path=args.input,
            output_path=args.output,
            min_length=args.min_length,
            timeout_sec=args.timeout,
            max_retries=args.retries,
            delay_sec=args.delay,
            append=args.append,
        )
    except Exception as exc:
        logger.error(f"Processing failed: {exc}")
        raise


if __name__ == "__main__":
    main()






