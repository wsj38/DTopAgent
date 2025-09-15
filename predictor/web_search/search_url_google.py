#!/usr/bin/env python3
"""
Google (Serper) Web Search Helper

This script queries the Serper Google Search API for a list of queries and
produces structured results (title, url, snippet) suitable for downstream
processing in the AdaComp SPARK pipeline.

Features:
- Robust retries with backoff and request timeouts
- Progress display via tqdm
- CLI for batch processing input/output files

Input JSON format (array):
[
  {"question": "...", "query": "...", "ground_truth": ["..."]},
  ...
]

Output JSON format (array):
[
  {
    "question": "...",
    "query": "...",
    "ground_truth": ["..."],
    "results": [
      {"title": "...", "url": "...", "snippet": "..."},
      ...
    ]
  },
  ...
]
"""

import argparse
import json
import logging
import time
from typing import Any, Dict, List

import requests
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def search(
    queries: List[str],
    api_key: str,
    endpoint: str = "https://google.serper.dev/search",
    max_results: int = 10,
    retries: int = 3,
    timeout_sec: int = 20,
    backoff_sec: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Perform web search for a list of queries using Serper API.

    Returns list of dicts: {"queries": <query>, "results": <raw_results>}.
    """
    search_results: List[Dict[str, Any]] = []

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    for query in tqdm(queries[:], desc="Searching for urls..."):
        payload = json.dumps({"q": query})

        attempt = 0
        last_exc: Exception | None = None
        response_json: Dict[str, Any] = {}
        while attempt < retries:
            try:
                resp = requests.post(endpoint, headers=headers, data=payload, timeout=timeout_sec)
                resp.raise_for_status()
                response_json = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                attempt += 1
                last_exc = exc
                logger.warning(f"Request failed ({attempt}/{retries}) for query '{query}': {exc}")
                time.sleep(backoff_sec * attempt)

        if attempt == retries and last_exc is not None:
            logger.error(f"All retries failed for query '{query}': {last_exc}")
            search_results.append({"queries": query, "results": []})
            continue

        organic = response_json.get("organic", [])
        if not isinstance(organic, list):
            organic = []

        # Keep raw items; downstream will pick fields
        results = organic[: max(0, max_results)]
        search_results.append({"queries": query, "results": results})

    return search_results


def process_search_queries(
    input_json_path: str,
    output_json_path: str,
    api_key: str,
    endpoint: str = "https://google.serper.dev/search",
    max_results: int = 10,
    retries: int = 3,
    timeout_sec: int = 20,
    backoff_sec: float = 1.5,
) -> None:
    """Load queries from input JSON, perform searches, and write structured results."""
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)

    final_output: List[Dict[str, Any]] = []

    for item in data:
        search_query = item.get('query', '')
        if not search_query:
            final_output.append({
                'question': item.get('question', ''),
                'query': '',
                'ground_truth': item.get('ground_truth', []),
                'results': []
            })
            continue

        results_wrapped = search(
            [search_query],
            api_key=api_key,
            endpoint=endpoint,
            max_results=max_results,
            retries=retries,
            timeout_sec=timeout_sec,
            backoff_sec=backoff_sec,
        )
        results = results_wrapped[0].get('results', []) if results_wrapped else []

        entries = [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in results[: max_results]
            if isinstance(r, dict) and "link" in r and "title" in r and "snippet" in r
        ]

        final_output.append({
            'question': item.get('question', ''),
            'query': item.get('query', ''),
            'ground_truth': item.get('ground_truth', []),
            'results': entries
        })

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search URLs via Serper Google API and save results.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--api-key", required=True, help="Serper API key")
    parser.add_argument("--endpoint", default="https://google.serper.dev/search", help="Serper API endpoint")
    parser.add_argument("--max-results", type=int, default=10, help="Max results per query")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries on failure")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds")
    parser.add_argument("--backoff", type=float, default=1.5, help="Backoff base seconds between retries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        process_search_queries(
            input_json_path=args.input,
            output_json_path=args.output,
            api_key=args.api_key,
            endpoint=args.endpoint,
            max_results=args.max_results,
            retries=args.retries,
            timeout_sec=args.timeout,
            backoff_sec=args.backoff,
        )
        logger.info("Web search completed successfully.")
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.error(f"Web search failed: {exc}")
        raise


if __name__ == "__main__":
    main()
