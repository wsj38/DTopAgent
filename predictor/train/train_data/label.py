#!/usr/bin/env python3
"""
Test script for evaluating candidate answers using Ollama API.

This script processes JSONL files containing questions, candidate answers,
and ground truth to determine which candidate answer best matches the
ground truth using semantic evaluation via a language model.

Author: AdaComp Team
Date: 2024
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatClientOllama:
    """
    Client for interacting with Ollama API for text generation and evaluation.
    
    This class provides a simple interface to communicate with an Ollama
    API endpoint for evaluating candidate answers against ground truth.
    """
    
    def __init__(
        self,
        url: str = "http://192.168.200.215:21004/v1/chat/completions",
        model_name: str = "Qwen2.5-72B-Instruct-GPTQ-Int4",
        temperature: float = 0.8
    ) -> None:
        """
        Initialize the Ollama chat client.
        
        Args:
            url: API endpoint URL
            model_name: Name of the model to use
            temperature: Sampling temperature for generation
        """
        self.url = url
        self.model_name = model_name
        self.temperature = temperature
        
    def get_response(self, user_prompt: str) -> str:
        """
        Get a response from the Ollama API.
        
        Args:
            user_prompt: The user's prompt/question
            
        Returns:
            The model's response as a string
            
        Raises:
            requests.RequestException: If API request fails
            KeyError: If response format is unexpected
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            raise



def construct_prompt(question: str, option: str, ground_truth: str) -> str:
    """
    Construct a prompt for evaluating candidate answers.
    
    Args:
        question: The question being asked
        option: The candidate answer to evaluate
        ground_truth: The ground truth answer(s)
        
    Returns:
        Formatted prompt string for the language model
    """
    return (
        f"Question: {question}\n"
        f"Candidate Answer: {option}\n"
        f"Ground Truth: {ground_truth}\n\n"
        f"Does the candidate answer semantically match any of the ground truth answers? "
        f"Only answer with 'yes' or 'no'."
    )

def process_file(input_path: str, output_path: str, client: ChatClientOllama) -> None:
    """
    Process a JSONL file to evaluate candidate answers.
    
    This function reads a JSONL file containing questions, candidate answers,
    and ground truth, then uses the provided client to evaluate which candidate
    answer best matches the ground truth.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        client: ChatClientOllama instance for API calls
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
        requests.RequestException: If API calls fail
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    processed_count = 0
    error_count = 0

    logger.info(f"Processing file: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing items"):
                try:
                    item = json.loads(line.strip())
                    question = item.get("question", "")
                    ground_truth_list = item.get("ground_truth", [])
                    ground_truth = "; ".join(ground_truth_list) if ground_truth_list else ""
                    final_answer = None

                    # Evaluate candidates 1 to 5
                    for i in range(1, 6):
                        candidate = item.get(str(i), "")
                        if not candidate:  # Skip empty candidates
                            continue
                            
                        prompt = construct_prompt(question, candidate, ground_truth)
                        response = client.get_response(prompt)
                        
                        if "yes" in response.lower():
                            final_answer = str(i)
                            break

                    # If none of 1-5 matched, check candidate 0
                    if final_answer is None:
                        candidate_0 = item.get("0", "")
                        if candidate_0:  # Only check if candidate 0 exists
                            prompt = construct_prompt(question, candidate_0, ground_truth)
                            response = client.get_response(prompt)
                            
                            if "yes" in response.lower():
                                final_answer = "0"
                            else:
                                final_answer = None
                        else:
                            final_answer = None

                    item["final"] = final_answer
                    results.append(item)
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error on line: {e}")
                    error_count += 1
                    continue
                except requests.RequestException as e:
                    logger.error(f"API request error: {e}")
                    error_count += 1
                    continue

        # Write results
        logger.info(f"Writing results to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as fout:
            for item in results:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"Processing completed. Processed: {processed_count}, Errors: {error_count}")
        
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        raise

def main() -> None:
    """
    Main function to run the candidate answer evaluation.
    
    This function sets up the client and processes the specified files.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate candidate answers using Ollama API"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--api-url",
        default="http://192.168.200.215:21004/v1/chat/completions",
        help="Ollama API endpoint URL"
    )
    parser.add_argument(
        "--model-name",
        default="Qwen2.5-72B-Instruct-GPTQ-Int4",
        help="Model name to use for evaluation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = ChatClientOllama(
        url=args.api_url,
        model_name=args.model_name,
        temperature=args.temperature
    )
    
    # Process file
    try:
        process_file(args.input, args.output, client)
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
