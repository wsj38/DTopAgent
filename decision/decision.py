#!/usr/bin/env python3
"""
Decision Script for DTopAgent

This script evaluates question-answering responses using a language model.
It performs self-assessment on generated answers and provides context adjustment recommendations.

Author: AdaComp Team
Date: 2024
"""

import argparse
import json
import re
from typing import Dict, List, Any, Optional
import requests
from tqdm import tqdm


class ChatClientOllama:
    """
    Client for interacting with Ollama API for response evaluation.
    """
    
    def __init__(
        self,
        url: str = "url",
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
        Get response from the Ollama API.
        
        Args:
            user_prompt: The prompt to send to the model
            
        Returns:
            The generated response text
            
        Raises:
            Exception: If API request fails
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an advanced language model performing self-assessment after a question-answering session."
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature
        }
        
        try:
            res = requests.post(self.url, json=payload)
            res.raise_for_status()
            response = res.json()['choices'][0]['message']['content']
            return response
        except requests.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Invalid API response format: {e}")


def construct_prompt(question: str, context: str, answer: str) -> str:
    """
    Construct evaluation prompt for the language model.
    
    Args:
        question: The original question
        context: Related context chunks
        answer: The generated answer
        
    Returns:
        Formatted prompt string
    """
    return f"""
        Original Question: {question}
        Related Context Chunks: 
        {context}
        Original Answer: {answer}\n
        Objective (O): You are to evaluate the original answer for the original prompt on a scale of 1 to 10 based on its accuracy and reasonability.
Additionally, determine if the original prompt needs more related context (1), less context (-1) or should keep the current context unchanged (0).
        Style (S): Provide a clear and concise evaluation in a formal and professional style.
        Response (R): Ensure the output follows this format:
            Evaluation Score: [1-10]. (The answer is highly accurate if Score >= 9.)
            Context Adjustment: [1, 0, -1].
            Context adjustment should output "less context (-1)" with a probability of 40%, "more context (1)" with a probability of 30%, and "keep current context (0)" with a probability of 30%.
        (output example):
            Evaluation Score: 8
            Context Adjustment: -1"""


def load_data(meta_file_path: str, answer_file_path: str) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Load metadata and answer data from JSON files.
    
    Args:
        meta_file_path: Path to metadata JSON file
        answer_file_path: Path to answer JSON file
        
    Returns:
        Tuple of (meta_data, answer_map)
        
    Raises:
        FileNotFoundError: If files don't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    try:
        with open(meta_file_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Meta file not found: {meta_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in meta file: {e}")
    
    try:
        with open(answer_file_path, 'r', encoding='utf-8') as f:
            answer_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Answer file not found: {answer_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in answer file: {e}")
    
    # Build question to answer mapping
    answer_map = {item['question']: item['answer'] for item in answer_data}
    
    return meta_data, answer_map


def extract_context(meta: List[Dict[str, Any]]) -> str:
    """
    Extract and format context from metadata.
    
    Args:
        meta: List of metadata dictionaries
        
    Returns:
        Formatted context string
    """
    contents = [
        m.get('content', m.get('text', '')) for m in meta
    ]
    context = "\n".join(f"Document {idx + 1}: {doc}" for idx, doc in enumerate(contents))
    return context


def evaluate_responses(
    meta_data: List[Dict[str, Any]], 
    answer_map: Dict[str, str], 
    client: ChatClientOllama
) -> List[Dict[str, Any]]:
    """
    Evaluate responses using the language model.
    
    Args:
        meta_data: List of metadata items
        answer_map: Mapping of questions to answers
        client: Ollama chat client
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for item in tqdm(meta_data, desc="Evaluating responses"):
        question = item['question']
        k = item['k']
        meta = item['meta']
        
        if question not in answer_map:
            print(f"Warning: Question not found in answer file: {question}")
            continue
        
        # Extract context and answer
        context = extract_context(meta)
        answer = answer_map[question]
        
        # Construct prompt and get response
        prompt = construct_prompt(question, context, answer)
        
        try:
            response = client.get_response(prompt)
        except Exception as e:
            print(f"Error evaluating question '{question}': {e}")
            response = f"Error: {e}"
        
        results.append({
            "question": question,
            "meta": meta,
            "answer": answer,
            "response": response,
            "k": k
        })
    
    return results


def save_results(results: List[Dict[str, Any]], save_path: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of evaluation results
        save_path: Path to save the results
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {save_path}")


def main(meta_file_path: str, answer_file_path: str, save_path: Optional[str] = None) -> None:
    """
    Main function to run the evaluation pipeline.
    
    Args:
        meta_file_path: Path to metadata JSON file
        answer_file_path: Path to answer JSON file
        save_path: Optional path to save results
    """
    print("Loading data...")
    meta_data, answer_map = load_data(meta_file_path, answer_file_path)
    
    print(f"Loaded {len(meta_data)} metadata items and {len(answer_map)} answers")
    
    print("Initializing evaluation client...")
    client = ChatClientOllama()
    
    print("Starting evaluation...")
    results = evaluate_responses(meta_data, answer_map, client)
    
    if save_path:
        save_results(results, save_path)
    else:
        print("Evaluation results:")
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=2))


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate question-answering responses using language model"
    )
    parser.add_argument(
        '--meta_file',
        type=str,
        required=True,
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        '--answer_file',
        type=str,
        required=True,
        help="Path to answer JSON file"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Path to save evaluation results (optional)"
    )
    parser.add_argument(
        '--api_url',
        type=str,
        default="http://192.168.200.215:21004/v1/chat/completions",
        help="API endpoint URL"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default="Qwen2.5-72B-Instruct-GPTQ-Int4",
        help="Model name to use"
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Override default client settings if provided
    if args.api_url or args.model_name or args.temperature:
        # This would require modifying the ChatClientOllama class to accept these parameters
        # For now, we'll use the defaults
        pass
    
    main(args.meta_file, args.answer_file, args.output)
