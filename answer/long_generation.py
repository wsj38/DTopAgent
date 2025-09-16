#!/usr/bin/env python3
"""
Long Generation Script for DTopAgent

This script generates long-form answers using a language model pipeline.
It processes questions with optional document context and generates detailed responses.

"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import torch
import transformers
from transformers import AutoTokenizer


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate long-form answers using language model pipeline"
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="/data/wsj/model/LLM-Research/Meta-Llama-3-8B-Instruct",
        help="Path to the language model"
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path to input JSON file containing questions and metadata"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help="Path to output JSON file for generated answers"
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help="Sampling temperature for generation"
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    args = parser.parse_args()
    return args


def load_qa(input_path: str) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Load questions, answers, and documents from a JSONL file.
    
    Args:
        input_path: Path to the input JSONL file
        
    Returns:
        Tuple containing lists of questions, answers, and documents
    """
    questions = []
    answers = []
    documents = []
    
    with open(input_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line.strip())
                questions.append(data.get('question', ''))
                answers.append(data.get('answer', ''))
                documents.append(data.get('documents', []))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"An error occurred on line {line_num}: {e}")
    
    return questions, answers, documents


def format_documents(meta_list: List[Dict[str, Any]]) -> str:
    """
    Format document metadata into a readable string.
    
    Args:
        meta_list: List of document metadata dictionaries
        
    Returns:
        Formatted string of documents
    """
    if not meta_list:
        return ""
    
    contents = [meta.get("content", "") for meta in meta_list]
    formatted_docs = "\n".join(
        f"Document {idx + 1}: {doc}" for idx, doc in enumerate(contents)
    )
    return formatted_docs.strip()


def create_messages(question: str, meta_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Create conversation messages for the language model.
    
    Args:
        question: The question to be answered
        meta_list: List of document metadata
        
    Returns:
        List of message dictionaries for the conversation
    """
    if meta_list:  # With document context
        formatted_docs = format_documents(meta_list)
        user_input = (
            f"Answer the following question in a detailed and informative manner "
            f"based on the provided documents. "
            f"If the documents do not contain enough information to accurately "
            f"answer the question, reply only with 'No relevant information'. "
            f"Do not guess, do not fabricate, and do not replace the subject of "
            f"the question with other similar names. "
            f"Do not include any explanation about the process. "
            f"Just return the final answer as a paragraph.\n\n"
            f"Documents:\n{formatted_docs}\n\n"
            f"Question: {question}"
        )
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful and honest assistant, and please use "
                          "documents provided to answer the question. If the answer "
                          "cannot be found in the documents, reply only with "
                          "'No relevant information'."
            },
            {"role": "user", "content": user_input},
        ]
    else:  # Without document context
        user_input = (
            f"Answer the following question in a detailed and informative manner "
            f"based on your knowledge. "
            f"Do not include any explanation about the process. "
            f"Just return the final answer as a paragraph.\n\n"
            f"Question: {question}"
        )
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful and honest assistant, and please use "
                          "your knowledge to answer the question."
            },
            {"role": "user", "content": user_input},
        ]
    
    return messages


def generate_answer(
    pipeline: transformers.Pipeline,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9
) -> str:
    """
    Generate an answer using the language model pipeline.
    
    Args:
        pipeline: The transformers pipeline
        messages: List of conversation messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Generated answer text
    """
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = pipeline(
        messages,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    return outputs[0]["generated_text"][-1]["content"]

def main() -> None:
    """
    Main function to run the long generation pipeline.
    """
    args = get_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    print(f"Processing input file: {args.input}")
    results = []
    
    try:
        # Read JSON file
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        total_items = len(data)
        print(f"Processing {total_items} items...")
        
        for i, item in enumerate(data, 1):
            print(f"Processing item {i}/{total_items}")
            
            question = item.get("question", "")
            meta_list = item.get("meta", [])
            ground_truth = item.get("ground_truth", [])
            
            if not question:
                print(f"Warning: Empty question at item {i}, skipping...")
                continue
            
            # Create messages for the conversation
            messages = create_messages(question, meta_list)
            
            # Generate answer
            try:
                answer = generate_answer(
                    pipeline=pipeline,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                print(f"Generated answer: {answer[:100]}...")
                
                # Save result
                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": answer,
                    "meta": meta_list
                })
                
            except Exception as e:
                print(f"Error generating answer for item {i}: {e}")
                # Add empty result to maintain order
                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": "",
                    "meta": meta_list,
                    "error": str(e)
                })
        
        # Write results to JSON file
        print(f"Saving results to: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=2)
        
        print(f"Successfully processed {len(results)} items")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        raise


if __name__ == '__main__':
    main()



