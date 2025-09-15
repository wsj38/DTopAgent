#!/usr/bin/env python3
"""
Predictor: Estimate Optimal K (Number of Documents) per Question

This script loads a text-generation model and, given a set of questions with
retrieved documents (meta), predicts the minimal number of documents K needed
to answer each question. It outputs a JSON file with the original fields plus
the predicted `k` value as a string.
"""

import argparse
import json
import logging
from typing import Any, Dict, List

import torch
import transformers
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_questions(
    model_id: str,
    input_path: str,
    output_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> None:
    """Run K-prediction over questions in input JSON and write results JSON."""
    # Load model pipeline
    logger.info(f"Loading model pipeline: {model_id}")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    results: List[Dict[str, Any]] = []

    # Read input
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    for i, item in enumerate(tqdm(data, desc="Predicting K"), 1):
        question = item.get("question", "")
        meta_list = item.get("meta", [])
        ground_truth = item.get("ground_truth", [])
        contents = [meta.get("content", "") for meta in meta_list]
        formatted_docs = "\n".join(f"Document {idx + 1}: {doc}" for idx, doc in enumerate(contents))

        instruction = (
            "Given a question and a set of retrieved documents, predict how many top documents (K) are needed to answer the question.\n"
            "- If the question is simple or the documents are high-quality and relevant, K should be low.\n"
            "- If the question is complex or the documents are poor or irrelevant, K should be high.\n"
            "- If the model can confidently answer the question without relying on any documents, set K = 0.\n"
            "- If the question cannot be answered by the model even with the help of the documents, set K = null.\n"
            "Your output should be one of: 0, 1, 2, 3, 4, 5, or null."
            f"\nDocuments:\n{formatted_docs.strip()}\n\n"
            f"Question: {question}\n"
        )

        # Send to model
        messages = [
            {
                "role": "system",
                "content": "You are an expert retrieval assistant specializing in determining the minimum number of documents needed to accurately answer a question.",
            },
            {"role": "user", "content": instruction},
        ]
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        try:
            outputs = pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            answer = outputs[0]["generated_text"][-1]["content"]
        except Exception as exc:
            logger.error(f"Generation failed on item {i}: {exc}")
            answer = "null"

        logger.info(f"Item {i} predicted K: {answer}")

        results.append(
            {
                "question": question,
                "meta": meta_list,
                "ground_truth": ground_truth,
                "k": answer,
            }
        )

    # Write output
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    logger.info(f"Saved results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict optimal K for questions using a text-generation model.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_questions(
        model_id=args.model_id,
        input_path=args.input,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
