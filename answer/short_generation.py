#!/usr/bin/env python3
"""
Short Answer Generation

Generate concise answers for a list of questions using an instruction-tuned
text-generation model. If retrieved documents (meta) are provided, they are used
as supporting context; otherwise the model answers from its knowledge.

Input JSON format (array):
[
  {"question": "...", "meta": [{"content": "..."}, ...], "ground_truth": ["..."]},
  ...
]

Output JSON format (array):
[
  {"question": "...", "ground_truth": ["..."], "answer": "..."},
  ...
]
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


def generate_short_answers(
    model_id: str,
    input_path: str,
    output_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> None:
    """Generate short answers and write results to JSON file."""
    logger.info(f"Loading model pipeline: {model_id}")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(tqdm(data, desc="Generating short answers"), 1):
        question = item.get("question", "")
        meta_list = item.get("meta", [])
        ground_truth = item.get("ground_truth", [])

        if meta_list:
            contents = [meta.get("content", "") for meta in meta_list]
            formatted_docs = "\n".join(f"Document {i + 1}: {doc}" for i, doc in enumerate(contents))
            user_input = (
                "Answer the following question as briefly as possible. "
                "Return only the final answer without explanation or extra words.\n\n"
                f"Documents:\n{formatted_docs.strip()}\n\n"
                f"Question: {question}"
            )
            messages = [
                {"role": "system", "content": "You are a helpful, respectful and honest assistant, and please use documents provided to answer the question."},
                {"role": "user", "content": user_input},
            ]
        else:
            user_input = (
                "Answer the following question as briefly as possible. "
                "Return only the final answer without explanation or extra words.\n\n"
                f"Question: {question}"
            )
            messages = [
                {"role": "system", "content": "You are a helpful, respectful and honest assistant, and please use your knowledge to answer the question."},
                {"role": "user", "content": user_input},
            ]

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
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
            logger.error(f"Generation failed on item {idx}: {exc}")
            answer = ""

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
        })

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    logger.info(f"Saved results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate short answers with an instruction-tuned model.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_short_answers(
        model_id=args.model_id,
        input_path=args.input,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )



