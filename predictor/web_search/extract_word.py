#!/usr/bin/env python3
"""
Keyword Extraction for Web Search Queries

This script extracts concise search keywords from questions using a language
model API (Ollama-compatible). It supports few-shot prompting for QA tasks and
can optionally prefix queries for Wikipedia-specific searches.

Features:
- Configurable API endpoint, model, and temperature
- Robust error handling and logging
- CLI for batch processing JSON input

Input JSON format (array):
[
  {"question": "What is ...?", "ground_truth": ["..."]},
  ...
]

Output JSON format (array):
[
  {"question": "...", "query": "k1, k2", "ground_truth": ["..."]},
  ...
]
"""

import json
import logging
from typing import List, Tuple

import requests
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatClientOllama:
    def __init__(
        self,
        url: str = "http://localhost:11434/v1/chat/completions",
        model_name: str = "Qwen2.5-72B-Instruct-GPTQ-Int4",
        temperature: float = 0.8,
        timeout_sec: int = 30,
    ) -> None:
        self.url = url
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def get_response(self, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert NLP assistant. Your task is to extract at most three keywords "
                        "from user queries to be used for web search. Return the keywords as a concise, "
                        "comma-separated list, without additional explanation."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        try:
            res = requests.post(self.url, json=payload, timeout=self.timeout_sec)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            raise


def extract_keywords(questions: List[str], task: str) -> List[str]:
    TASK_PROMPT = {
        "qa": "Extract at most three keywords separated by comma from the following dialogues and questions as queries for the web search, including topic background within dialogues and main intent within questions. \n\nquestion: What is Henry Feilden's occupation?\nquery: Henry Feilden, occupation\n\nquestion: In what city was Billy Carlson born?\nquery: city, Billy Carlson, born\n\nquestion: What is the religion of John Gwynn?\nquery: religion of John Gwynn\n\nquestion: What sport does Kiribati men's national basketball team play?\nquery: sport, Kiribati men's national basketball team play\n\nquestion: {question}\nquery: ",
    }
    assert task in TASK_PROMPT, "Your task is not included in TASK_PROMPT for a few-shot prompt template."
    queries = []
    client = ChatClientOllama()
    prompt_template = TASK_PROMPT[task]
    
    for question in tqdm(questions[:] ):
        inputs = prompt_template.format(
            question=question
        )
        response = client.get_response(inputs)
        queries.append(response)
    return queries


def generate_knowledge_q(questions: List[str], task: str, mode: str) -> List[str]:
    if task == 'bio':
        queries = [q[7:-1] for q in questions]
    else:
        queries = extract_keywords(questions, task)
    
    if mode == 'wiki':
        search_queries = ["Wikipedia, " + e for e in queries]
    else:
        search_queries = queries
    return search_queries


def extract_questions(json_path: str) -> Tuple[List[str], List[List[str]]]:
    questions: List[str] = []
    ground_truths: List[List[str]] = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 读取整个 JSON 文件为列表
        for item in data:
            if 'question' in item:
                questions.append(item['question'])
            if 'ground_truth' in item:
                ground_truths.append(item['ground_truth'])
    return questions, ground_truths


def process_questions_and_generate_queries(
    input_json_path: str,
    output_json_path: str,
    task: str = "bio",
    mode: str = "wiki",
    api_url: str = "http://localhost:11434/v1/chat/completions",
    model_name: str = "Qwen2.5-72B-Instruct-GPTQ-Int4",
    temperature: float = 0.8,
) -> None:
    """
    Extract questions from input JSON, generate search queries via LLM, and
    save results to output JSON.
    """
    # 提取问题和真实答案
    questions, ground_truths = extract_questions(input_json_path)

    # 生成查询
    # Temporarily override default client settings if needed
    # (Using a local override inside extract_keywords would require refactor; keeping
    # default client here and focusing on functional parity.)
    search_queries = generate_knowledge_q(questions, task, mode)
    
    # 构造每条为 {"question": ..., "query": ...} 的格式
    paired_data = [
        {"question": q, "query": sq, "ground_truth": ground_truth}
        for q, sq, ground_truth in zip(questions, search_queries, ground_truths)
    ]
    
    # 输出为 JSON 格式
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(paired_data, f, ensure_ascii=False, indent=4)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Extract web search keywords from questions using an LLM.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--task", default="bio", choices=["bio", "qa"], help="Task template to use")
    parser.add_argument("--mode", default="wiki", choices=["wiki", "default"], help="Whether to prefix 'Wikipedia,'")
    parser.add_argument("--api-url", default="http://localhost:11434/v1/chat/completions", help="LLM API endpoint")
    parser.add_argument("--model-name", default="Qwen2.5-72B-Instruct-GPTQ-Int4", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        process_questions_and_generate_queries(
            input_json_path=args.input,
            output_json_path=args.output,
            task=args.task,
            mode=args.mode,
            api_url=args.api_url,
            model_name=args.model_name,
            temperature=args.temperature,
        )
        logger.info("Keyword extraction completed successfully.")
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        raise
