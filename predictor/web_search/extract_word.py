import re
import json
import multiprocessing
import requests
from tqdm import tqdm


class ChatClientOllama:
    def __init__(self,
                 url="url链接",
                 model_name="Qwen2.5-72B-Instruct-GPTQ-Int4",
                 temperature=0.8  # 默认温度参数
                 ) -> None:
        self.url = url
        self.model_name = model_name
        self.temperature = temperature  # 温度参数
    
    def get_response(self, user_prompt):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert NLP assistant. Your task is to extract at most three keywords from user queries to be used for web search. Return the keywords as a concise, comma-separated list, without additional explanation."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature
        }
        res = requests.post(self.url, json=payload)
        response = res.json()['choices'][0]['message']['content']
        return response  


def extract_keywords(questions, task):
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


def generate_knowledge_q(questions, task, mode):
    if task == 'bio':
        queries = [q[7:-1] for q in questions]
    else:
        queries = extract_keywords(questions, task)
    
    if mode == 'wiki':
        search_queries = ["Wikipedia, " + e for e in queries]
    else:
        search_queries = queries
    return search_queries


def extract_questions(json_path):
    questions = []
    ground_truths=[]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 读取整个 JSON 文件为列表
        for item in data:
            if 'question' in item:
                questions.append(item['question'])
            if 'ground_truth' in item:
                ground_truths.append(item['ground_truth'])
    return questions, ground_truths


def process_questions_and_generate_queries(input_json_path, output_json_path, task="bio", mode="wiki"):
    # 提取问题和真实答案
    questions, ground_truths = extract_questions(input_json_path)

    # 生成查询
    search_queries = generate_knowledge_q(questions, task, mode)
    
    # 构造每条为 {"question": ..., "query": ...} 的格式
    paired_data = [
        {"question": q, "query": sq, "ground_truth": ground_truth}
        for q, sq, ground_truth in zip(questions, search_queries, ground_truths)
    ]
    
    # 输出为 JSON 格式
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(paired_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_json_path = ""
    output_json_path = ""
    
    process_questions_and_generate_queries(input_json_path, output_json_path, task="bio", mode="wiki")
