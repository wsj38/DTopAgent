import json
import os
from tqdm import tqdm
from PIL import Image
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig

# 加载DPR模型
dpr_model_name = "../model/dpr-question_encoder-single-nq-base"  # 可以根据需要选择其他预训练模型
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_model_name)
dpr_model = DPRQuestionEncoder.from_pretrained(dpr_model_name).to('cuda').eval()
# 获取句子特征
def get_sentence_features(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')   
    with torch.no_grad():
        outputs = dpr_model(**inputs)
        text_features = outputs.pooler_output.cpu()
    
    return text_features


# 句子到句子检索函数(8的版本)
def sentence_retrieve_sentence(url, user, pwd, idx, sentence, question_id):
    es = Elasticsearch(url, basic_auth=(user, pwd))
    sentence_embedding = get_sentence_features(sentence)
    search_query = {
        "knn": {
            "field": "context_embedding",
            "k": 5,
            "num_candidates": 10,
            "query_vector": sentence_embedding[0].tolist(),
            "filter": {
                "term": {
                    "question_id": question_id
                }
            }
        },
        "_source": False,
        "fields": ["title", "content"]
    }

    # # 执行检索
    res = es.search(index=idx, body=search_query)
    results = []

    # 提取检索结果
    for item in res["hits"]["hits"]:
        meta = {}
        meta["title"] = item["fields"].get("title", [None])[0]
        meta["content"] = item["fields"].get("content", [None])[0]
        meta["score"]= item["_score"]  # 添加相似度分数
        results.append(meta)
    
    return results


# 测试用例
if __name__ == '__main__':
    all_results = []  # 用于保存所有问题的检索结果
    json_file_path = ""  # 这里需要提供实际的 JSON 文件路径
    output_file_path = ""  # 这里是输出的文件路径
    # 检索与句子相似的句子
    print("\n检索与句子相似的句子：")
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as load_f:
            records = json.load(load_f)
            for record in tqdm(records):
                question = record.get("question", None)
                question_id = record.get("question_id")
                if question and question_id:
                    # 对每个问题执行检索
                    retrieved_results = sentence_retrieve_sentence('url', '用户名', '密码', "索引名称", question, question_id)
                    
                    # 保存结果，结构为 question 和 meta
                    result = {
                        "question_id": question_id,
                        "question": question,
                        "meta": retrieved_results
                    }
                    all_results.append(result)
    
        # 写入到新的 JSON 文件
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            json.dump(all_results, out_f, ensure_ascii=False, indent=4)

        print(f"检索结果已写入: {output_file_path}")

