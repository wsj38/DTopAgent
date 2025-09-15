import json
import os
from tqdm import tqdm
from PIL import Image
from elasticsearch import Elasticsearch, helpers
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
from elasticsearch import Elasticsearch
import json
from tqdm import tqdm
import os
from elasticsearch.helpers import bulk
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from elasticsearch.helpers import bulk, BulkIndexError
# 加载DPR模型
dpr_model_name = "../model/dpr-ctx_encoder-single-nq-base"  # 可以根据需要选择其他预训练模型
tokenizer = AutoTokenizer.from_pretrained(dpr_model_name)
dpr_model = DPRContextEncoder.from_pretrained(dpr_model_name).to('cuda').eval()

# 获取句子特征
def get_sentence_features(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
    
    with torch.no_grad():
        outputs = dpr_model(**inputs)
        text_features = outputs.pooler_output.cpu()
    
    return text_features

def insert_data_es_single_file(json_file_path, url, user, pwd, idx):
    es = Elasticsearch(url, basic_auth=(user, pwd),
    request_timeout=60 )
    actions = []

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as load_f:
            records = json.load(load_f)
            for record in tqdm(records):
                title = record.get("title", None)
                text = record.get("text", None)
                question_id = record.get("question_id", None)
                if text:  # 有文本再处理
                    doc = {
                        "title": title,
                        "content": text,
                        "context_embedding": get_sentence_features(text)[0].tolist(),  # 嵌入
                        "question_id":question_id
                    }

                    actions.append({"_index": idx, "_source": doc})
    try:
        if actions:
            success, failed = bulk(es, actions, stats_only=False, raise_on_error=False)
            print(f"成功插入 {success} 条，失败 {failed} 条")
    except BulkIndexError as e:
        print("BulkIndexError:", e.errors)  # 打印详细错误

# 测试用例
if __name__ == '__main__':
    # 插入数据
    json_path="输入数据"
    url="url"
    idx="索引名称"
    insert_data_es_single_file(json_path,url, '用户名', '密码', idx)



   