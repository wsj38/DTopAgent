import json
import os
from elasticsearch import Elasticsearch
# # 初始化 Elasticsearch 客户端
es = Elasticsearch(
    'url',
    basic_auth=("用户名", "密码"),
)
# 索引名称
idxnm = "索引名称"

# 检查索引是否存在
if es.indices.exists(index=idxnm):
    es.indices.delete(index=idxnm)
    print(f"索引 {idxnm} 已删除")

# 创建索引
with open("../mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)
es.indices.create(index=idxnm, body=mapping, request_timeout=60)
print(f"索引 {idxnm} 已创建")





