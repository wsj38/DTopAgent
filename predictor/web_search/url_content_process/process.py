import requests
from bs4 import BeautifulSoup
import json
import re
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup, NavigableString


# 定义超时处理函数
def timeout():
    raise RuntimeError("Timeout")


def clean_page_text(url, min_length=10, timeout_sec=10, max_retries=1):
    """
    解析网页内容并返回清洗过的段落列表。
    
    参数：
        url (str): 要请求的网页链接
        min_length (int): 段落最小长度，过滤短文本
        timeout_sec (int): 请求超时时间（秒）
        max_retries (int): 最大重试次数
    
    返回：
        List[str]: 每段为字符串，形式为 "标题: 段落内容"
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout_sec)
            response.raise_for_status()
            # 设置正确的编码（防止乱码）
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            break
        except (requests.RequestException, ValueError) as e:
            print(f"[{attempt+1}/{max_retries}] Failed to fetch {url}: {e}")
            time.sleep(1)
    else:
        return []
    
    paragraphs = soup.find_all('p')
    cleaned_paragraphs = []

    for p in paragraphs:
        text = p.get_text(separator=' ', strip=True)
        # 清除特殊空格字符
        text = re.sub(r'[\xa0\u200b\u202f]', ' ', text)
        # 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) >= min_length:
            cleaned_paragraphs.append(text)

    return "\n".join(cleaned_paragraphs)


def process_urls(input_path, output_path):
    """
    处理输入的 JSON 文件，获取每个 URL 的清洗内容，并写入输出的 JSONL 文件。
    
    参数：
        input_path (str): 输入文件的路径（JSON）
        output_path (str): 输出文件的路径（JSONL）
    """
    # 加载原始 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 逐条处理并写入 JSONL
    with open(output_path, "a", encoding="utf-8") as out_f:
        # 遍历每条数据，获取每个 URL 的内容
        for item in data:
            for result in item.get("results", []):
                url = result.get("url", "")
                print(f"Fetching content from: {url}")
                # 获取网页内容并清洗
                result["context"] = clean_page_text(url)
                if not result["context"]:
                    result["context"] = result.get("snippet", "")
            
            # 写入结果到 JSONL
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 所有数据已逐条写入 {output_path}")


# 调用函数，传入输入输出路径
if __name__ == "__main__":
    input_path = ""
    output_path = ""
    
    process_urls(input_path, output_path)





