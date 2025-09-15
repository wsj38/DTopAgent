import re
import json
import multiprocessing
from tqdm import tqdm

def split_context_by_words_and_sentences(text, max_words=300):
    """
    通过单词和句子拆分文本，确保每个片段最多有 `max_words` 个单词。
    """
    text = re.sub(r'\[\s*\d+\s*\]', '', text)  # 移除数字引用
    words = text.strip().split()
    total_words = len(words)
    context_list = []
    start_idx = 0

    while start_idx < total_words:
        end_idx = min(start_idx + max_words, total_words)
        chunk_words = words[start_idx:end_idx]
        chunk_text = ' '.join(chunk_words)

        if end_idx == total_words:
            context_list.append(chunk_text.strip())
            break

        remaining_words = words[end_idx:]
        sentence_end_idx = 0
        found_end = False
        for i, word in enumerate(remaining_words):
            chunk_text += ' ' + word
            sentence_end_idx += 1
            if re.search(r'[.!?]"?$', word):
                found_end = True
                break

        context_list.append(chunk_text.strip())
        start_idx = end_idx + sentence_end_idx

    # 合并最后两个片段，如果它们不足两句
    if len(context_list) >= 2:
        last_chunk = context_list[-1]
        sentence_count = len(re.findall(r'[.!?]"?', last_chunk))
        if sentence_count < 2:
            context_list[-2] = context_list[-2].rstrip() + ' ' + last_chunk.lstrip()
            context_list.pop()
    return context_list

# ⏱️ 增加带超时处理的函数
def run_with_timeout(func, args=(), timeout=60):
    def wrapper(queue, *args):
        try:
            result = func(*args)
            queue.put(result)
        except Exception as e:
            queue.put([])

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(queue, *args))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return []  # 超时返回空列表
    return queue.get()

def process_input(data):
    formatted_data = []
    for idx, item in enumerate(tqdm(data, desc="Processing questions")):
        question_id = f"question_866_{idx}"
        question = item.get("question", "")
        query = item.get("query", "")
        results = []

        for doc_idx, doc in enumerate(item.get("results", [])):
            try:
                print(f"➡️ 正在处理第 {idx} 条 question，第 {doc_idx} 个 document")
                title = doc.get("title", "")
                url = doc.get("url", "")
                context = doc.get("context", "")
                snippet = doc.get("snippet", "")
                if not context:
                    context = snippet

                # ⏱️ 使用带超时的函数处理 context_list
                context_list = run_with_timeout(split_context_by_words_and_sentences, args=(context, 300), timeout=30) if context else []

                if not context_list:
                    context_list = [snippet]

                results.append({
                    "title": title,
                    "url": url,
                    "context": context,
                    "context_list": context_list,
                    "question_id": question_id
                })
            except Exception as e:
                print(f"❌ 第 {idx} 条的第 {doc_idx} 个 document 出错：{e}")

        if results:
            formatted_data.append({
                "question_id": question_id,
                "question": question,
                "query": query,
                "results": results
            })
    return formatted_data

def main(input_file, output_file):
    # 从输入文件读取数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理数据
    formatted_data = process_input(data)

    # 将处理后的数据输出到文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = ""
    output_file = ""

    main(input_file, output_file)
