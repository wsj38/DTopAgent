import os
import torch
import json
import transformers
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer

def process_questions(model_id, input_path, output_path):
    # 载入模型
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    results = []
    # 读取输入 JSON 文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for i, item in enumerate(data, 1):
            print(f"Processing question {i}...")
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
            
            # 发送到模型
            messages = [
                {"role": "system", "content": "You are an expert retrieval assistant specializing in determining the minimum number of documents needed to accurately answer a question."},
                {"role": "user", "content": instruction},
            ]
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = pipeline(
                        messages,
                        max_new_tokens=256,
                        eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                    )
            
            # 获取答案
            answer = outputs[0]["generated_text"][-1]["content"]
            print(f"K: {answer}")
            
            # 保存结果
            results.append({
                "question": question,
                "meta": meta_list,
                "ground_truth": ground_truth,
                "k": answer
            })
    
    # 写入输出 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    print(f"\n所有结果已保存到 {output_path}")


# 调用函数，传入参数
if __name__ == "__main__":
    model_id = "模型路径"
    input_path = ""
    output_path = ""

    process_questions(model_id, input_path, output_path)
