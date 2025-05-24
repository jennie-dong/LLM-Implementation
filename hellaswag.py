import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
# 数据下载路径
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

# 下载函数
def download_file(url: str, fname: str, chunk_size=1024):
    """辅助函数：从指定 URL 下载文件"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

# 定义 HellaSwag 各个 split 的下载地址
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# 使用 GPT-2 的 tokenizer
enc = tiktoken.get_encoding("gpt2")

# 下载某个 split（train/val/test）的 HellaSwag 数据
def download(split):
    """下载指定 split 的数据到本地缓存目录"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"正在下载 {data_url} 到 {data_filename}...")
        download_file(data_url, data_filename)

# 将单个样本转换为 token、mask 和标签
def render_example(example):
    """
    给定一个样本（字典），返回三个张量：
    - tokens：上下文 + 4 个结尾的 token，形状 (4, N)
    - mask：标记哪些 token 属于结尾部分（用于计算 loss）
    - label：正确结尾的索引（0,1,2,3）
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # 编码上下文部分
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []

    # 对每个结尾进行编码，并生成 mask（前面是 0，后面是 1）
    for end in endings:
        end_tokens = enc.encode(" " + end)  # 结尾前加空格符合 GPT-2 的预期
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # 统一 padding 到最大长度
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    """按行读取指定 split 的 HellaSwag 样本"""
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

# 主评估函数
@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision('high')  # 开启 TF32 加速（在支持的 GPU 上）

    # 加载 GPT2 模型
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model) # 可选加速

    num_correct_norm = 0  # 标准化准确率
    num_correct = 0       # 非标准化准确率
    num_total = 0         # 样本总数

    # 遍历验证集样本
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # 模型前向传播
        logits = model(tokens).logits  

        # 计算自回归损失
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # 只对结尾区域计算平均损失
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # 累计统计
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        if num_total < 10:
            print("---")
            print(f"上下文:\n {example['ctx']}")
            print(f"候选结尾:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"预测：{pred_norm}, 正确答案：{label}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="使用的模型类型")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="使用的设备（cuda 或 cpu）")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
