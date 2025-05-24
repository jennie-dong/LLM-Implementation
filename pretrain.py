import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------

# 注意力模块
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 对所有注意力头进行 q、k、v 的线性变换
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 下三角掩码
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # B: 批大小, T: 序列长度, C: 嵌入维度
        B, T, C = x.size()  
        # 计算 q、k、v，形状为 (B, T, 3*C)，然后分离
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # 拼接所有 head 的输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 输出投影
        y = self.c_proj(y)
        return y

# MLP
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Transformer 块
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  
        x = x + self.mlp(self.ln_2(x))   
        return x

# 配置数据类
@dataclass
class GPTConfig:
    block_size: int = 1024          
    vocab_size: int = 50257         
    n_layer: int = 12               
    n_head: int = 12                
    n_embd: int = 768               

# GPT 模型主体
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 构建 GPT 模块
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),      
            wpe = nn.Embedding(config.block_size, config.n_embd),      
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  
            ln_f = nn.LayerNorm(config.n_embd),                        
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  

        # 权重共享
        self.transformer.wte.weight = self.lm_head.weight

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"序列长度 {T} 超过最大块大小 {self.config.block_size}"

        # 获取位置和 token 嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 
        pos_emb = self.transformer.wpe(pos)        
        tok_emb = self.transformer.wte(idx)         
        x = tok_emb + pos_emb                      

        # 通过 Transformer 块
        for block in self.transformer.h:
            x = block(x)

        # LayerNorm 和输出层
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) 

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """从 Hugging Face 加载预训练的 GPT-2 权重"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("从预训练模型加载权重：%s" % model_type)

        # 根据模型类型设置参数
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), 
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), 
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]  

        # 加载 transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 过滤掉非参数的权重项
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # 需要转置的权重项
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"权重键数量不一致：{len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------
import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B  
        self.T = T  

        # 读取文本，并编码为 token 存入内存
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')  
        tokens = enc.encode(text)  
        self.tokens = torch.tensor(tokens) 
        print(f"loaded {len(self.tokens)} tokens") 
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches") 

        # 初始化当前位置
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        # 构造输入 x和目标 y
        x = (buf[:-1]).view(B, T) 
        y = (buf[1:]).view(B, T)  

        self.current_position += B * T

        # 如果下一批超出范围，就重置回开头
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y  

# -----------------------------------------------------------------------------
# 自动检测设备
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=16, T=1024)

torch.set_float32_matmul_precision('high')

# 初始化模型，可以得出logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    # 计算时间
    torch.cuda.synchronize() 
    t1 = time.time()
    dt = (t1 - t0)*1000 
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)

# 前缀token
# 为了生成多个结果，把原始输入复制 5 份，形状变成 (5, 8)
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) 
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) 
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    # 前向传播
    with torch.no_grad():
        logits = model(x)  
        logits = logits[:, -1, :] 
        # 把logits转换为概率分布 
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1) 
        xcol = torch.gather(topk_indices, -1, ix) 
        x = torch.cat((x, xcol), dim=1)

# 打印生成的文本
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
