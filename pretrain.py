import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x    

@dataclass
class GPTconfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trnsformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

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
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.trnsformer.wpe(pos)
        tok_emb = self.trnsformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.trnsformer.h:
            x = block(x)
        x = self.trnsformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


@classmethod
def from_pretrained(cls, model_type):
    assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], "Invalid model type"
    from transformers import GPT2LMHeadModel
    print("loading weight from pretrained gpt: %s" % model_type)

    config_args = {
        "gpt2": dict(n_embd=768, n_layer=12, n_head=12),
        "gpt2-medium": dict(n_embd=1024, n_layer=24, n_head=16),    
        "gpt2-large": dict(n_embd=1280, n_layer=36, n_head=20),
        "gpt2-xl": dict(n_embd=1600, n_layer=48, n_head=25),
    }[model_type]
    config_args['block_size'] = 1024
    config_args['vocab_size'] = 50257
    config = GPTconfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endwith('.attn.bias') ]

    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked.bias')]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    assert len(sd_keys_hf) == len(sd_keys), f"mismached keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endwith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])
    
    return model

# ---------------------------------------------------------------------------------------------------

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        # init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = ({len(self.tokens)} // (B * T)) batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # get the next batch
        x = (self.tokens[self.current_position:self.current_position + B*T]).view(B, T)  # inputs
        y = (self.tokens[self.current_position + 1:self.current_position + B*T + 1]).view(B, T)  # targets
        self.current_position += B * T

        # if loading the next batch would be out of bounds, reset
        if self.current_position + B * T >= len(self.tokens):
            self.current_position = 0

        return x, y

#---------------------------------------------------------------------------------------------------

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
train_loader = DataLoaderLite(B=4, T=32)
# num_return_sequences = 5
# max_length = 30

# model = GPT.from_pretrained('gpt2')
# model.eval()
# model.to('cuda')

# import tiktoken
# enc = tiktoken.get_encoding("gpt2")
# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

model = GPT(GPTconfig())
model.to(device)

# optimizer!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    import code; code.interact(local=locals())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    # torch.cuda.synchronize()
    # t1 = time.time()
    # dt = t1-t0
    # token_processed = train_loader.B * train_loader.T
    # tokens_per_sec = token_processed / dt
    print(f"step {i}, loss: {loss.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}")
    
import sys; sys.exit(0)


# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = logits[:,-1,:]
#         probs = F.softmax(logits, dim=-1)
#         topk_probs, topk_indices = torch.topk(probs , 50, dim=-1)
#         ix = torch.multinomial(topk_probs, num_samples=1)
#         xcoi = torch.gather(topk_indices, dim=-1, index=ix)
#         x = torch.cat([x, xcoi], dim=1)

#     for i in range(num_return_sequences):
#         tokens = x[i, :max_length].tolist()
#         decoded = enc.decode(tokens)
#         print(">", decoded)

