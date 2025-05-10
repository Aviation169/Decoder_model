import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
from torch.cuda.amp import autocast

@dataclass
class ModelArgs:
    dim: int = 2048  # Reduced for smaller model
    n_layers: int = 24  # Reduced
    n_heads: int = 16  # Reduced
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: int = 5504  # Adjusted for SwiGLU
    max_seq_len: int = 512  # Reduced
    norm_eps: float = 1e-5
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # Ensure FP16 compatibility by casting intermediate results
        x = x.float()  # Upcast to float32 for stability
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight).type_as(self.weight)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[3])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((1, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((1, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.cache_k = self.cache_k.to(xq.device)
        self.cache_v = self.cache_v.to(xq.device)
        self.cache_k[:bsz, start_pos:start_pos+seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos+seqlen] = xv

        keys = self.cache_k[:bsz, :start_pos+seqlen]
        values = self.cache_v[:bsz, :start_pos+seqlen]

        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=2)
            values = values.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = args.hidden_dim
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class TransformerColumn(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)

    def forward(self, tokens: torch.Tensor, start_pos: int, lateral_inputs: Optional[List[torch.Tensor]] = None):
        bsz, seqlen = tokens.shape
        with autocast():
            h = self.tok_embeddings(tokens)
            freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen].to(h.device)

            mask = None
            if seqlen > 1:
                mask = torch.triu(torch.full((seqlen, seqlen + start_pos), float("-inf"), device=tokens.device), diagonal=1 + start_pos)
                mask = mask[None, None, :, :]

            layer_outputs = []
            for i, layer in enumerate(self.layers):
                if lateral_inputs and i < len(lateral_inputs):
                    h = h + lateral_inputs[i]
                h = layer(h, start_pos, freqs_cis, mask)
                layer_outputs.append(h)
            h = self.norm(h)
            logits = self.output(h)
        return layer_outputs, logits

class ProgressiveLLaMA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.columns: List[TransformerColumn] = nn.ModuleList()
        self.lateral_projections: List[nn.ModuleList] = nn.ModuleList()
        self.add_column()  # Single column

    def add_column(self):
        new_column = TransformerColumn(self.args)
        self.columns.append(new_column)
        
        lateral_projs = nn.ModuleList()
        for _ in range(len(self.columns) - 1):
            projs = nn.ModuleList([
                nn.Linear(self.args.dim, self.args.dim, bias=False)
                for _ in range(self.args.n_layers)
            ])
            lateral_projs.append(projs)
        self.lateral_projections.append(lateral_projs)

        for col in self.columns[:-1]:
            for param in col.parameters():
                param.requires_grad = False

    def forward(self, tokens: torch.Tensor, task_id: int, start_pos: int = 0):
        if task_id >= len(self.columns):
            raise ValueError(f"Task ID {task_id} exceeds number of columns {len(self.columns)}")
        for column in self.columns:
            for layer in column.layers:
                layer.attention.cache_k.zero_()
                layer.attention.cache_v.zero_()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        hidden_states = []
        for i in range(task_id + 1):
            lateral_inputs = []
            if i > 0:
                for j in range(i):
                    prev_hidden = hidden_states[j]
                    projs = self.lateral_projections[i][j]
                    with autocast():
                        lateral = [projs[k](prev_hidden[k]) for k in range(self.args.n_layers)]
                    lateral_inputs.extend(lateral)
            
            layer_outputs, logits = self.columns[i](tokens, start_pos, lateral_inputs)
            hidden_states.append(layer_outputs)

        return logits

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Total parameters (in millions): {total_params / 1_000_000:.2f}M")
        if total_params >= 1_000_000_000:
            print(f"Total parameters (in billions): {total_params / 1_000_000_000:.2f}B")
        return total_params

if __name__ == "__main__":
    args = ModelArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = ProgressiveLLaMA(args).to(device).half()
        model.count_parameters()

        tokens = torch.randint(0, args.vocab_size, (1, 64)).to(device)  # LongTensor
        logits = model(tokens, task_id=0)
        print(f"Output logits shape: {logits.shape}")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        if "out of memory" in str(e).lower():
            print("Out of GPU memory. Try reducing model size, sequence length, or using CPU.")
        elif "scalar type" in str(e).lower():
            print("Type mismatch. Check tensor dtypes and ensure mixed precision is handled correctly.")
    except Exception as e:
        print(f"Unexpected error: {e}")
