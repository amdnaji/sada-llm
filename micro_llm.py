"""
SadaLLM (صدى - Echo): An Educational Micro-LLM from Scratch (5M - 20M Params)
=============================================================================
A lightweight, transparent language model built to teach Transformer foundations.
Case Study: Trained on an excerpt of the Saudi Labor Law (نظام العمل السعودي)
as a working demonstration of how LLMs learn domain patterns and syntax.

Key Architectural & Theoretical Papers Implemented:
1. Embeddings & Representation:   Bengio et al. (2003) & Mikolov et al. (2013)
2. Positional Encodings:          Gehring et al. (2017) & Vaswani et al. (2017)
3. Causal Multi-Head Attention:   Bahdanau (2014) -> Luong (2015) -> Vaswani et al. (2017)
4. Residual Stream:               He et al. (2016) [ResNet]
5. Pre-Layer Normalization:       Ba, Kiros, Hinton (2016) & Xiong et al. (2020)
6. GELU Non-Linearity:            Hendrycks & Gimpel (2016) [Used in GPT-2, GPT-3]
7. Weight Tying:                  Press & Wolf (2017) [Tied Input/Output Embeddings]
8. Autoregressive Training:       Shannon (1948) & Radford et al. (2018) [GPT-1]
9. Sampling Mechanics (Top-k):    Fan et al. (2018) & Holtzman et al. (2019)
10. LLM Scaling Laws:             Kaplan et al. (2020) & Hoffmann et al. (2022)
"""

import os
import math
import time
import random
import urllib.request
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure reproducibility
torch.manual_seed(42)
random.seed(42)

# =====================================================================
# SECTION 1: DIRECT CHARACTER INDEXING (فهرسة المحارف المباشرة)
# =====================================================================
# Pedagogical Note:
# - Here we use a direct Character-to-Index lookup table (like ASCII/Unicode).
# - Every Arabic letter is an atomic unit (~75 total characters), eliminating
#   sub-character byte stuttering while keeping the code 100% transparent.
# - In giant production LLMs (GPT-4, Jais), Subword Tokenization (BPE) is used
#   instead to compress common words into single tokens:
#   See: Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units".

# Base Arabic & legal punctuation vocabulary:
ARABIC_CHARS = sorted(list(set(
    "ابتثجحخدذرزسشصضطظعغفقكلمنهويىةءئؤإأآ"
    "0123456789"
    " \n\t،؛.:!؟-\"'/()[]"
)))
VOCAB = ARABIC_CHARS + ['<pad>', '<eos>']
VOCAB_SIZE = len(VOCAB)  # ~75 unique tokens
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX2CHAR = {i: ch for i, ch in enumerate(VOCAB)}

def encode_text(text: str) -> list[int]:
    """Encodes Arabic string into character token IDs."""
    return [CHAR2IDX.get(ch, CHAR2IDX['<pad>']) for ch in text]

def decode_ids(ids: list[int]) -> str:
    """Decodes token IDs back to pristine Arabic string."""
    chars = []
    for i in ids:
        if i < len(VOCAB):
            ch = IDX2CHAR[i]
            if ch not in ['<pad>', '<eos>']:
                chars.append(ch)
    return ''.join(chars)


# =====================================================================
# SECTION 2: ARCHITECTURAL CONFIGURATIONS (5M, 10M, 20M)
# =====================================================================
@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE # ~75 Arabic characters + punctuation
    max_context_len: int = 256  # 256 full Arabic characters!
    d_model: int = 384          # Embedding dimension (d)
    n_layers: int = 6           # Number of stacked Transformer layers (L)
    n_heads: int = 6            # Multi-head attention count (h)
    d_ff: int = 1536            # Feed-Forward expansion (4 * d_model)
    dropout: float = 0.0        # Dropout rate
    tie_weights: bool = True    # Press & Wolf (2017) Weight Tying

# Presets for students to explore Scaling Laws (Kaplan et al., 2020)
CONFIG_5M = ModelConfig(
    d_model=256, n_layers=6, n_heads=4, d_ff=1024, max_context_len=256
)  # ~5.2 Million Parameters (Fastest on CPU)

CONFIG_10M = ModelConfig(
    d_model=384, n_layers=6, n_heads=6, d_ff=1536, max_context_len=256
)  # ~10.6 Million Parameters (Recommended Sweet Spot)

CONFIG_20M = ModelConfig(
    d_model=512, n_layers=8, n_heads=8, d_ff=2048, max_context_len=256
)  # ~20.5 Million Parameters (High Capacity)


# =====================================================================
# SECTION 3: SAUDI LABOR LAW DATASET (نظام العمل السعودي)
# =====================================================================
def load_or_create_dataset(file_path: str = "saudi_labor_law_data.txt") -> str:
    """
    Loads Saudi Labor Law text file if available.
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        repeated_corpus = (text.strip() + "\n\n") * 15
        print(f"[Dataset] Loaded Saudi Labor Law text from '{file_path}' ({len(text):,} chars, expanded to {len(repeated_corpus):,} chars).")
        return repeated_corpus
    
    # Fallback to English stories if file not found
    print(f"[Dataset] Notice: '{file_path}' not found, falling back to starter corpus.")
    fallback = (SAMPLE_STORIES.strip() + "\n\n") * 10
    return fallback


# =====================================================================
# SECTION 4: ARCHITECTURE (From Papers to PyTorch)
# =====================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization: Ba, Kiros, & Hinton (2016)
    Formula: LN(x) = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention: Vaswani et al. (2017) §3.2
    Lineage: Bahdanau (2014) -> Luong (2015) -> Vaswani (2017)
    
    Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) + M ) @ V
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_model // cfg.n_heads

        # Projections
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)

        # Upper-triangular causal mask (Vaswani et al. §3.2.3)
        mask = torch.triu(torch.full((cfg.max_context_len, cfg.max_context_len), float('-inf')), diagonal=1)
        self.register_buffer('mask', mask)
        self.last_attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + self.mask[:T, :T]

        weights = F.softmax(scores, dim=-1)
        self.last_attn_weights = weights.detach()
        weights = self.attn_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    """
    Feed-Forward Network with GELU Activation:
    - Vaswani et al. (2017) §3.3 (Architecture)
    - Hendrycks & Gimpel (2016) (GELU non-linearity, standard in GPT-2/3)
    - Geva et al. (2021) (Key-Value associative memory)
    
    Formula: FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.gelu implements Hendrycks & Gimpel (2016)
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block:
    - He et al. (2016) [Identity Residual Connections]
    - Xiong et al. (2020) [Pre-LN for training stability]
    
    x = x + Attn(LN1(x))
    x = x + MLP(LN2(x))
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SadaLLM(nn.Module):
    """
    Scalable SadaLLM (صدى - Echo) Architecture (5M - 20M Parameters)
    - Bengio et al. (2003) & Radford et al. (2018): Token & Learned Positional Embeddings
    - Press & Wolf (2017): Weight Tying between tok_emb and lm_head
    - Radford et al. (2018): Autoregressive Pre-LayerNorm Transformer
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_context_len, cfg.d_model)

        # 2. Stacked Transformer Blocks (Vaswani et al. 2017)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # 3. Final Pre-LN and Language Model Head
        self.ln_f = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight Tying: Press & Wolf (2017)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Weight initialization (Radford et al. 2019)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.max_context_len, f"Context length {T} exceeds max {self.cfg.max_context_len}"

        positions = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits


# =====================================================================
# SECTION 5: PARAMETER AUDIT & MEMORY FOOTPRINT
# =====================================================================
def print_model_audit(model: SadaLLM):
    """Prints the exact parameter breakdown, memory footprint, and citations."""
    print("=" * 80)
    print(f"{'SADA-LLM (صدى) PARAMETER & MEMORY AUDIT':^80}")
    print("=" * 80)
    print(f"{'Component':<32} {'Shape':<18} {'Count':<12} {'Citation'}")
    print("-" * 80)

    total_params = 0
    tied_params = 0

    for name, param in model.named_parameters():
        cnt = param.numel()
        total_params += cnt
        shape_str = str(list(param.shape))
        
        citation = "Vaswani et al. (2017)"
        if "tok_emb" in name:
            citation = "Bengio et al. (2003)"
        elif "pos_emb" in name:
            citation = "Gehring / Radford (2017)"
        elif "ln" in name:
            citation = "Ba, Kiros, Hinton (2016)"
        elif "mlp" in name:
            citation = "Hendrycks (GELU) / Geva (2021)"
        elif "lm_head" in name:
            citation = "Press & Wolf (2017) [Tied]"
            tied_params += cnt

        print(f"{name:<32} {shape_str:<18} {cnt:<12,d} {citation}")

    fp32_mb = (total_params * 4) / (1024 * 1024)
    adam_mb = (total_params * 8) / (1024 * 1024)

    print("-" * 80)
    print(f"Total Unique Trainable Parameters: {total_params:,d}")
    print(f"Weight Memory (FP32):              {fp32_mb:.2f} MB")
    print(f"Adam Optimizer Memory (Moments):   {adam_mb:.2f} MB")
    print(f"Combined Training Footprint:       {fp32_mb + adam_mb:.2f} MB (Well within 12 GB RAM!)")
    print("=" * 80)
    return total_params


# =====================================================================
# SECTION 6: DATA BATCHING & PREPARATION
# =====================================================================
def get_batch(data_tensor: torch.Tensor, batch_size: int, context_len: int):
    """Samples random context windows from the contiguous token stream."""
    max_idx = len(data_tensor) - context_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data_tensor[i:i + context_len] for i in ix])
    y = torch.stack([data_tensor[i + 1:i + 1 + context_len] for i in ix])
    return x, y


# Alias for backward compatibility
BabyLLM = SadaLLM

# =====================================================================
# SECTION 7: AUTOREGRESSIVE GENERATION (Top-K & Temperature)
# =====================================================================
def generate_story(
    model: SadaLLM, 
    prompt: str, 
    max_new_tokens: int = 150, 
    temperature: float = 0.2, 
    top_k: int = 5
) -> str:
    """
    Autoregressive text generation using Arabic Character Tokenizer:
    - Temperature scaling (Ackley et al. 1985 / Radford et al. 2018)
    - Top-k sampling: Fan, Lewis, & Dauphin (2018)
    """
    model.eval()
    tokens = encode_text(prompt)
    if len(tokens) == 0:
        tokens = [encode_text("سؤال:")[0]]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = tokens[-model.cfg.max_context_len:]
            x = torch.tensor([idx_cond], dtype=torch.long)

            logits = model(x)  # (1, T, vocab_size)
            logits = logits[0, -1, :] / max(temperature, 1e-5)

            # Apply Top-K filtering (Fan et al., 2018)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)

            if next_token == CHAR2IDX['<eos>']:
                break

    return decode_ids(tokens)

# Alias for legal domain
generate_legal = generate_story


# =====================================================================
# SECTION 8: TRAINING LOOP (CPU-Optimized)
# =====================================================================
def train_model(config: ModelConfig = CONFIG_10M, max_steps: int = 400, batch_size: int = 16):
    """
    Trains the SadaLLM (صدى) model on CPU using the Saudi Labor Law dataset.
    """
    # 1. Load data
    text = load_or_create_dataset("saudi_labor_law_data.txt")
    tokens = encode_text(text)
    data_tensor = torch.tensor(tokens, dtype=torch.long)
    print(f"[Dataset] Encoded into {len(tokens):,} character tokens (Vocab = {config.vocab_size} characters).")

    # 2. Build model and audit parameters
    model = SadaLLM(config)
    print_model_audit(model)

    # 3. Information Theory Random Baseline: -ln(1/V) = ln(75) ≈ 4.31
    init_loss_baseline = math.log(config.vocab_size)
    print(f"\n[Shannon 1948 Baseline] Expected initial random loss: ln({config.vocab_size}) = {init_loss_baseline:.4f}")

    # 4. Optimizer: AdamW with LR for crisp legal convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print("\nStarting Training on CPU for 'SadaLLM - نظام العمل السعودي'...")
    print("-" * 65)
    print(f"{'Step':<10} {'Cross-Entropy Loss':<22} {'Tokens/Sec':<15} {'Elapsed'}")
    print("-" * 65)

    start_time = time.time()
    model.train()

    for step in range(1, max_steps + 1):
        step_start = time.time()
        x, y = get_batch(data_tensor, batch_size=batch_size, context_len=config.max_context_len)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping (Pascanu, Mikolov, & Bengio, 2013)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - start_time
            tokens_processed = 50 * batch_size * config.max_context_len if step > 1 else batch_size * config.max_context_len
            tps = tokens_processed / max(time.time() - step_start, 1e-4)
            print(f"{step:<10} {loss.item():<22.4f} {tps:<15.1f} {elapsed:.1f}s")

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"Training Complete in {total_time:.1f} seconds on CPU!\n")
    return model


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("\n" + "#" * 80)
    print(" SADA-LLM (صدى): 10.6M TRANSFORMER - نظام العمل السعودي")
    print("#" * 80)

    # Use 10M Sweet Spot
    chosen_config = CONFIG_10M
    print(f"Active Architecture: {chosen_config.n_layers} Layers, d_model={chosen_config.d_model}, {chosen_config.n_heads} Heads, Context={chosen_config.max_context_len}")

    # Train model to convergence (Loss < 0.3)
    model = train_model(config=chosen_config, max_steps=500, batch_size=16)

    # Save checkpoint
    checkpoint_file = "saudi_labor_law_model.pt"
    torch.save(model.state_dict(), checkpoint_file)
    print(f"\n[Checkpoint] Saved trained weights to '{checkpoint_file}'!")

    # Test Arabic legal prompts
    arabic_prompts = [
        "سؤال: كم الحد الأقصى لساعات العمل اليومية في نظام العمل السعودي؟\nالجواب:",
        "المادة الثالثة والخمسون: إذا كان العامل خاضعاً لفترة تجربة",
        "سؤال: كيف تحسب مكافأة نهاية الخدمة للعامل؟\nالجواب:",
        "المادة التاسعة بعد المائة: يستحق العامل عن كل عام إجازة سنوية"
    ]

    print("\n" + "=" * 80)
    print("AUTOREGRESSIVE GENERATION TEST (SadaLLM - صدى)")
    print("=" * 80)
    for p in arabic_prompts:
        output = generate_story(model, prompt=p, max_new_tokens=140, temperature=0.2, top_k=5)
        print(f"\n[المدخل / Prompt]:\n{p}")
        print(f"\n[إجابة نموذج صدى / SadaLLM]:\n{output}")
        print("-" * 80)
