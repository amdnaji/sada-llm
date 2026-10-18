# SadaLLM (صدى - Echo): An Educational Micro-LLM from Scratch (10.6M Parameters)
### *A Transparent Decoder-Only Transformer for Teaching LLM Foundations*

**SadaLLM** (Arabic for *Echo* / صدى) is an **educational, open-source Micro-Language Model** built from scratch to teach students the mathematical, architectural, and empirical foundations of Generative Pre-trained Transformers (Decoder-only LLMs).

Rather than relying on black-box libraries (`nn.Transformer`, HuggingFace), every matrix multiplication, attention score, normalization layer, and residual connection is implemented with explicit PyTorch tensor mechanics.

> **Pedagogical Framing:** SadaLLM is **not** a specialized commercial legal model. It is an **educational micro-LLM**, using an excerpt of the **Saudi Labor Law (نظام العمل السعودي)** as a working **case study** to demonstrate how Transformers learn structured syntax, domain grammar, and autoregressive next-token prediction. The architecture is general-purpose and can be trained on any text corpus (stories, code, medical notes, or arithmetic) simply by swapping the training file.

---

## 📚 Foundational Research Papers (Academic Syllabus)

Every component in SadaLLM is directly grounded in landmark literature:

| Architectural Component | Seminal Research Paper | Theoretical & Empirical Insight |
| :--- | :--- | :--- |
| **0. Tokenization Trade-offs** | [Sennrich et al. (2016) - BPE](https://arxiv.org/abs/1508.07909) | For teaching clarity, SadaLLM uses an atomic Character-level vocabulary ($V \approx 75$) to eliminate sub-character byte stuttering; production LLMs use Subword BPE to compress common words into single tokens. |
| **1. Dense Vector Embeddings** | [Bengio et al. (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) & [Mikolov et al. (2013)](https://arxiv.org/abs/1310.4546) | Overcomes the curse of dimensionality by projecting discrete tokens into a continuous feature space $\mathbb{R}^d$. |
| **2. Learned Positional Embeddings** | [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) & [Gehring et al. (2017)](https://arxiv.org/abs/1705.03122) | Injects sequence order to break the permutation-invariance of the self-attention mechanism. |
| **3. Causal Multi-Head Self-Attention** | [Bahdanau (2014)](https://arxiv.org/abs/1409.0473) $\to$ [Luong (2015)](https://arxiv.org/abs/1508.04025) $\to$ [Vaswani (2017)](https://arxiv.org/abs/1706.03762) | Scaled dot-product attention with an upper-triangular causal mask ($M_{ij} = -\infty$) ensuring tokens only attend to past context. |
| **4. Residual Connections & Pre-LN** | [He et al. (2016) - ResNet](https://arxiv.org/abs/1512.03385) & [Ba et al. (2016) - LayerNorm](https://arxiv.org/abs/1607.06450) | Pre-Layer Normalization and identity shortcut streams create an uninterrupted gradient highway back to early layers ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)). |
| **5. Feed-Forward MLP & GELU** | [Hendrycks & Gimpel (2016)](https://arxiv.org/abs/1606.08415) & [Geva et al. (2021)](https://arxiv.org/abs/2012.14913) | Gaussian Error Linear Units smooth activations; the FFN acts as an associative key-value memory layer storing factual patterns. |
| **6. Weight Tying** | [Press & Wolf (2017)](https://arxiv.org/abs/1608.05859) | Shares parameters between the input token embedding and the output language modeling head ($W_{\text{head}} = W_e^\top$), saving memory and regularizing representations. |
| **7. Loss & Sampling Mechanics** | [Shannon (1948)](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) & [Radford et al. (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Autoregressive cross-entropy loss, paired with Temperature scaling and Top-$K$ truncation ([Fan et al., 2018](https://arxiv.org/abs/1805.04833)) to prevent degenerate text. |

---

## 🧮 Hardware & Memory Analysis (12 GB RAM System)

Students often worry that training a 10M-parameter model will exceed their laptop's memory. In reality, SadaLLM uses less than **4%** of a 12 GB RAM machine:

| Component | Precision & Sizing | Working RAM Footprint |
| :--- | :--- | :--- |
| **Model Parameters (Weights)** | $10{,}600{,}000 \times 4\text{ bytes (Float32)}$ | **~42.4 MB** |
| **Gradients ($\nabla W$)** | $10{,}600{,}000 \times 4\text{ bytes}$ | **~42.4 MB** |
| **Adam Optimizer States ($m_t, v_t$)** | $2 \times 10{,}600{,}000 \times 4\text{ bytes}$ | **~84.8 MB** |
| **Forward Activations (Batch=16, Context=256)** | Cached tensors retained for backward pass | **~200–250 MB** |
| **Total Active Training Memory** | | **$\approx$ 420 MB** |

> **CPU Execution:** SadaLLM runs comfortably on any modern multi-core CPU (Intel/AMD) without needing a dedicated GPU.

---

## ⚙️ Model Configurations

SadaLLM includes modular presets in the `ModelConfig` dataclass:

| Preset | Layers ($L$) | Hidden Dim ($d$) | Heads ($h$) | Context ($T$) | MLP Dim ($4d$) | Total Parameters | CPU Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `CONFIG_5M` | 6 | 256 | 4 | 256 | 1024 | **~5.2M** | ~2–3 minutes on CPU |
| `CONFIG_10M` *(Default)* | 6 | 384 | 6 | 256 | 1536 | **~10.6M** | ~4–5 minutes on CPU |
| `CONFIG_20M` | 8 | 512 | 8 | 256 | 2048 | **~20.5M** | ~10–12 minutes on CPU |

---

## 🗂️ Project Structure

```
hamsah/
├── micro_llm.py              # Main source: SadaLLM architecture, tokenizer & CPU training loop
├── saudi_labor_law_data.txt  # Case study corpus: Selected articles and Q&A pairs
├── web_chat.py               # Lightweight local browser interface with native Arabic RTL support
├── test_model.py             # Interactive terminal testing script with preset questions
├── micro_llm_lecture.ipynb   # Interactive Jupyter notebook for Google Colab with lecture notes
└── README.md                 # Complete documentation and academic syllabus
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Install the lightweight CPU build of PyTorch (~180 MB):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Train the Model on CPU
Run the training script directly from your terminal:
```bash
python micro_llm.py
```
* **Execution:** Trains for 450–500 steps (~4–5 minutes on CPU).
* **Observation:** Watch cross-entropy loss drop from initial entropy baseline ($\ln(75) \approx 4.31$) down to $< 0.3$.
* **Checkpoint:** Automatically saves trained weights to `saudi_labor_law_model.pt`.

### 3. Interactive Web Interface (Recommended for Teaching)
Launch the built-in browser UI:
```bash
python web_chat.py
```
* Automatically opens `http://127.0.0.1:8000` in your default browser.
* Features native **Right-to-Left (RTL)** Arabic rendering, preset question buttons, and interactive Temperature & Top-$K$ sliders.

### 4. Interactive Terminal Testing
Test the model directly inside your command line:
```bash
python test_model.py
```
* Select predefined questions by entering numbers `1` through `7`, or type custom prompts.
* *(Tip for Windows: run `chcp 65001` first to enable UTF-8 support in Command Prompt).*

### 5. Running on Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Upload `micro_llm_lecture.ipynb`.
3. Step through each cell to explore theory, view loss convergence plots, and run generation tests interactively.
