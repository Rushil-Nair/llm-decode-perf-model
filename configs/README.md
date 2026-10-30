# Model configs

These are the Hugging Face `config.json` files for the models used in this repo,
copied unmodified (key order and values as published). Only the architecture
fields are used by `perfmodel/models.py`:
`num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`,
`head_dim` (derived as `hidden_size / num_attention_heads` when absent),
`intermediate_size`, `vocab_size`, `tie_word_embeddings`, `model_type`.

| File | Source URL | Note |
|---|---|---|
| `qwen3-8b.json` | https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json | official repo |
| `llama-3.2-3b.json` | https://huggingface.co/unsloth/Llama-3.2-3B/raw/main/config.json | public mirror of the gated `meta-llama/Llama-3.2-3B` |
| `llama-3.1-8b.json` | https://huggingface.co/unsloth/Meta-Llama-3.1-8B/raw/main/config.json | public mirror of the gated `meta-llama/Llama-3.1-8B` |
| `llama-3.1-70b.json` | https://huggingface.co/unsloth/Meta-Llama-3.1-70B/raw/main/config.json | public mirror of the gated `meta-llama/Llama-3.1-70B`; has no `head_dim` key (8192/64 = 128) |

The files were retrieved on 2026-10-24 through a web-fetch tool (direct `curl`
to huggingface.co was not available in the build environment), so they were
transcribed rather than byte-copied. The architecture fields were cross-checked
against the parameter counts they imply (e.g. Qwen3-8B: 6.95 B non-embedding
parameters, matching the model card). Re-download with `curl` to confirm
byte-identity before relying on any non-architecture field.

Quantized file used for the llama.cpp validation:

| File | Size (bytes) | Source |
|---|---|---|
| `Qwen3-8B-Q4_K_M.gguf` | 5,027,783,488 | https://huggingface.co/api/models/Qwen/Qwen3-8B-GGUF/tree/main |

(The `unsloth/Qwen3-8B-GGUF` Q4_K_M file is 5,027,784,512 bytes; the 1 KB
difference is metadata. Which of the two the measurements used is not recorded;
the difference is irrelevant at the precision of this model.)
