# ============================================================================
# Cell 1 — SETUP, LOGINS & PATHS (edit only these top variables if needed)
# ----------------------------------------------------------------------------
# This cell:
#  - installs missing packages (safe on Kaggle/Colab)
#  - sets the dataset/model paths discovered in your uploaded notebook
#  - loads HF token from env var HF_TOKEN or from /kaggle/input/hf_token/token.txt
#  - prints helpful diagnostics
# ----------------------------------------------------------------------------
MODEL_NAME_OR_PATH     = "unsloth/meta-llama-3.1-8b-bnb-4bit"   # from your notebook
TRAIN_FILE             = "/kaggle/input/training-data-chatml/training_data_chatml.jsonl"
TRAIN_FILE_FORMAT      = "jsonl"   # chat-ml jsonl from your notebook
OUTPUT_DIR             = "./output-llm-finetune"
HF_TOKEN_ENV_VAR_NAME  = "HF_TOKEN"
BASE_ADAPTER_DIR       = "nifty50_llama_lora"   # adapter folder used in your notebook
GGUF_OUT_DIR           = "nifty50_analyst_v1"   # optional saved quantized model dir from notebook
USE_LORA               = True
MAX_LENGTH             = 1024
BATCH_SIZE             = 4
EPOCHS                 = 1
LEARNING_RATE          = 2e-5
SEED                   = 42

# ---------- Auto-install minimal deps ----------
import os, sys, subprocess
def pip_install(packages):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

# Try required libs, install if missing
try:
    import torch, transformers, datasets
except Exception:
    pip_install(["transformers>=4.35.0", "datasets", "evaluate", "accelerate", "safetensors"])
    import torch, transformers, datasets

# Optional PEFT / bitsandbytes for LoRA + 8-bit. Try to install only if user wants LoRA.
if USE_LORA:
    try:
        import peft
    except Exception:
        try:
            pip_install(["peft", "bitsandbytes"])
        except Exception as e:
            print("Warning: couldn't auto-install peft/bitsandbytes:", e)
            USE_LORA = False

# ---------- Hugging Face token handling ----------
hf_token = os.environ.get(HF_TOKEN_ENV_VAR_NAME)
# Also try common Kaggle token-file pattern: /kaggle/input/hf_token/token.txt
if not hf_token:
    token_file = "/kaggle/input/hf_token/token.txt"
    if os.path.exists(token_file):
        with open(token_file, "r", encoding="utf-8") as f:
            hf_token = f.read().strip()
            os.environ[HF_TOKEN_ENV_VAR_NAME] = hf_token
            print(f"Loaded HF token from {token_file}")

if hf_token:
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("Hugging Face token loaded from env var or token file.")
    except Exception as e:
        print("HF login attempt failed (continuing anyway):", e)
else:
    print(f"Warning: No Hugging Face token found in env var '{HF_TOKEN_ENV_VAR_NAME}'.")
    print("If the model requires auth, set HF_TOKEN (Kaggle: Secrets → Add new secret) or provide token file at /kaggle/input/hf_token/token.txt")

# ---------- Environment checks ----------
print("PyTorch version:", torch.__version__)
print("Transformers version:", getattr(transformers, "__version__", "unknown"))
print("CUDA available:", torch.cuda.is_available())
try:
    print("Number of GPUs:", torch.cuda.device_count())
except Exception:
    pass
print("Working dir:", os.getcwd())
print("TRAIN_FILE:", TRAIN_FILE)
print("MODEL_NAME_OR_PATH:", MODEL_NAME_OR_PATH)
print("BASE_ADAPTER_DIR (adapter files expected):", BASE_ADAPTER_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================================================
# End Cell 1
# ============================================================================


# ============================================================================
# Cell 2 (Final Fix) — CHAT TEMPLATING & TRAINING
# ============================================================================
# ============================================================================
# Cell 2 (Fixed) — LoRA ADAPTERS & TRAINING
# ============================================================================
import random
random.seed(42)
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
# IMPORT PEFT (Required for 4-bit training)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# --- CONFIG ---
MODEL_NAME = "unsloth/meta-llama-3.1-8b-bnb-4bit"
TRAIN_FILE = "/kaggle/input/training-data-chatml/training_data_chatml.jsonl"
OUTPUT_DIR = "./output-llm-finetune"
MAX_LENGTH = 1024

# 1. Load Dataset & Format Chat
print("Loading dataset...")
ds = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train")

def format_chat(example):
    # Convert 'messages' list to a single string using ChatML format
    messages = example["messages"]
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return {"text": text}

print("Applying chat formatting...")
ds = ds.map(format_chat)

# 2. Load Tokenizer & Model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Add special tokens for ChatML if missing
tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|endoftext|>"})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # This automatically handles 4-bit loading if bitsandbytes is installed
    device_map="auto",
    torch_dtype=torch.float16,
)
model.resize_token_embeddings(len(tokenizer))

# 3. ENABLE LORA (The Fix for 'Purely Quantized' Error)
print("Preparing model for LoRA training...")
# A. Prepare model for k-bit training (freezes base layers)
model = prepare_model_for_kbit_training(model)

# B. Define LoRA Config
peft_config = LoraConfig(
    r=16,                    # Rank (higher = more trainable params, usually 8, 16, 32)
    lora_alpha=32,           # Alpha (usually 2x Rank)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all linear layers for Llama
)

# C. Get PEFT Model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Verify we are training ~1-5% of params

# 4. Tokenize
def tokenize_and_chunk(examples):
    texts = examples["text"]
    enc = tokenizer(texts, return_special_tokens_mask=False, truncation=False)
    input_ids = enc["input_ids"]
    out_ids = []
    for ids in input_ids:
        for i in range(0, len(ids), MAX_LENGTH):
            chunk = ids[i:i+MAX_LENGTH]
            if len(chunk) > 0:
                out_ids.append(chunk)
    return {"input_ids": out_ids}

print("Tokenizing...")
tokenized = ds.map(tokenize_and_chunk, batched=True, remove_columns=ds.column_names)

def to_examples(batch):
    out = {"input_ids": [], "attention_mask": []}
    for ids in batch["input_ids"]:
        out["input_ids"].append(ids)
        out["attention_mask"].append([1]*len(ids))
    return out

tokenized = tokenized.map(to_examples, batched=True, remove_columns=["input_ids"])
tokenized.set_format(type="torch")

# 5. Train
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,      # LoRA usually needs higher LR than full finetune (2e-4 vs 2e-5)
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    optim="paged_adamw_8bit" # optimized optimizer for 4-bit
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    processing_class=tokenizer, # Replaces the deprecated 'tokenizer' arg
)

print("Starting training...")
trainer.train()

# 6. Save Adapter
print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")
# ============================================================================
# End Cell 2
# ============================================================================



# ============================================================================
# Cell 3 — POST-TRAINING: load best model and run a sample generation (prints example)
# ----------------------------------------------------------------------------
# Uses OUTPUT_DIR (saved model) or falls back to base model + adapter dir if present.
# ----------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Preparing generation model from", OUTPUT_DIR)
try:
    gen_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True)
    gen_model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, device_map="auto" if torch.cuda.is_available() else None)
    print("Loaded fine-tuned model from OUTPUT_DIR")
except Exception as e:
    print("Could not load model from OUTPUT_DIR (error):", e)
    # fallback: try base model + adapter if present
    print("Trying base model and adapter (if present)...")
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    gen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH, device_map="auto" if torch.cuda.is_available() else None)
    try:
        if USE_LORA and os.path.isdir(BASE_ADAPTER_DIR):
            from peft import PeftModel
            gen_model = PeftModel.from_pretrained(gen_model, BASE_ADAPTER_DIR)
            print("Applied adapter to base model for generation.")
    except Exception as e2:
        print("Could not apply adapter fallback:", e2)

gen_model.eval()
prompt = "### Instruction:\nSummarize in one sentence: \"The quick brown fox jumps over the lazy dog.\"\n\n### Response:\n"
inputs = gen_tokenizer(prompt, return_tensors="pt").input_ids
if torch.cuda.is_available():
    gen_model.to("cuda")
    inputs = inputs.to("cuda")

out = gen_model.generate(
    inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=gen_tokenizer.eos_token_id,
    pad_token_id=gen_tokenizer.pad_token_id,
)
generated = gen_tokenizer.decode(out[0], skip_special_tokens=True)
print("=== Generated sample ===")
print(generated)

# Example plausible output (your actual text will vary):
# === Generated sample ===
# ### Instruction:
# Summarize in one sentence: "The quick brown fox jumps over the lazy dog."
#
# ### Response:
# A lively fox quickly leaps over a sleepy dog, illustrating a contrast between energetic and lazy animals.
# ============================================================================
# End Cell 3
# ============================================================================
