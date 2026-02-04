"""
# CELL ONE: SETUP INSTRUCTIONS (READ FIRST)
# =============================================================================

# This script performs "Teacher Training" (Data Distillation) using Llama-3-70B.
# It uses a sophisticated "Chain-of-Thought" prompt to generate high-quality
# financial reasoning and labels for your dataset.

# ----------------- PREREQUISITES FOR KAGGLE -----------------

# 1. GROQ API KEY:
#    - Sign up at https://console.groq.com/
#    - Generate a free API Key.
#    - In your Kaggle Notebook, go to the top menu: "Add-ons" -> "Secrets".
#    - Click "Add a new secret".
#    - Label: GROQ_API_KEY
#    - Value: [Paste your API Key here]
#    - Click "Save".

# 2. DATASET:

#    - Ensure your input dataset (raw JSONL file) is added to the Kaggle notebook.
#    - Update 'TARGET_FILENAME' below if your file is named differently.
#    - Default expected structure per line: {"input": "Text to analyze..."}

# 3. INSTALLATION:
#    - Run this command in a separate cell before running this script:
#      !pip install groq
# =============================================================================

"""

import os
import json
import time
from groq import Groq

# Try to import Kaggle secrets, handle gracefully if running locally
try:
    from kaggle_secrets import UserSecretsClient
    HAS_KAGGLE_SECRETS = True
except ImportError:
    HAS_KAGGLE_SECRETS = False
    print("⚠️ Kaggle Secrets not found. Ensure GROQ_API_KEY is set in environment variables if running locally.")

# --- CONFIGURATION ---
TARGET_FILENAME = "nifty50_train_data.jsonl"  # REPLACE with your actual file name
OUTPUT_FILE = "llama_70b_labeled_dataset.jsonl"
FINAL_TRAIN_FILE = "training_data_chatml.jsonl"
MODEL_ID = "llama-3.3-70b-versatile" 

# --- ADVANCED SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a Senior Portfolio Manager and Quantitative Analyst at a top-tier hedge fund. Your task is to analyze financial news/reports and provide a structured, high-signal assessment.

### ANALYTICAL PROCESS (INTERNAL STEPS):
1. **Analyze**: Identify the core event (Earnings, Merger, Macro, Product Launch, Regulation).
2. **Rethink**: Look for nuance. Does a "beat" in earnings hide weak guidance? Is the news already priced in? Are there contradictory signals in the text?
3. **Evaluate**: Assess the credibility of the signal. Is this a short-term noise or a long-term fundamental shift?
4. **Synthesize**: Formulate a "Bullish", "Bearish", or "Neutral" stance with a confidence score (0.0 to 1.0).

### OUTPUT FORMAT (STRICT):
You must output the final analysis in the following key-value format. Do not use Markdown code blocks or introduction text. Just the raw block.

Sentiment: [Bullish/Bearish/Neutral]
Confidence: [0.0 - 1.0]
Impact: [Strong/Medium/Weak]
Horizon: [Short/Medium/Long]
Event: [Brief Category, e.g., Earnings, Geopolitics]
Summary: [A dense, professional paragraph summarizing the "Why". Mention specific tickers, numbers, and the rationale for the sentiment. Focus on the causal link between the event and the price action.]
Decision: [BUY/SELL/HOLD]
"""

# --- 1. SETUP & UTILITIES ---

def get_api_client():
    """Retrieves API Key from Kaggle Secrets or Env Vars."""
    api_key = None
    if HAS_KAGGLE_SECRETS:
        try:
            api_key = UserSecretsClient().get_secret("GROQ_API_KEY")
        except Exception:
            pass
    
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
        
    if not api_key:
        raise ValueError("❌ CRITICAL: GROQ_API_KEY not found. Please add it to Kaggle Secrets.")
        
    return Groq(api_key=api_key)

def find_input_file(filename):
    """Recursively searches /kaggle/input for the specific filename."""
    search_path = '/kaggle/input'
    if not os.path.exists(search_path):
        search_path = '.' # Local fallback
        
    for dirname, _, filenames in os.walk(search_path):
        for f in filenames:
            if f == filename:
                return os.path.join(dirname, f)
    return None

def get_teacher_label(client, text):
    """Sends text to the Teacher model and retrieves the structured label."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text:\n\n{text[:15000]}"} 
            ],
            temperature=0.2, 
            max_tokens=1024,
            top_p=0.9
        )
        return completion.choices[0].message.content
    except Exception as e:
        if "429" in str(e):
            return "LIMIT_REACHED"
        print(f"⚠️ API Error: {e}")
        return None

# --- 2. MAIN LABELING PIPELINE (FIXED) ---

def run_labeling_task():
    print(f"🔍 Searching for input file: {TARGET_FILENAME}...")
    input_path = find_input_file(TARGET_FILENAME)
    
    if not input_path:
        print(f"❌ Error: Input file '{TARGET_FILENAME}' not found. Please attach the dataset.")
        return 0

    print(f"✅ Found input: {input_path}")
    
    try:
        client = get_api_client()
    except ValueError as e:
        print(e)
        return 0

    # --- RESUME LOGIC ---
    processed_inputs = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"🔄 Checking existing progress in {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'input' in data:
                        processed_inputs.add(data['input'])
                except json.JSONDecodeError:
                    continue
        print(f"✅ Found {len(processed_inputs)} already labeled samples. Resuming...")
    # ---------------------

    print("🚀 Starting Teacher Labeling Task (Llama-3-70B)...")
    
    new_processed_count = 0
    
    # Open Input (Read) and Output (Append)
    with open(input_path, 'r') as f_in, open(OUTPUT_FILE, 'a') as f_out:
        lines = f_in.readlines()
        total_lines = len(lines)
        
        for i, line in enumerate(lines):
            try:
                item = json.loads(line)
                input_text = item.get('input', '')
                
                if not input_text.strip(): continue

                # SKIP if already done
                if input_text in processed_inputs:
                    continue

                print(f"[{i+1}/{total_lines}] Analyzing...", end=" ", flush=True)
                
                # Fetch Label
                summary = get_teacher_label(client, input_text)
                
                if summary == "LIMIT_REACHED":
                    print("\n🛑 Rate Limit Hit (429). Progress saved. Stopping now.")
                    break 
                
                if summary:
                    if "Sentiment:" in summary and "Decision:" in summary:
                        item['output'] = summary
                        
                        # Write immediately + flush (Saves against crashes)
                        f_out.write(json.dumps(item) + '\n')
                        f_out.flush()
                        
                        new_processed_count += 1
                        print("✅ Label Generated.")
                    else:
                        print("⚠️ Malformed Output (Skipping).")
                else:
                    print("❌ Failed.")

                # Rate limit buffer (Increase to 5-10s if you hit limits often)
                time.sleep(3) 
                
            except json.JSONDecodeError:
                continue

    print(f"\n✨ Job Finished. Newly processed items: {new_processed_count}")
    return new_processed_count

# --- 3. FORMATTING FOR TRAINING (CHATML) ---

def convert_to_chatml():
    """Converts the labeled data into ChatML format for fine-tuning."""
    if not os.path.exists(OUTPUT_FILE):
        return

    print("\n🔄 Converting to ChatML format...")
    formatted_data = []
    
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                chat_entry = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": item['input']},
                        {"role": "assistant", "content": item['output']}
                    ]
                }
                formatted_data.append(chat_entry)
            except json.JSONDecodeError:
                continue

    with open(FINAL_TRAIN_FILE, 'w') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Training Data Ready: {FINAL_TRAIN_FILE}")
    print(f"📊 Total Samples: {len(formatted_data)}")

# --- EXECUTION ---
if __name__ == "__main__":
    count = run_labeling_task()
    # Always run conversion if the output file exists, even if count is 0 (in case we are just re-running to format)
    if os.path.exists(OUTPUT_FILE):
        convert_to_chatml()