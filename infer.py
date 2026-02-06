"""Inference Script - Using Standard OpenAI API

Process message-format JSONL data and call OpenAI-compatible APIs for inference.

Input File:
    CL-bench.jsonl - Each line contains {"messages": [...], "rubrics": [...], "metadata": {...}}

Output File:
    outputs/{model_name}.jsonl

Usage:
    # Using default OpenAI API
    python infer.py --model gpt-5.1 --input CL-bench.jsonl --output outputs/gpt5-1.jsonl
    
    # Using other compatible APIs (e.g., DeepSeek, Qwen, etc.)
    python infer.py --model deepseek-chat --base-url https://api.deepseek.com/v1 --api-key your_key
    
    # Concurrent inference
    python infer.py --model gpt-5.1 --workers 5
"""

import json
import os
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(message):
    """Print log message with timestamp."""
    print(f"[{get_timestamp()}] {message}")


def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(item, file_path):
    """Append a single record to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def call_openai_api(client, messages, model, max_retries=3, retry_delay=3):
    """
    Call OpenAI-compatible API.
    
    Args:
        client: OpenAI client instance
        messages: List of messages
        model: Model name
        max_retries: Maximum number of retries
        retry_delay: Delay between retries (seconds)
    
    Returns:
        response_text: Model response text
        error: Error message (if any)
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content, None
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                log(f"   ⚠️ Call failed (attempt {attempt + 1}): {error_msg[:100]}")
                log(f"   ⏳ Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                log(f"   ❌ Final failure: {error_msg[:200]}")
                return None, error_msg
    
    return None, "Unknown error"


def process_single_case(args):
    """Process a single data sample."""
    idx, item, client, model = args
    
    # Get messages
    messages = item.get("messages") 
    
    if not messages:
        return idx, None, "No messages found"
    
    # Call API
    response_text, error = call_openai_api(client, messages, model)
    
    if error:
        return idx, None, error
    
    result = {
        "idx": idx,
        "messages": messages,
        "model_output": response_text,
        "rubrics": item.get("rubrics", []),
        "metadata": item.get("metadata", {})
    }
    
    return idx, result, None


def main():
    parser = argparse.ArgumentParser(description="Simple Inference Script - OpenAI API")
    parser.add_argument("--model", type=str, default="gpt-5.1", help="Model name")
    parser.add_argument("--input", type=str, default="CL-bench.jsonl", help="Input file path")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL (optional)")
    parser.add_argument("--api-key", type=str, default=None, help="API Key (optional, defaults to env var)")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process (for testing)")
    parser.add_argument("--retry-delay", type=int, default=3, help="Retry delay in seconds")
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        model_name_safe = args.model.replace("/", "_").replace(":", "_")
        args.output = f"outputs/{model_name_safe}.jsonl"
    
    log(f"📂 Input file: {args.input}")
    log(f"📂 Output file: {args.output}")
    log(f"🤖 Model: {args.model}")
    log(f"🔧 Workers: {args.workers}")
    
    # Initialize OpenAI client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("❌ Error: Please set OPENAI_API_KEY environment variable or use --api-key argument")
        return
    
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
        log(f"🔗 Using custom API: {args.base_url}")
    
    client = OpenAI(**client_kwargs)
    
    # Load data
    log("📖 Loading data...")
    data = load_jsonl(args.input)
    log(f"   Total {len(data)} samples")
    
    if args.max_samples:
        data = data[:args.max_samples]
        log(f"   Limited to {args.max_samples} samples")
    
    # Check completed samples (resume from checkpoint)
    completed_indices = set()
    if os.path.exists(args.output):
        existing_data = load_jsonl(args.output)
        completed_indices = {item.get("idx") for item in existing_data if item.get("idx") is not None}
        log(f"📌 Found {len(completed_indices)} completed, resuming remaining")
    
    # Use metadata.task_id as stable unique identifier
    def get_task_id(item):
        return item["metadata"]["task_id"]
    
    # Filter pending tasks
    tasks = [(get_task_id(item), item, client, args.model) for item in data if get_task_id(item) not in completed_indices]
    
    if not tasks:
        log("✅ All samples already processed")
        return
    
    log(f"🚀 Starting inference ({len(tasks)} pending)...")
    
    # Statistics
    success_count = 0
    fail_count = 0
    
    if args.workers == 1:
        # Single-threaded sequential execution
        for task in tqdm(tasks, desc="Inference"):
            idx, result, error = process_single_case(task)
            if result:
                append_jsonl(result, args.output)
                success_count += 1
            else:
                log(f"   ❌ Sample {idx} failed: {error}")
                fail_count += 1
    else:
        # Multi-threaded concurrent execution
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_case, task): task[0] for task in tasks}
            
            with tqdm(total=len(tasks), desc="Inference") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        idx, result, error = future.result()
                        if result:
                            append_jsonl(result, args.output)
                            success_count += 1
                        else:
                            log(f"   ❌ Sample {idx} failed: {error}")
                            fail_count += 1
                    except Exception as e:
                        log(f"   ❌ Sample {idx} exception: {str(e)}")
                        fail_count += 1
                    pbar.update(1)
    
    # Summary
    log("=" * 50)
    log(f"✅ Inference completed!")
    log(f"   Success: {success_count}")
    log(f"   Failed: {fail_count}")
    log(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
