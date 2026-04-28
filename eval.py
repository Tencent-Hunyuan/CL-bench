"""Evaluation Script - Using OpenAI API for Grading

Use GPT or other LLMs as the judge to grade model outputs with binary scores (0/1).

Input File:
    JSONL file with model outputs, each line contains:
    {"idx": 0, "messages": [...], "model_output": "...", "ref_answer": "...", "rubrics": [...]}

Output File:
    outputs/{model_name}_graded.jsonl

Usage:
    # Using default OpenAI API
    python eval.py --input outputs/model_output.jsonl --output outputs/model_graded.jsonl

    # Using other compatible APIs
    python eval.py --input outputs/model_output.jsonl --base-url https://api.deepseek.com/v1 --api-key your_key

    # Concurrent evaluation
    python eval.py --input outputs/model_output.jsonl --workers 5
"""

import json
import os
import argparse
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def append_jsonl(item, file_path):
    """Append a single record to JSONL file."""
    os.makedirs(
        os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True
    )
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_rubrics_text(rubrics):
    """Build rubrics checklist from rubrics list."""
    if not rubrics:
        return "No specific rubrics provided."

    lines = []
    for i, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = rubric.get("rubric_criteria", "").strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f"{i}. {criteria}")

    return "\n".join(lines) if lines else "No specific rubrics provided."


def parse_judge_result(result_text):
    """Parse judge JSON, with a best-effort fallback for chatty model outputs."""
    text = result_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    candidates = [text]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    score_match = re.search(r'"?Overall Score"?\s*[:：]\s*"?([01])', text)
    status_match = re.search(
        r'"?List of Requirement Satisfaction Status"?\s*[:：]\s*(\[[^\]]*\])',
        text,
        re.DOTALL,
    )
    rationale_match = re.search(
        r'"?Grading Rationale"?\s*[:：]\s*(".*?"|.+?)(?:,\s*"?List of Requirement Satisfaction Status"?|,\s*"?Overall Score"?|$)',
        text,
        re.DOTALL,
    )
    if score_match:
        status = []
        if status_match:
            try:
                status = json.loads(status_match.group(1))
            except json.JSONDecodeError:
                status = []
        rationale = rationale_match.group(1).strip() if rationale_match else text[:500]
        if rationale.startswith('"') and rationale.endswith('"'):
            try:
                rationale = json.loads(rationale)
            except json.JSONDecodeError:
                rationale = rationale[1:-1]
        return {
            "Grading Rationale": rationale,
            "List of Requirement Satisfaction Status": status,
            "Overall Score": int(score_match.group(1)),
        }

    raise json.JSONDecodeError("Could not parse judge result", text, 0)


def call_judge_api(
    client, model, rubrics_text, model_output, max_retries=3, retry_delay=3
):
    """
    Call judge model API for grading (only handles API call, returns raw text).

    Args:
        client: OpenAI client instance
        model: Judge model name
        rubrics_text: Formatted rubrics text
        model_output: Model's response to be graded
        max_retries: Maximum number of retries for API call
        retry_delay: Delay between retries (seconds)

    Returns:
        result_text: Raw response text from API, or None if failed
    """
    grading_prompt = (
        "Starting now, you are a rigorous instruction-following grading teacher. Your task is to accurately grade and score student answers based on the 【Rubrics】.\n\n"
        "Grading Criteria\n"
        "This is a strict, all-or-nothing grading system. The final score is binary.\n"
        "To receive a score of 1, the student's answer must perfectly satisfy every single requirement listed in the 【Rubrics】.\n"
        "If even one requirement is not fully met, the final score will be 0.\n"
        "However, do not invent stricter requirements than the 【Rubrics】 state. Be tolerant of equivalent formatting unless exact syntax is explicitly required by the 【Rubrics】.\n"
        "Formatting tolerance rules: do not fail answers solely for capitalization, punctuation, heading style, numbering style, bullet indentation, Markdown variants, code block fences, JSON key order, whitespace, or equivalent label formatting. If an answer clearly provides the requested section, field, list, or schema item, count it as present even if the presentation differs, unless the rubric explicitly requires exact visible syntax. If visible formatting such as bold or italics is applied to a whole line, paragraph, list item, or section label plus its contents, count the contained words/items as formatted; do not require each contained item to be individually wrapped. If bullets are requested for breakdowns within sections, do not fail isolated one-sentence section entries solely because they are not bullets when there is no nested breakdown to bullet.\n"
        "Structured output tolerance rules: if the task requests JSON, a schema, or a Task Result, accept either a valid JSON object or a clearly equivalent textual JSON/schema representation. Do not fail solely because structured content is stored as plain text rather than a parsed JSON file.\n"
        "Cross-section tolerance rules: if a required value, estimate, confidence/credible interval, caveat, next step, or status is clearly present elsewhere in the answer, count it as satisfying a named-section or handoff requirement unless the rubric explicitly says it must be repeated verbatim inside that exact subsection. Treat clear finality, validation, routing, action, request, or use/proceed language as an adequate status or next-step signal when the substance is clear.\n"
        "Exact-text tolerance rules: if the exact required sentence or phrase appears with identical words in the answer, count it as exact even if it is not surrounded by quotation marks or appears after a label.\n"
        "Implementation tolerance rules: when a rubric names a concrete implementation detail, accept a clearly equivalent implementation that satisfies the same functional intent unless the rubric explicitly says the exact library, file name, field name, or syntax is mandatory. For example, tabular SQL results, records, or DataFrame-like structures can satisfy data-loading requirements when the answer still provides the requested profile; read-only parsing or coercion used only to measure data quality is not an autonomous cleaning decision if it does not write or modify source data; top-level imports and constants do not violate a requirement that generated logic be wrapped in functions. If a rubric asks for comments or instructions about secure handling, count clear prose, policy fields, inline comments, docstrings, or code-adjacent notes as satisfying it when the answer conveys the secure handling rule. If a rubric asks for try-except/error handling around I/O or conversions, accept explicit validation, deliberate exception raising, error propagation with clear failure messages, or centralized wrappers as equivalent unless the answer leaves routine failures completely unhandled.\n"
        'You MUST mark these non-substantive cases as passing, not failing: imports, constants, dataclass definitions, or an `if __name__ == "__main__"` guard outside functions; static schema/table identifiers from configuration or environment variables composed into SQL while query values such as dates, IDs, filters, and user values are still bound parameters; validation that raises clear errors plus normal library exception propagation for engine creation, SQL reads, parsing, or conversions; an identifier that exactly matches the identifier shown in the task text; placeholder/null/example values in a template when executable code or instructions are also provided to populate the requested values.\n'
        "Security tolerance rules: do not treat environment variable names, placeholders, configuration keys, argparse options, examples such as PGUSER/PGPASSWORD/OPENAI_API_KEY, or instructions to set/use environment variables as asking the user to reveal credentials. Raising an error when a required secret environment variable is missing is secure handling, not a credential request. Only fail credential-related requirements when the answer hardcodes actual secret values, asks the user to paste actual secret values into the answer/code, or recommends insecure handling contrary to the 【Rubrics】.\n"
        "Source-data tolerance rules: when the prompt provides tables, artifacts, records, or other source data, using those provided values correctly counts as using the source data; do not require the answer to use one exact verb such as 'extract' unless the rubric explicitly requires that wording.\n"
        "Grading Process\n"
        "Please strictly follow the steps below for analysis—no steps may be skipped:\n"
        "Step 1: Analyze the Standard Answer\n"
        "List all explicit requirements in the 【Rubrics】 item by item (including format, content, quantity, order, etc.).\n"
        "Identify implicit requirements in the 【Rubrics】 (e.g., language style, logical structure).\n"
        'Define specific evaluation criteria for each requirement (e.g., "must include X," "must not exceed Y").\n'
        "Step 2: Check Each Requirement Against the Student's Answer\n"
        "For every requirement in the 【Rubrics】, verify one by one whether the student's answer fully satisfies it.\n"
        "Step 3: Self-Reflection\n"
        "Before giving the final score, you must conduct the following checks:\n"
        "  Completeness Check: Whether all requirements in the standard answer have been reviewed with no omissions.\n"
        '  Strictness Check: Whether the evaluation strictly adheres to the "fully satisfied" standard without relaxing requirements due to subjective judgment.\n'
        "  Consistency Check: Whether the grading rationale aligns logically with the final score.\n"
        "  Objectivity Check: Whether judgments are based on objective facts rather than subjective speculation.\n"
        "Output Format Requirements\n"
        "【Grading Rationale】: xxx\n"
        '【List of Requirement Satisfaction Status】: [x₁, x₂, …, xᵢ, …, xₙ] (where n is the total number of requirements in the 【Rubrics】, and xᵢ indicates whether the student\'s answer meets the i-th requirement, with values "yes"/"no")\n'
        "【Overall Score】: x points (x is an integer, either 0 or 1.)\n\n"
        "Content to Be Graded\n"
        f"【Rubrics】:\n{rubrics_text}\n"
        f"【Student Response】:\n{model_output}\n"
        "\nPlease strictly output ONLY the following JSON format (do not output any other content):\n"
        "{\n"
        '  "Grading Rationale": "Your detailed grading rationale",\n'
        '  "List of Requirement Satisfaction Status": ["yes", "no", ...],\n'
        '  "Overall Score": 0 or 1\n'
        "}\n"
    )

    messages = [{"role": "user", "content": grading_prompt}]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            result_text = response.choices[0].message.content.strip()

            # Remove code block wrapper if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            return result_text

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                log(
                    f"   ⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}"
                )
                time.sleep(retry_delay)
            else:
                log(
                    f"   ❌ API call failed after {max_retries} attempts: {error_msg[:100]}"
                )
                return None

    return None


def get_task_id(item):
    """Get stable task id from item metadata, fallback to idx."""
    metadata = item.get("metadata", {})
    return metadata.get("task_id", item.get("idx", -1))


def process_single_item(args):
    """Process a single item for grading."""
    item, client, judge_model, max_retries = args
    idx = get_task_id(item)

    model_output = item.get("model_output", "")
    rubrics = item.get("rubrics", [])

    # Skip if no model output
    if not model_output or not model_output.strip():
        result = {
            **item,
            "idx": idx,
            "grading_rationale": "No model output (counted as score 0)",
            "requirement_status": [],
            "score": 0,
        }
        return idx, result, None  # None is no error

    # Build rubrics text
    rubrics_text = build_rubrics_text(rubrics)

    # JSON parsing retry logic (re-call API if JSON parsing fails)
    for parse_attempt in range(max_retries):
        # Call judge API
        grading_result = call_judge_api(
            client, judge_model, rubrics_text, model_output, max_retries
        )

        if not grading_result:
            log(
                f"   ❌ [idx={idx}] API call failed (attempt {parse_attempt + 1}/{max_retries})"
            )
            if parse_attempt < max_retries - 1:
                log("      Waiting 2s before retry...")
                time.sleep(2)
                continue
            else:
                # All retries failed
                result = {
                    **item,
                    "idx": idx,
                    "grading_rationale": "API call failed (counted as score 0)",
                    "requirement_status": [],
                    "score": 0,
                }
                return idx, result, "API call failed"  # error

        # Try to parse JSON
        try:
            result_json = parse_judge_result(grading_result)

            # Validate required field
            if "Overall Score" not in result_json:
                raise ValueError("Missing 'Overall Score' field")

            # Parse success
            result = {
                **item,
                "idx": idx,
                "grading_rationale": result_json.get("Grading Rationale", ""),
                "requirement_status": result_json.get(
                    "List of Requirement Satisfaction Status", []
                ),
                "score": result_json.get("Overall Score", ""),
            }
            return idx, result, None  # None is no error

        except (json.JSONDecodeError, ValueError) as e:
            log(
                f"   ⚠️ [idx={idx}] JSON parse failed (attempt {parse_attempt + 1}/{max_retries}): {e}"
            )
            log(f"      Raw response: {grading_result[:200]}...")

            if parse_attempt < max_retries - 1:
                log("      Waiting 2s before re-grading...")
                time.sleep(2)
            else:
                log(f"   ❌ [idx={idx}] JSON parse failed after {max_retries} attempts")
                result = {
                    **item,
                    "idx": idx,
                    "grading_rationale": f"JSON parse failed ({max_retries} attempts): {grading_result[:500]}",
                    "requirement_status": [],
                    "score": 0,
                }
                return idx, result, f"JSON parse failed: {e}"  # error

    # Should not reach here
    result = {
        **item,
        "idx": idx,
        "grading_rationale": "Unknown error (counted as score 0)",
        "requirement_status": [],
        "score": 0,
    }
    return idx, result, "Unknown error"


def main():
    parser = argparse.ArgumentParser(description="Evaluation Script - OpenAI API Judge")
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSONL file path"
    )
    parser.add_argument(
        "--judge-model", type=str, default="gpt-5.1", help="Judge model name"
    )
    parser.add_argument(
        "--base-url", type=str, default=None, help="API Base URL (optional)"
    )
    parser.add_argument("--api-key", type=str, default=None, help="API Key (optional)")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of concurrent workers"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per item"
    )
    args = parser.parse_args()

    # Set output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"outputs/{base_name}_graded.jsonl"

    log("=" * 60)
    log("🎯 Evaluation Task")
    log("=" * 60)
    log(f"📥 Input file: {args.input}")
    log(f"📤 Output file: {args.output}")
    log(f"🤖 Judge model: {args.judge_model}")
    log(f"⚡ Workers: {args.workers}")
    log("=" * 60)

    # Initialize OpenAI client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("❌ Error: Please set OPENAI_API_KEY or use --api-key argument")
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

    # Check completed samples (resume from checkpoint)
    completed_indices = set()
    if os.path.exists(args.output):
        existing_data = load_jsonl(args.output)
        completed_indices = {
            get_task_id(item) for item in existing_data if get_task_id(item) is not None
        }
        log(f"📌 Found {len(completed_indices)} completed, resuming remaining")

    # Filter pending tasks
    pending_items = [
        item for item in data if get_task_id(item) not in completed_indices
    ]

    if not pending_items:
        log("✅ All samples already evaluated")
        # Calculate final statistics
        calculate_statistics(args.output)
        return

    log(f"🚀 Starting evaluation ({len(pending_items)} pending)...")

    # Prepare tasks
    tasks = [
        (item, client, args.judge_model, args.max_retries) for item in pending_items
    ]

    # Statistics
    success_count = 0
    fail_count = 0

    if args.workers == 1:
        # Single-threaded
        for task in tqdm(tasks, desc="Evaluating"):
            idx, result, error = process_single_item(task)

            if error:
                fail_count += 1
            else:
                append_jsonl(result, args.output)
                success_count += 1
    else:
        # Multi-threaded
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_item, task): task[0].get("idx")
                for task in tasks
            }

            with tqdm(total=len(tasks), desc="Evaluating") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, result, error = future.result()

                        if error:
                            fail_count += 1
                        else:
                            append_jsonl(result, args.output)
                            success_count += 1
                    except Exception as e:
                        log(f"   ❌ Exception: {str(e)}")
                        fail_count += 1
                    pbar.update(1)

    # Summary
    log("=" * 60)
    log("✅ Evaluation completed!")
    log(f"   Success: {success_count}")
    log(f"   Failed: {fail_count}")
    log(f"   Output: {args.output}")

    # Calculate final statistics
    calculate_statistics(args.output)


def calculate_statistics(output_path):
    """Calculate and display final statistics."""
    if not os.path.exists(output_path):
        return

    data = load_jsonl(output_path)

    total = len(data)
    score_0 = sum(1 for item in data if item.get("score") == 0)
    score_1 = sum(1 for item in data if item.get("score") == 1)

    log("\n📊 Final Statistics:")
    log(f"   Total samples: {total}")
    log(f"   Score 0: {score_0}")
    log(f"   Score 1: {score_1}")

    if total > 0:
        solving_rate = score_1 / total
        log(f"\n📈 Solving Rate: {solving_rate:.4f} ({score_1}/{total})")

    # Category-level scores
    category_stats = {}
    for item in data:
        metadata = item.get("metadata", {})
        category = metadata.get("context_category", "Unknown")
        stats = category_stats.setdefault(
            category, {"total": 0, "score_0": 0, "score_1": 0}
        )
        stats["total"] += 1
        if item.get("score") == 1:
            stats["score_1"] += 1
        else:
            stats["score_0"] += 1

    if category_stats:
        log("\n📂 Scores by context_category:")
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            rate = stats["score_1"] / stats["total"] if stats["total"] else 0
            log(
                f"   {category}: total={stats['total']}, "
                f"score_1={stats['score_1']}, score_0={stats['score_0']}, "
                f"rate={rate:.4f}"
            )

    log("=" * 60)


if __name__ == "__main__":
    main()
