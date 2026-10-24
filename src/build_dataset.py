"""Build the automotive QA dataset with persistent reviewed answer overrides."""

import glob
import json
import os
import random
import re
import runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw", "qa_raw.jsonl")
OVERRIDE_PATH = os.path.join(ROOT, "data", "overrides.json")
BATCH_DIR = os.path.join(ROOT, "data", "batches")
OUTPUT_DIR = os.path.join(ROOT, "data", "processed")

SKIP_TOPICS = {"tyre_pressure", "tyre_rotation_alignment", "tyre_tread_age"}
ALLOWED_OVERRIDE_FIELDS = {"answer", "review_status"}


def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_overrides():
    if not os.path.exists(OVERRIDE_PATH):
        return {}

    with open(OVERRIDE_PATH, encoding="utf-8-sig") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        raise ValueError("data/overrides.json must contain a JSON object.")

    for key, value in overrides.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Every override must have a non-empty ID.")

        # A string is shorthand for a replacement answer.
        if isinstance(value, str):
            overrides[key] = {"answer": value}
        elif not isinstance(value, dict):
            raise ValueError(
                f"Override {key!r} must be an answer string or an object."
            )

        unknown = set(overrides[key]) - ALLOWED_OVERRIDE_FIELDS
        if unknown:
            raise ValueError(
                f"Unsupported fields for override {key!r}: {sorted(unknown)}"
            )

        if not overrides[key]:
            raise ValueError(f"Override {key!r} contains no corrections.")

        for field, content in overrides[key].items():
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Override {key!r}: {field} must be a non-empty string."
                )

    return overrides


def apply_override(row, override):
    if "answer" in override:
        row["answer"] = override["answer"]
    if "review_status" in override:
        row["review_status"] = override["review_status"]


overrides = load_overrides()
applied = set()
rows = []

# 1. Load sourced examples, retaining their original IDs and metadata.
raw_rows = load_jsonl(RAW_PATH)
raw_ids = [r.get("example_id") for r in raw_rows]

if any(not isinstance(i, str) or not i for i in raw_ids):
    raise ValueError("Every raw example must have a non-empty example_id.")

if len(raw_ids) != len(set(raw_ids)):
    raise ValueError("Duplicate example_id values found in qa_raw.jsonl.")

raw_id_set = set(raw_ids)

for r in raw_rows:
    row = {
        "origin": r["source_id"],
        "source_id": r["source_id"],
        "source_example_id": r["example_id"],
        "source_title": r.get("source_title", ""),
        "source_url": r.get("source_url", ""),
        "passage_id": r["passage_id"],
        "group_id": r["passage_id"],
        "topic": r["topic"],
        "safety_level": r.get("safety_level", ""),
        "language": r.get("language", ""),
        "question": r["question"],
        "answer": r["answer"],
        "review_status": r["review_status"],
    }

    if r["example_id"] in overrides:
        apply_override(row, overrides[r["example_id"]])
        applied.add(r["example_id"])

    rows.append(row)

# 2. Load authored batches. Keep the existing topic exclusions.
for path in sorted(glob.glob(os.path.join(BATCH_DIR, "batch*.py"))):
    data = runpy.run_path(path)["DATA"]

    for tid, (title, pairs) in data.items():
        if tid in SKIP_TOPICS:
            continue

        for q, a in pairs:
            rows.append({
                "origin": "claude_authored",
                "group_id": tid,
                "topic": title,
                "question": q,
                "answer": a,
                "source_url": "",
                "review_status": "unverified_needs_fact_review",
            })

# 3. Deduplicate questions, preserving the existing first-occurrence rule.
seen = set()
unique_rows = []

for row in rows:
    key = re.sub(r"[^a-z0-9 ]", "", row["question"].lower()).strip()
    if key not in seen:
        seen.add(key)
        unique_rows.append(row)

rows = unique_rows

# Preserve the existing generated IDs used by the review workflow.
for i, row in enumerate(rows):
    row["example_id"] = f"ex{i:04d}"

# 4. Apply overrides by generated ID or exact question.
for row in rows:
    keys = [row["example_id"], "question:" + row["question"]]
    matching = [key for key in keys if key in overrides]
    if len(matching) > 1:
        raise ValueError(f"Multiple overrides match {row['question']!r}")
    if matching:
        key = matching[0]
        apply_override(row, overrides[key])
        applied.add(key)

unknown_ids = set(overrides) - applied
if unknown_ids:
    raise ValueError(
        "Overrides reference unknown or deduplicated IDs: "
        + ", ".join(sorted(unknown_ids))
        + ". No processed files were written."
    )

# 5. Split by group, retaining the original deterministic split strategy.
groups = sorted({row["group_id"] for row in rows})
random.seed(42)
random.shuffle(groups)

n = len(groups)
n_test = round(0.20 * n)
n_val = round(0.10 * n)

test_groups = set(groups[:n_test])
val_groups = set(groups[n_test:n_test + n_val])

splits = {"train": [], "validation": [], "test": []}

for row in rows:
    if row["group_id"] in test_groups:
        split = "test"
    elif row["group_id"] in val_groups:
        split = "validation"
    else:
        split = "train"
    splits[split].append(row)

# Validate before replacing any processed files.
group_sets = [
    {row["group_id"] for row in split_rows}
    for split_rows in splits.values()
]

assert not (group_sets[0] & group_sets[1])
assert not (group_sets[0] & group_sets[2])
assert not (group_sets[1] & group_sets[2])

# 6. Write processed datasets only after validation succeeds.
os.makedirs(OUTPUT_DIR, exist_ok=True)

for name, split_rows in splits.items():
    output_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in split_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"{name}: {len(split_rows)} examples, "
        f"{len({r['group_id'] for r in split_rows})} groups"
    )

print(f"PASS: no group overlap. Total {len(rows)} examples, {n} groups.")
print(f"Applied overrides: {len(applied)}")
if applied:
    print("Override IDs:", ", ".join(sorted(applied)))
