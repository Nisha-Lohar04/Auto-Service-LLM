import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python src/merge_overrides.py <path-to-overrides.json>"
    )

incoming_path = Path(sys.argv[1])
target_path = Path("data/overrides.json")

if not incoming_path.exists():
    raise SystemExit(f"ERROR: Input file not found: {incoming_path}")

with incoming_path.open(encoding="utf-8-sig") as f:
    incoming = json.load(f)

with target_path.open(encoding="utf-8-sig") as f:
    current = json.load(f)

if not isinstance(incoming, dict) or not isinstance(current, dict):
    raise SystemExit("ERROR: Expected both files to contain JSON objects.")

added = 0
updated = 0
skipped = 0

for question, value in incoming.items():
    key = question if question.startswith("question:") else "question:" + question

    if isinstance(value, str):
        answer = value
    elif isinstance(value, dict) and isinstance(value.get("answer"), str):
        answer = value["answer"]
    else:
        raise SystemExit(f"ERROR: Invalid answer for question: {question}")

    if key in current:
        current[key]["answer"] = answer
        current[key]["review_status"] = "edited_by_reviewer"
        updated += 1
    else:
        current[key] = {
            "answer": answer,
            "review_status": "edited_by_reviewer"
        }
        added += 1

with target_path.open("w", encoding="utf-8") as f:
    json.dump(current, f, ensure_ascii=False, indent=2)
    f.write("\n")

print(f"New overrides added: {added}")
print(f"Existing overrides updated: {updated}")
print(f"Total overrides now: {len(current)}")
print("Merge completed.")
