import json, random

rows = [json.loads(x) for x in open("data/processed/train.jsonl", encoding="utf-8")]
random.seed(1)
sample = random.sample(rows, 50)

blocks = []
for r in sample:
    blocks.append(
        f"### {r['example_id']} [{r['origin']}]\n"
        f"Q: {r['question']}\n"
        f"A: {r['answer']}\n"
        f"VERDICT: ok / fix / delete"
    )

with open("data/review_sample.md", "w", encoding="utf-8") as f:
    f.write("\n\n".join(blocks))
print("Wrote data/review_sample.md with", len(sample), "rows")
