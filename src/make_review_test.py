import json
rows = [json.loads(x) for x in open("data/processed/test.jsonl", encoding="utf-8")]
with open("data/review_test.md", "w", encoding="utf-8") as f:
    f.write("\n\n".join(
        f"### {r['example_id']} [{r['origin']}]\nQ: {r['question']}\nA: {r['answer']}\nVERDICT: ok / fix / delete"
        for r in rows))
print(len(rows), "rows written")
