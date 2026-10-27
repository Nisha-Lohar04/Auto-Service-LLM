# AutoService-LLM: results

A small study of whether fine-tuning a 1.5B open model on car-maintenance Q&A beats prompting it. This is a prototype experiment, not a production system.

## Setup

| Item | Detail |
|---|---|
| Model | Qwen2.5-1.5B-Instruct |
| Fine-tuning | 4-bit QLoRA (NF4), LoRA r=16, alpha=32, dropout 0.05, all attention and MLP projections |
| Training | 3 epochs, lr 2e-4 (cosine), batch 4 x grad-accum 2, fp16, Google Colab T4 |
| Final losses | train 0.828, validation 1.158 (best validation 1.138 at epoch 2) |
| Compared | **Base** (question only), **Prompted** (system prompt + 3 fixed train examples), **Fine-tuned** (adapter, question only) |
| Decoding | greedy, max 200 new tokens |

## Data

- 233 Q&A examples in 66 topic groups: 49 drafted from NHTSA tyre-safety guidance and 184 written by an LLM (Claude) from general automotive knowledge.
- Split by topic group (no group appears in two splits): train 162 / validation 24 / test 47. A leakage check runs in `src/build_dataset.py`.
- Review: all 47 test rows were checked by hand (11 needed corrections, 23%) and a random 50 of the 162 training rows (16 needed corrections, 32%). The rest of the training rows are unreviewed.
- Corrections live in `data/overrides.json` and are applied on every rebuild.

## Results on the 47 held-out test questions

**Automatic metric (ROUGE-L against the reference answer)**

| Model | ROUGE-L | Avg answer length (words) |
|---|---|---|
| Base | 0.098 | 147 |
| Prompted | 0.161 | 28 |
| Fine-tuned | 0.183 | 26 |

**Blind hand scoring (20 random test questions, one rater, 0-2 scale)**
2 = correct and safe, 1 = partly right or vague but not harmful, 0 = wrong or unsafe. Answers were shuffled so the rater did not know which model wrote which.

| Model | Total (max 40) | Average | Score 2 | Score 1 | Score 0 |
|---|---|---|---|---|---|
| Base | 26 | 1.30 | 10 | 6 | 4 |
| Prompted | 22 | 1.10 | 7 | 8 | 5 |
| Fine-tuned | 22 | 1.10 | 7 | 8 | 5 |

Head to head on the 20 questions, the fine-tuned model beat base on 3, lost on 7 and tied on 10. Per-answer scores are in `results/hand_scores.csv`.

## What I found

- Fine-tuning changed the style: answers became short and direct, and ROUGE-L rose from 0.098 to 0.183.
- It did **not** improve correctness in the hand-scored sample. The long base answers covered more correct points, which ROUGE-L does not reward.
- One safety example did improve: for "Can I ignore the tire-pressure warning light if the car drives normally?", base and prompted answered "Yes", while the fine-tuned model answered "No ... have the vehicle inspected promptly". This is a single example, not a measured rate.
- The fine-tuned model was weakest on procedure and storage questions (jump-starting, handbrake, starting a stored car, drying brakes), where short answers dropped key steps or invented details.
- With 162 training rows, fine-tuning teaches answer style, not new facts. For factual accuracy, retrieval over trusted manuals is a more promising next step.

## Limitations

- Small test set (47) and a small hand-scored sample (20); a 4-point difference could be noise.
- One rater, who is also the dataset author. No inter-rater agreement.
- 184 of 233 examples were LLM-written and only partly verified. Reference answers carry that uncertainty.
- Latency numbers in the CSVs are not a fair speed comparison (base answers hit the token cap; the adapter was not merged), so they are not reported here.
- NHTSA guidance is US-specific.

## Next steps

1. Add retrieval over source passages and compare against these three settings.
2. Retrain with 2 epochs and more procedural examples, and re-score.
3. Review the remaining training rows and add a second rater.

## Files

- `data/` raw sources, batches, overrides, processed splits
- `src/build_dataset.py` builds and checks the splits
- `notebooks/` baseline evaluation and QLoRA fine-tuning (Colab)
- `results/` predictions, summaries and hand scores
- Adapter weights are not stored in this repository.
