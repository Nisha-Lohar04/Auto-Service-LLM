
# AutoService-LLM

**Does fine-tuning a small open-source LLM improve automotive maintenance Q&A compared with prompting alone?**

AutoService-LLM is an end-to-end experiment that investigates this question through dataset construction, QLoRA fine-tuning, and comparative evaluation of three configurations of Qwen2.5-1.5B-Instruct.

The project compares a base model, a few-shot prompted model, and a fine-tuned model using ROUGE-L and blind human evaluation to examine whether improvements in generated answers translate into better correctness and safety.

> **Project scope:** An experimental prototype for learning and evaluation. It is not a production system and should not replace a vehicle owner's manual or advice from a qualified mechanic.

## Key Findings

- **ROUGE-L improved from 0.098 to 0.183** after fine-tuning.
- **Human evaluation showed no improvement:** the fine-tuned model scored 22/40, compared with 26/40 for the base model.
- **Answer style changed substantially:** outputs became shorter and more direct, which may have contributed to the higher ROUGE-L score.
- **Retrieval-augmented generation (RAG)** is a promising next step for grounding automotive answers in trusted technical documentation.

The central finding is that better automatic evaluation scores do not necessarily indicate better factual answers.

## Experiment Overview

| Component | Details |
|---|---|
| Task | Automotive service and maintenance question answering |
| Base model | Qwen2.5-1.5B-Instruct |
| Fine-tuning | 4-bit QLoRA |
| LoRA configuration | Rank 16, alpha 32 |
| Training environment | Google Colab, free T4 GPU |
| Dataset | 233 Q&A examples across 66 topic groups |
| Data split | 162 training / 24 validation / 47 test |
| Evaluation | ROUGE-L and blind human scoring |
| Frameworks | Transformers, PEFT, TRL, bitsandbytes |

## What I Built

### 1. Dataset Construction and Validation

Implemented a reproducible dataset pipeline in `src/build_dataset.py` to combine automotive Q&A examples, apply reviewer corrections, and produce leakage-safe train, validation, and test splits.

- Combined NHTSA-sourced tire-related Q&A with LLM-drafted examples covering 46 other topics.
- Removed duplicate questions and applied 26 reviewer corrections stored in `data/overrides.json`.
- Applied corrections using exact question text so they persist across dataset rebuilds.
- Split the dataset by topic group to prevent questions belonging to the same group from appearing across different splits.
- Added assertions to verify group separation and validate the resulting dataset.

The pipeline produces 233 examples distributed across 66 topic groups, with 47 questions reserved for held-out evaluation.

### 2. Human Review of the Dataset

Reviewed the reference answers to improve dataset quality before evaluating the models.

| Review activity | Examples reviewed | Corrections |
|---|---:|---:|
| Held-out test references | 47 | 11 |
| Sample of training rows | 50 | 16 |

Corrections are maintained separately from the original data and reapplied during dataset construction, making the process reproducible.

### 3. Baseline and Prompted Evaluation

Implemented `notebooks/baseline_eval.ipynb` to compare the base model with a prompted configuration on the held-out test set.

The prompted model uses a system prompt and three examples to encourage concise, task-appropriate automotive maintenance answers without updating model weights.

### 4. QLoRA Fine-Tuning

Implemented `notebooks/finetune_qlora.ipynb` to fine-tune Qwen2.5-1.5B-Instruct using parameter-efficient training.

The training setup includes:

- 4-bit NF4 quantization.
- LoRA adapters with rank 16 and alpha 32.
- Hugging Face Transformers, PEFT, TRL, and bitsandbytes.
- Training and validation loss tracking across epochs.

The fine-tuning notebook saves the adapter and generated predictions. Adapter weights are not included in this repository.

### 5. Blind Human Evaluation

Designed a blind comparison to assess answer correctness and safety independently of the automatic metric.

- Sampled 20 questions from the held-out test set.
- Presented answers from all three model configurations in shuffled, unlabeled form.
- Scored each answer on a three-level scale: 0, 1, or 2.
- Evaluated whether answers were correct and safe, rather than rewarding concise wording alone.

**Scoring rubric**

| Score | Interpretation |
|---:|---|
| 2 | Correct and safe |
| 1 | Partly correct or vague |
| 0 | Incorrect or unsafe |

The evaluation was conducted by a single rater, who also authored the dataset. The results should therefore be interpreted as preliminary evidence rather than a definitive ranking.

## Results

### Automatic Evaluation

All three configurations were evaluated on the same held-out test set of 47 questions.

| Model | ROUGE-L | Average answer length |
|---|---:|---:|
| Base | 0.098 | 147 words |
| Prompted | 0.161 | 28 words |
| Fine-tuned | 0.183 | 26 words |

Fine-tuning achieved the highest ROUGE-L score, an absolute increase of 0.085 over the base model.

However, ROUGE-L measures lexical overlap with reference answers. It does not directly establish factual correctness, completeness, or safety.

### Blind Human Evaluation

Twenty test questions were scored, giving each model a maximum possible score of 40.

| Model | Total score | Average score | Score 2 | Score 1 | Score 0 |
|---|---:|---:|---:|---:|---:|
| Base | 26/40 | 1.30 | 10 | 6 | 4 |
| Prompted | 22/40 | 1.10 | 7 | 8 | 5 |
| Fine-tuned | 22/40 | 1.10 | 7 | 8 | 5 |

The fine-tuned model tied with the prompted model and scored below the base model in this sample.

The four-point difference between the base and the other configurations is descriptive, not evidence of a statistically significant difference.

### Interpreting the Results

**1. Higher ROUGE-L did not translate into higher human scores.**

The fine-tuned model produced shorter answers that more closely matched the reference wording. Human scoring, however, did not show improved correctness or safety.

**2. Concision sometimes came at the expense of completeness.**

The base model's longer answers often covered more relevant points. The fine-tuned model was weaker on procedural questions involving jump-starting, vehicle storage, and brake care, where omitted steps can matter.

**3. One safety-relevant answer improved.**

When asked whether a tire-pressure warning light could be ignored, the base and prompted models answered yes, whereas the fine-tuned model answered no and recommended inspection. This is a useful qualitative example, not evidence of an overall safety improvement.

**4. The dataset may be too small to teach reliable new factual knowledge.**

With only 162 training examples, fine-tuning may have primarily influenced response style rather than establishing dependable automotive knowledge.

These findings motivate evaluating retrieval over trusted service documentation as an alternative or complement to fine-tuning.

## Limitations

- **Small evaluation sample:** The test set contains 47 questions, and blind human evaluation covers only 20.
- **Single evaluator:** No inter-rater agreement or independent assessment was available.
- **Mixed data provenance:** Of the 233 examples, 184 were LLM-written from general knowledge and only partially verified. The remaining 49 tire-related examples were based on NHTSA guidance, which is US-specific.
- **Metric limitations:** ROUGE-L measures text overlap and cannot independently verify factual correctness, procedural completeness, or safety.
- **Limited domain coverage:** The dataset does not establish performance across vehicle makes, models, manufacturers, or service-manual specifications.
- **Latency not reported:** Measurements were not comparable because base-model answers hit the token limit and the fine-tuned adapter was evaluated without merging its weights.

The results should be interpreted as an exploratory comparison under a specific dataset, model, and evaluation setup.

## Repository Structure

```text
auto-service-llm/
├── data/
│   ├── raw/                   # Source passages, catalog, initial Q&A
│   ├── batches/               # LLM-drafted Q&A by topic
│   ├── overrides.json         # Persistent reviewer corrections
│   └── processed/             # Train, validation, and test JSONL
├── notebooks/
│   ├── baseline_eval.ipynb    # Base vs. prompted evaluation
│   └── finetune_qlora.ipynb    # QLoRA fine-tuning and evaluation
├── results/                   # Predictions, summaries, human scores
├── src/
│   └── build_dataset.py       # Dataset construction and validation
├── RESULTS.md                 # Detailed experimental report
└── README.md
```

Each example in `data/processed/*.jsonl` contains:

- `example_id`
- `group_id`
- `topic`
- `question`
- `answer`
- `origin`
- `source_url`
- `review_status`

## Reproducing the Experiment

### 1. Rebuild the Dataset

Requires Python 3.

```bash
python src/build_dataset.py
```

The script reports the dataset split sizes, applied overrides, and the result of the topic-group leakage checks.

Expected split sizes:

```text
Train:      162
Validation:  24
Test:        47
Total:      233
```

### 2. Run Baseline Evaluation

1. Open `notebooks/baseline_eval.ipynb` in Google Colab.
2. Select a T4 GPU runtime.
3. Run the notebook cells to evaluate the base and prompted models.

### 3. Run Fine-Tuning

1. Open `notebooks/finetune_qlora.ipynb` in Google Colab.
2. Select a T4 GPU runtime.
3. Run the notebook cells to fine-tune the model and generate predictions.

Both notebooks clone the repository to access the dataset. The fine-tuning notebook saves the trained adapter and prediction files in its execution environment.

For the complete experimental setup, evaluation details, and results, see [`RESULTS.md`](RESULTS.md).

## Tech Stack

- **Language:** Python
- **Models:** Qwen2.5-1.5B-Instruct
- **Fine-tuning:** QLoRA, LoRA, NF4 quantization
- **ML frameworks:** Hugging Face Transformers, PEFT, TRL, Datasets
- **Quantization:** bitsandbytes
- **Evaluation and data processing:** rouge-score, pandas
- **Environment:** Google Colab, T4 GPU

## Future Work

1. **Retrieval-augmented generation:** Retrieve relevant passages from trusted automotive service manuals and compare grounded answers against the three existing configurations.
2. **Improve procedural coverage:** Add verified examples covering multi-step maintenance and safety-critical procedures, then evaluate the effect of additional training.
3. **Expand human evaluation:** Include a larger sample and a second independent rater to assess consistency.
4. **Improve answer evaluation:** Add explicit checks for factual accuracy, required procedural steps, unsupported claims, and safety-critical omissions.

## Author

**Nisha Lohar**

[GitHub Profile](https://github.com/Nisha-Lohar04)

---

*AutoService-LLM is an educational research project. Always verify vehicle-specific procedures against the manufacturer's documentation and consult a qualified mechanic when appropriate.*
