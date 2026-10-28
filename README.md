
# AutoService-LLM

**Can fine-tuning a small language model improve automotive maintenance question answering beyond prompting alone?**

AutoService-LLM is an experimental project that investigates the effectiveness of parameter-efficient fine-tuning for automotive question answering. It compares three configurations of Qwen2.5-1.5B-Instruct: the base model, a few-shot prompted model, and a QLoRA fine-tuned model.

The experiment combines custom dataset construction, memory-efficient fine-tuning, automatic evaluation, and blind human assessment to examine whether improved text similarity translates into better answer correctness and safety.


## Key Findings

- **ROUGE-L increased from 0.098 to 0.183** after fine-tuning on 162 training examples.
- **Human evaluation showed no improvement:** the fine-tuned model scored 22/40, compared with 26/40 for the base model.
- **Fine-tuning changed response style:** answers became shorter and more direct, potentially contributing to higher lexical overlap with reference answers.
- **Next direction:** Evaluate retrieval-augmented generation (RAG) using trusted automotive documentation to improve factual grounding.

**Key takeaway:** A higher automatic evaluation score does not necessarily mean an AI model produces more correct or safer answers.

## Experiment Overview

| Component | Details |
|---|---|
| Task | Automotive service and maintenance Q&A |
| Model | Qwen2.5-1.5B-Instruct |
| Configurations | Base, few-shot prompted, QLoRA fine-tuned |
| Fine-tuning method | 4-bit QLoRA with NF4 quantization |
| LoRA configuration | Rank 16, alpha 32 |
| Training environment | Google Colab, T4 GPU |
| Dataset | 233 Q&A examples across 66 topic groups |
| Data split | 162 training / 24 validation / 47 test |
| Evaluation | ROUGE-L and blind human scoring |
| Core frameworks | Transformers, PEFT, TRL, bitsandbytes |

## Project Workflow

1. **Dataset construction:** Combine automotive Q&A sources, remove duplicate questions, and apply persistent reviewer corrections.
2. **Leakage-safe splitting:** Separate examples by topic group into training, validation, and test sets.
3. **Baseline evaluation:** Compare the base model with a few-shot prompted model.
4. **Parameter-efficient fine-tuning:** Train QLoRA adapters using a Google Colab T4 GPU.
5. **Comparative evaluation:** Measure ROUGE-L and conduct blind human scoring.
6. **Analysis:** Compare text similarity with human-assessed correctness and identify limitations.

## Implementation Details

### 1. Dataset Engineering

Implemented a reproducible data pipeline in `src/build_dataset.py` to construct and validate the automotive Q&A dataset.

**Dataset composition**

- 49 Q&A examples based on NHTSA tire-safety guidance.
- 184 LLM-drafted Q&A examples covering 46 other automotive topics.
- 233 total examples organized into 66 topic groups.

**Data quality and leakage prevention**

- Removed duplicate questions.
- Stored 26 reviewer corrections in `data/overrides.json`.
- Applied corrections using exact question text so they persist across dataset rebuilds.
- Split data by topic group to prevent the same group from appearing in multiple splits.
- Added assertions to validate split sizes and detect group overlap.

| Split | Examples | Purpose |
|---|---:|---|
| Training | 162 | Fine-tuning |
| Validation | 24 | Monitoring training and model selection |
| Test | 47 | Held-out evaluation |
| **Total** | **233** | |

### 2. Manual Dataset Review

Reviewed reference answers to identify errors and improve the quality of the evaluation data.

| Review activity | Examples reviewed | Corrections |
|---|---:|---:|
| Training sample | 50 | 16 |
| Held-out test references | 47 | 11 |
| **Total reviewed** | **97** | **27 review events** |

The project records 26 persistent corrections in `data/overrides.json`. The table counts correction events during review; a correction event is not necessarily a unique override entry.

The original dataset and correction records are kept separate so corrections can be reapplied reproducibly.

### 3. Baseline and Few-Shot Prompting

Implemented `notebooks/baseline_eval.ipynb` to evaluate the base and prompted configurations on the held-out test set.

The prompted configuration uses a system prompt and three examples to guide the model toward concise, task-appropriate automotive answers without updating its weights.

This establishes a comparison for determining whether fine-tuning offers an advantage over prompt engineering.

### 4. QLoRA Fine-Tuning

Implemented `notebooks/finetune_qlora.ipynb` to adapt Qwen2.5-1.5B-Instruct using parameter-efficient fine-tuning.

**Training configuration**

| Parameter | Setting |
|---|---|
| Quantization | 4-bit NF4 |
| Adaptation method | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Training epochs | 3 |
| Learning rate | 2e-4 |
| Hardware | Google Colab T4 GPU |

**Implementation stack**

- Hugging Face Transformers
- PEFT for LoRA adapters
- TRL for supervised fine-tuning
- bitsandbytes for quantization
- Datasets for data handling

Training and validation losses were monitored across epochs. Validation loss reached its best observed value at epoch 2 before increasing slightly at epoch 3, suggesting possible overfitting.

The notebook saves the trained adapter and prediction files in its execution environment. Adapter weights are not included in the repository.

### 5. Blind Human Evaluation

To assess answer quality beyond lexical similarity, 20 questions were sampled from the held-out test set.

Answers from all three configurations were shuffled and presented without model labels. Each answer was scored for correctness and safety.

| Score | Definition |
|---:|---|
| 2 | Correct and safe |
| 1 | Partly correct or vague |
| 0 | Incorrect or unsafe |

The evaluation was conducted by one rater who also authored the dataset. The findings are preliminary and do not establish statistical significance or general performance across automotive questions.

## Results

### Automatic Evaluation: ROUGE-L

All configurations were evaluated on the same 47 held-out questions.

| Model | ROUGE-L | Average answer length |
|---|---:|---:|
| Base | 0.098 | 147 words |
| Few-shot prompted | 0.161 | 28 words |
| QLoRA fine-tuned | 0.183 | 26 words |

The fine-tuned model achieved the highest ROUGE-L score, an absolute increase of 0.085 over the base model.

However, ROUGE-L measures lexical overlap with reference answers. It does not directly measure factual accuracy, completeness, or safety.

### Blind Human Evaluation

The human evaluation covered 20 questions, with a maximum score of 40 per model.

| Model | Total score | Average score | Score 2 | Score 1 | Score 0 |
|---|---:|---:|---:|---:|---:|
| Base | 26/40 | 1.30 | 10 | 6 | 4 |
| Few-shot prompted | 22/40 | 1.10 | 7 | 8 | 5 |
| QLoRA fine-tuned | 22/40 | 1.10 | 7 | 8 | 5 |

The fine-tuned model tied with the prompted model and scored below the base model in this sample.

The four-point difference between the base and the other models is descriptive. Given the small sample and single-rater setup, it should not be interpreted as a statistically significant difference.

### What the Results Suggest

**1. Text similarity and correctness are different objectives.**

Fine-tuning improved ROUGE-L, but human evaluation did not show a corresponding improvement in correctness.

**2. Concise answers can omit important information.**

The base model generated substantially longer responses. In the reviewed sample, the fine-tuned model was weaker on procedural questions involving jump-starting, vehicle storage, and brake care, where omitted steps can matter.

**3. One safety-relevant response improved.**

When asked whether a tire-pressure warning light could be ignored, the base and prompted models answered yes, while the fine-tuned model advised against ignoring it and recommended inspection. This is a qualitative observation, not evidence of an overall safety improvement.

**4. More training is not automatically better.**

With 162 training examples, the experiment did not demonstrate that fine-tuning provided more reliable automotive knowledge. It may have primarily influenced answer style.

These findings motivate testing retrieval over trusted service documentation as an alternative or complement to fine-tuning.

## Limitations

- **Small dataset:** Only 233 examples were available, including 162 training examples.
- **Limited evaluation:** The test set contains 47 questions, with blind human scoring covering 20.
- **Single rater:** No independent evaluation or inter-rater agreement measurement was available.
- **Partially verified data:** The 184 LLM-drafted examples were only partly verified. The 49 NHTSA-based tire examples rely on US-specific guidance.
- **Metric limitations:** ROUGE-L does not establish factual correctness, procedural completeness, or safety.
- **Limited generalizability:** Results do not establish performance across different vehicle manufacturers, models, or service manuals.
- **Latency not reported:** Comparisons were not sufficiently consistent because base-model answers hit the token limit and the fine-tuned adapter was evaluated without merging its weights.

The results should be interpreted as an exploratory comparison under a specific experimental setup.

## Repository Structure

```text
auto-service-llm/
├── data/
│   ├── raw/                    # Source passages, catalog, initial Q&A
│   ├── batches/                # LLM-drafted Q&A by topic
│   ├── overrides.json          # Persistent reviewer corrections
│   └── processed/              # Train, validation, test JSONL
├── notebooks/
│   ├── baseline_eval.ipynb     # Base vs. prompted evaluation
│   └── finetune_qlora.ipynb    # QLoRA fine-tuning and evaluation
├── results/                    # Predictions, summaries, human scores
├── src/
│   └── build_dataset.py        # Dataset construction and validation
├── RESULTS.md                  # Detailed experimental report
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

### Prerequisites

- Python 3
- Google Colab account for GPU-based experiments
- Access to a T4 GPU runtime for the documented training setup

### Step 1: Rebuild the Dataset

Run from the repository root:

```bash
python src/build_dataset.py
```

The script reports split sizes, the number of applied overrides, and the result of topic-group leakage checks.

Expected split sizes:

```text
Train:      162
Validation:  24
Test:        47
Total:      233
```

### Step 2: Run Baseline Evaluation

1. Open `notebooks/baseline_eval.ipynb` in Google Colab.
2. Select a T4 GPU runtime.
3. Run the notebook cells to evaluate the base and prompted models.

### Step 3: Run Fine-Tuning

1. Open `notebooks/finetune_qlora.ipynb` in Google Colab.
2. Select a T4 GPU runtime.
3. Run the notebook cells to fine-tune the model and generate predictions.

Both notebooks clone the repository to access the dataset. Fine-tuning outputs are saved in the notebook execution environment; trained adapter weights are not committed to this repository.

For the detailed experimental setup and results, see [`RESULTS.md`](RESULTS.md).

## Tech Stack

- **Language:** Python
- **LLM:** Qwen2.5-1.5B-Instruct
- **Fine-tuning:** QLoRA, LoRA, NF4 quantization
- **ML frameworks:** Hugging Face Transformers, PEFT, TRL, Datasets
- **Quantization:** bitsandbytes
- **Evaluation:** rouge-score
- **Data processing:** pandas
- **Compute:** Google Colab, NVIDIA T4 GPU

## Future Work

1. **Build a RAG baseline:** Retrieve relevant passages from trusted automotive service manuals and compare grounded answers with all three existing configurations.
2. **Improve procedural coverage:** Add verified examples for multi-step maintenance and safety-critical procedures.
3. **Expand human evaluation:** Increase the sample size and include an independent second rater.
4. **Strengthen evaluation:** Assess factual accuracy, required procedural steps, unsupported claims, and safety-critical omissions.
5. **Revisit fine-tuning:** Evaluate whether more verified examples improve correctness after establishing a reliable retrieval baseline.

