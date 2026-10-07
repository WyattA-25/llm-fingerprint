# Who Wrote It? Identifying LLM Families from Their Responses

CMPSC 448 midterm project. Classifies which LLM family (Claude, DeepSeek, GPT, Gemini,
Grok, Llama, Mistral, Qwen) wrote a response, using a CNN and a BiLSTM, and studies
why it works (RQ1 to RQ4).

## Data
Source: [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)
(Chatbot Arena battles, 2025). Each battle is one real user prompt answered by two anonymous
models. We keep single-turn English battles, map model names to families, and balance to
2,000 responses per family (16,000 rows). See the dataset card for the license.

| Step | Script | Output |
|---|---|---|
| Build dataset | `prepare_data.py` | `data/dataset.parquet` |
| Clean, mask names, split | `python -m src.preprocess` | `data/processed.parquet` |

Preprocessing strips `<think>` traces, masks vendor/model names in prompts and responses
(so the model can't just read "I'm Claude"), and splits 70/15/15 **grouped by prompt**
so no prompt appears in two splits.

## Models (`src/models.py`)
- **CNN**: character-level, parallel convolutions (widths 3/5/7), second conv stage, max+mean pooling.
- **RNN**: word-level BiLSTM with attention pooling.
- Baselines: TF-IDF (word 1-2 + char 2-5 grams) + logistic regression; gradient boosting on 27 stylometric features.

## Experiments
| RQ | Script | Question |
|---|---|---|
| 1 | `experiments/rq1.py` | Identify family from output only |
| 2 | `experiments/rq2.py` | Input only vs output only vs both; matched-prompt and timestamp controls |
| 3 | `experiments/rq3.py` | Cross-domain transfer with size-matched in-domain comparison |
| 4 | `experiments/rq4.py` | Stylometric profiles, feature importance, attention, ablations |

Results go to `results/*.json`, figures to `results/figures/`.

## Running
```
pip install -r requirements.txt
python prepare_data.py          # downloads from Hugging Face, ~1.6 GB cache
python run_all.py               # everything; a GPU is strongly recommended
python run_all.py --smoke       # synthetic data, tiny settings, verifies the pipeline in minutes
```
Scripts pick CUDA, then Apple MPS, then CPU automatically. All runs are seeded.

Every finished training run is cached in `results/cache/` (keyed by config, seed, and a
hash of the exact data), so if the machine restarts, rerunning `python run_all.py` skips
completed runs and resumes. `python run_all.py --only rq3 rq4` runs selected steps.

## Report
The final report is [report/report.pdf](report/report.pdf).
