# DTS406 Coursework

This repository contains the implementation for the DTS406 coursework. It is organised into two independent parts:

- **Part One:** Response selection in dialogue systems
- **Part Two:** Document summarisation with an instruction-tuned model and intra-document RAG

The code is written to use local data and local model folders. Large datasets, model weights, checkpoints, and generated outputs are intentionally excluded from Git.

## Repository Structure

```text
DTS406_CW2
|-- assignment.md
|-- part_one
|   |-- experiment
|   |   |-- config.py
|   |   |-- dataset.py
|   |   |-- train.py
|   |   |-- evaluate.py
|   |   `-- main.py
|   `-- utils
|       |-- data_prepare.py
|       |-- metrics.py
|       |-- plot_results.py
|       `-- ...
|-- part_two
|   |-- experiment
|   |   |-- config.py
|   |   |-- dataset.py
|   |   |-- generate.py
|   |   |-- evaluate.py
|   |   `-- main.py
|   `-- utils
|       |-- data_prepare.py
|       |-- rag_utils.py
|       |-- metrics.py
|       |-- plot_results.py
|       `-- ...
|-- data
|   |-- part_one
|   `-- part_two
|-- models
`-- outputs
```

## Part One: Response Selection

Part One implements a response selection system using the Ubuntu Dialogue Corpus v2.0.

### Method

- Dataset: Ubuntu Dialogue Corpus v2.0
- Base model: `microsoft/MiniLM-L12-H384-uncased`
- Architecture: cross-encoder
- Training objective: binary classification over context-response pairs
- Evaluation format: 1-of-10 candidate ranking
- Metrics:
  - MRR
  - Recall@1
  - Recall@2
  - Recall@5

### Expected Input Files

Place the raw Ubuntu files here:

```text
data/part_one/raw/train.csv
data/part_one/raw/valid.csv
data/part_one/raw/test.csv
```

Expected raw format:

```text
train.csv:
Context, Utterance, Label

valid.csv / test.csv:
Context, Ground Truth Utterance, Distractor_0, ..., Distractor_8
```

Place the MiniLM model here:

```text
models/MiniLM-L12-H384-uncased
```

### Run

```powershell
python .\part_one\experiment\main.py
```

This runs data preparation, training, validation, test evaluation, and prediction export.

### Outputs

Part One outputs are saved under:

```text
outputs/part_one
```

Important files:

```text
outputs/part_one/checkpoints/best
outputs/part_one/logs/run.log
outputs/part_one/results/validation_metrics.csv
outputs/part_one/results/test_metrics.json
outputs/part_one/results/test_predictions.csv
outputs/part_one/results/error_examples.csv
outputs/part_one/results/data_statistics.json
```

To generate figures:

```powershell
python .\part_one\utils\plot_results.py
```

Figures are saved to:

```text
outputs/part_one/results/figures
```

## Part Two: Document Summarisation

Part Two implements document summarisation using FLAN-T5 and intra-document RAG on CNN/DailyMail stories.

### Method

- Dataset: CNN/DailyMail from `ccdv/cnn_dailymail`
- Generator: `google/flan-t5-base`
- Retriever: `sentence-transformers/all-MiniLM-L6-v2`
- Baseline method: first-512-token article summarisation
- RAG method:
  - sentence-aware chunking
  - first three article sentences as retrieval query
  - top-2 retrieved chunks
  - selected chunks ordered by original article position
- Metrics:
  - ROUGE-1
  - ROUGE-2
  - ROUGE-L
  - corpus BLEU

### Expected Input Files

Place the CNN/DailyMail story archives here:

```text
data/part_two/raw/cnn_stories.tgz
data/part_two/raw/dailymail_stories.tgz
```

Place the models here:

```text
models/flan-t5-base
models/all-MiniLM-L6-v2
```

### Run

```powershell
python .\part_two\experiment\main.py
```

This runs raw story parsing, sampling, RAG retrieval, baseline generation, RAG generation, and metric evaluation.

### Outputs

Part Two outputs are saved under:

```text
outputs/part_two
```

Important files:

```text
outputs/part_two/logs/run.log
outputs/part_two/results/generation_results.csv
outputs/part_two/results/rouge_results.csv
outputs/part_two/results/final_metrics.json
outputs/part_two/results/qualitative_examples.csv
outputs/part_two/results/data_statistics.json
```

To generate figures:

```powershell
python .\part_two\utils\plot_results.py
```

Figures are saved to:

```text
outputs/part_two/results/figures
```

## Notes

- The project uses fixed random seed `42` for sampling and candidate shuffling.
- Part One trains on `train_100k` and evaluates on `valid_5k` and `test_5k`.
- Part Two samples 1,500 stories first, filters examples whose RAG prompt exceeds 512 FLAN-T5 tokens, and keeps 1,000 valid examples.
- Generated data, model weights, checkpoints, and result outputs are not tracked by Git.
