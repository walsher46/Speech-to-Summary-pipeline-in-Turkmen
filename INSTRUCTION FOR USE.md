# Turkmen Speech-to-Summary Pipeline

A cascaded neural pipeline for automatic speech recognition (ASR)
and abstractive text summarization for the Turkmen language.
  
The notebook `pipeline_turk.ipynb` covers three independent run modes plus a custom-input block.

---

## Project Structure

```
project root/
├── pipeline_turk.ipynb          # Main pipeline notebook (this file)
│
├── turk_asr/                    # ASR model & data
│   ├── tk/                      # Datasets (train, test, eval, etc.)
│   │   ├── clips/               # .wav audio clips
│   │   ├── train.tsv
│   │   ├── test.tsv
│   │   └── dev.tsv
│   ├── output/
│   │   └── model_35epochs/      # Fine-tuned ASR model (checkpoint)
│   │       ├── adapter_config.json
│   │       ├── adapter_model.*
│   │       ├── lm_head.pt
│   │       ├── vocab.json
│   │       └── tokenizer_config.json
│   └── cache/                   # Hugging Face cache for ASR
│
└── turk_sum/                    # Summarization model & data
    ├── data/                    # Dataset
    │   ├── raw/                 # Raw source data
    │   └── processed/           # Pre-processed data ready for training
    ├── models/
    │   └── mbart_finetune/
    │       └── final/           # Fine-tuned mBART-50 model (checkpoint)
    │           ├── config.json
    │           ├── tokenizer_config.json
    │           └── pytorch_model.bin / model.safetensors
    └── cache/                   # Hugging Face cache for summarization
```

---

## Google Drive Paths (as used in the notebook)

| Variable | Path |
|---|---|
| `ASR_MODEL_DIR` | `/content/drive/MyDrive/Final_models/turk_asr/output/model_35epochs` |
| `ASR_CLIPS_DIR` | `/content/drive/MyDrive/Final_models/turk_asr/tk/clips` |
| `ASR_TSV_PATH` | `/content/drive/MyDrive/Final_models/turk_asr/tk/test.tsv` |
| `SUM_MODEL_DIR` | `/content/drive/MyDrive/Final_models/turk_sum/models/mbart_finetune/final` |
| `OUTPUT_DIR` | `/content/drive/MyDrive/turkmen_pipeline_results` |

---

## How to Use

### 1. Setup

Open `pipeline_turk.ipynb` in Google Colab and mount your Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Run the **install cell** at the top, then **Block 0** (configuration) to set all paths.

### 2. Load Models

Run **Block 2A** to load the ASR model and **Block 2B** to load the summarization model.  
Both blocks must be run before any of the test blocks.

If a model checkpoint is not found, the block will automatically download the base model to the configured directory. Replace the weights with your fine-tuned checkpoint and re-run.

### 3. Run Modes

All three blocks below are **independent** — rerun any of them without reloading models.

| Block | Description |
|---|---|
| **Block 1 — Pipeline** | Loads dataset samples → ASR → Summarization. Saves results to `pipeline_results.json`. |
| **Block 2 — ASR Only** | Loads dataset samples → ASR transcription + metrics (WER, CER, BLEU, chrF2). Saves to `asr_only_results.json`. |
| **Block 3 — Summarization Only** | Runs summarization on the `SUM_TEST_SAMPLES` list defined in Block 0. Saves to `sum_only_results.json`. |

### 4. Dataset Selection (Blocks 1 & 2)

Control which samples are used by setting these variables in **Block 0**:

```python
ASR_SELECTION_MODE  = "random"   # "random" or "manual"
ASR_RANDOM_N        = 5          # number of random samples
ASR_MANUAL_INDICES  = [0, 1, 2]  # for manual mode: 0-based indices in TSV
ASR_RANDOM_SEED     = 42         # None = new random seed each run
```

Only `.wav` files that physically exist in `ASR_CLIPS_DIR` are accepted.  
Other formats (`.mp3`, `.ogg`, etc.) are silently skipped.

### 5. Custom Input (Block 4)

Block 4 lets you provide your own audio files or texts without modifying the dataset.

Set `CUSTOM_MODE` to one of:

| Mode | Input | Output |
|---|---|---|
| `"pipeline"` | List of `.wav` paths | ASR transcription + summary |
| `"asr"` | List of `.wav` paths | ASR transcription + metrics |
| `"sum"` | List of text strings | Summarization + metrics |

You can also set `USE_DATASET_FOR_CUSTOM = True` to pull samples from the configured dataset instead of providing manual paths.

Example for audio upload in Colab:

```python
from google.colab import files
uploaded = files.upload()               # select .wav file(s)
CUSTOM_AUDIO_FILES = list(uploaded.keys())
CUSTOM_MODE = "pipeline"
run_custom()
```

---

## Output Files

All results are saved as JSON to `OUTPUT_DIR`:

| File | Contents |
|---|---|
| `pipeline_results.json` | ASR + summary for each sample, with optional metrics |
| `asr_only_results.json` | Transcriptions + ASR metrics (raw and normalized) |
| `sum_only_results.json` | Summaries + ROUGE / BERTScore metrics |
| `custom_results.json` | Output from Block 4 custom runs |

---

## Models

| Model | Base | Fine-tuned for |
|---|---|---|
| ASR | `facebook/mms-1b-all` (MMS, `tur` language) | Turkmen speech recognition via LoRA |
| Summarization | `facebook/mbart-large-50` | Turkmen text summarization (`tr_TR`) |

The ASR model uses a LoRA adapter (`adapter_config.json` + `adapter_model.*`) and a saved `lm_head.pt` with the extended Turkmen vocabulary.

---

## Requirements

```
transformers
peft
accelerate
datasets
evaluate
jiwer
sacrebleu
soundfile
librosa
rouge-score
bert-score
```

Install with:

```bash
pip install transformers peft accelerate datasets evaluate jiwer sacrebleu soundfile librosa rouge-score bert-score
```
