# Domain-Specific Assistant via LLM Fine-Tuning

**Course project (60 pts)** — Build a domain-specific assistant by fine-tuning an LLM for a chosen domain. This repository implements an **Agriculture QA assistant** using [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) fine-tuned on [sowmya14/agriculture_QA](https://huggingface.co/datasets/sowmya14/agriculture_QA) with **LoRA (PEFT)**. The pipeline runs end-to-end on **Google Colab** (free GPU) and is deployed on **Hugging Face Spaces**.

---

## Deliverables (Submission)

| Deliverable | Description |
|-------------|-------------|
| **PDF report** | 7–10 pages: links to **code** and **demo video**; dataset & preprocessing; methodology; experiment table; performance metrics; base vs fine-tuned comparison; UI screenshots. |
| **Demo video** | **5–10 minutes**: fine-tuning process, model functionality, user interactions, key insights; clear demonstration of training workflow and **comparisons between base and fine-tuned models**. |
| **GitHub repo** | Well-documented; Jupyter notebook (Colab-ready); README with dataset, methodology, metrics, run instructions, and example conversations. |
| **Colab badge / link** | One-click open in Colab for easy testing. |

**Submit:** A **PDF** that includes links to this repository and the demo video (per assignment instructions).

---

## Project Definition & Domain Alignment (5 pts)

- **Purpose:** A chatbot that answers agriculture-related questions (crops, soil, pests, fertilizing, practices) accurately and in a consistent, in-domain style.
- **Domain:** Agriculture — for farmers, students, and practitioners.
- **Relevance:** Fine-tuning a general-purpose LLM on agriculture QA improves answer quality and relevance for this domain compared to the base model.

---

## Dataset & Preprocessing (10 pts)

- **Source:** [sowmya14/agriculture_QA](https://huggingface.co/datasets/sowmya14/agriculture_QA) (Hugging Face).
- **Size:** ~999 question–answer pairs (within the 1,000–5,000 recommended range).
- **Preprocessing:**  
  - Load with `load_dataset("sowmya14/agriculture_QA")`.  
  - Column detection, text normalization (whitespace, empty/missing handling).  
  - Format into **instruction–response** pairs for causal LM.  
  - **Tokenization:** Base model tokenizer (BPE); sequences truncated/padded to `MAX_SEQ_LENGTH` (512); train/validation split (e.g. 90/10).  
- **Documentation:** The notebook (Sections 3–5) documents loading, cleaning, tokenization, and train/val split.

---

## Methodology

1. **Model:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
2. **Fine-tuning:** Parameter-efficient fine-tuning with **LoRA (PEFT)** on attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`), rank 8, alpha 16. Optional 4-bit loading for Colab (`USE_4BIT = False` by default to avoid bitsandbytes issues).
3. **Training:** Hugging Face `Trainer`; configurable hyperparameters (learning rate 5e-5, batch size 2, gradient accumulation, 3 epochs). **Experiment table** (notebook Section 9) records runs: Val loss, ROUGE-L, BLEU, training time, GPU memory.
4. **Evaluation:** ROUGE, BLEU, perplexity; qualitative checks; **Section 8b** compares **base** vs **fine-tuned** model on the same questions.

---

## Model Fine-Tuning (15 pts)

- **Hyperparameter tuning:** Learning rate, batch size, epochs, LoRA rank documented; experiment table compares runs.
- **Impact:** Validation loss, ROUGE-L, BLEU, and qualitative comparison show improvement over the base model; table in Section 9 documents experiments.

---

## Performance Metrics (5 pts)

- **ROUGE** (e.g. ROUGE-L) and **BLEU** on the evaluation set.
- **Perplexity** from evaluation loss.
- **Qualitative:** In-domain and out-of-domain questions; base vs fine-tuned comparison in Section 8b.

---

## UI Integration (10 pts)

- **Gradio** chat interface: intuitive input, clear output, Submit/Clear.
- **Deployment:** [Hugging Face Space — mariamissah/agriculture-qa-assistant](https://huggingface.co/spaces/mariamissah/agriculture-qa-assistant). The Space loads the fine-tuned LoRA adapter when available; otherwise falls back to the base model.
- **Local/notebook:** Gradio UI in the notebook (Section 10) and in `app/app.py` for local or Space use.

---

## How to Run

### Google Colab (recommended)

1. Open the notebook in Colab (badge or link below).  
2. **Runtime → Change runtime type → GPU** (e.g. T4). Do this before running any cell.  
3. Run all cells in order: install deps → config → load data → preprocessing → tokenization → load model + LoRA → **training** (~15–60 min) → evaluation → experiment table → Gradio UI.  
4. After training, run the **first cell again** (zip adapter) and download `agriculture_assistant_lora.zip` from the Files panel if you want to deploy the adapter on the Space.

**Colab — open notebook:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MariamIssah/MLT-1_Domain-Specific-Assistant_via-LLMs-Fine-Tuning/blob/main/notebooks/domain_assistant_finetune.ipynb)

Direct link: [notebooks/domain_assistant_finetune.ipynb](https://github.com/MariamIssah/MLT-1_Domain-Specific-Assistant_via-LLMs-Fine-Tuning/blob/main/notebooks/domain_assistant_finetune.ipynb)

### Hugging Face Space

- **Live demo:** [mariamissah/agriculture-qa-assistant](https://huggingface.co/spaces/mariamissah/agriculture-qa-assistant)  
- The Space uses `app/app.py`; adapter files at repo root enable the fine-tuned model.

### Local

- `pip install -r requirements.txt` (or `transformers datasets peft accelerate evaluate gradio torch`).  
- After training, save adapter + tokenizer to `./agriculture_assistant_lora`; run `app/app.py` or the notebook’s Gradio cell with that path.

---

## Example Conversations (Impact of Fine-Tuning)

Try in the Space or notebook UI:

- *What are the best practices for preparing soil before planting maize?*
- *How can I control aphid infestation in mustard crops in an eco-friendly way?*
- *How can I improve nitrogen levels in my soil without overusing chemical fertilizers?*
- *What fodder crops are good for feeding dairy cattle in the dry season?*

Compare the same questions with the **base** TinyLlama vs the **fine-tuned** model and describe the differences in your report and demo video.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── domain_assistant_finetune.ipynb    # Full pipeline (Colab)
│   └── domain_assistant_finetune_kaggle.ipynb
└── app/
    ├── app.py          # Gradio app for HF Space / local (adapter or base)
    └── gradio_app.py   # Standalone script (expects agriculture_assistant_lora/)
```

---

## Rubric Alignment (60 pts)

| Criterion | Points | Where it’s covered |
|-----------|--------|--------------------|
| **Project definition & domain** | 5 | README “Project Definition & Domain”; notebook intro. |
| **Dataset & preprocessing** | 10 | Sect. 3–5: load dataset; normalization, cleaning; BPE tokenization; train/val split; documentation in notebook. |
| **Model fine-tuning** | 15 | Sect. 6–7: LoRA (PEFT), TrainingArguments; Sect. 9: experiment table (LR, batch, epochs, val loss, ROUGE-L, BLEU, time, GPU). |
| **Performance metrics** | 5 | Sect. 8: ROUGE, BLEU, perplexity; Sect. 8b: base vs fine-tuned; qualitative testing. |
| **UI integration** | 10 | Gradio chat UI; HF Space deployment; clear instructions in interface. |
| **Code quality & documentation** | 5 | README; notebook markdown and comments; structured repo. |
| **Demo video (5–10 min)** | 10 | Video: code structure, model implementation, functionality, features; base vs fine-tuned; engaging presentation. |

---

## Deliverables Checklist

- [x] Well-documented GitHub repo with Colab-ready notebook and README.
- [x] Colab badge or direct link for easy testing.
- [ ] **PDF report** (7–10 pages) with links to **code** and **demo video**; dataset, methodology, experiment table, metrics, base vs fine-tuned, UI screenshots.
- [ ] **5–10 minute demo video**: fine-tuning process, functionality, user interactions, key insights; training workflow and base vs fine-tuned comparison.

---

## License & Citation

This project is for educational use (MLT-1 assignment). TinyLlama and the dataset have their own licenses; see the linked Hugging Face pages.
