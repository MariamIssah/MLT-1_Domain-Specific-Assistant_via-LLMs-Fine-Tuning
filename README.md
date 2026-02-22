# Domain-Specific Assistant via LLM Fine-Tuning

This repository implements an **Agriculture QA assistant** by fine-tuning [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on [sowmya14/agriculture_QA](https://huggingface.co/datasets/sowmya14/agriculture_QA) with **LoRA (PEFT)**. The notebook covers the complete LLM fine-tuning pipeline: data preprocessing, model training with PEFT, and inference demonstration. It is designed to run end-to-end on **Google Colab** with minimal setup.

---

## Project Definition & Domain

- **Purpose:** A chatbot that answers agriculture-related questions (crops, soil, pests, fertilizing, practices) accurately and in a consistent, in-domain style.
- **Domain:** Agriculture — for farmers, students, and practitioners.
- **Relevance:** Fine-tuning a general-purpose LLM on agriculture QA improves answer quality and relevance for this domain compared to the base model.

---

## Dataset

- **Source:** [sowmya14/agriculture_QA](https://huggingface.co/datasets/sowmya14/agriculture_QA) (Hugging Face Datasets Hub).
- **Form:** Question–answer pairs aligned with the agriculture domain; diverse user intents.
- **Size:** ~999 instruction–response pairs (within the 1,000–5,000 range).
- **Preprocessing:** Column detection; text normalization (whitespace, handling missing/empty values); formatting into clear instruction–response templates; sequences fit within the model’s context window (max length 512).
- **Tokenization:** Base model tokenizer (BPE); train/validation split (e.g. 90/10). The notebook (Sections 3–5) documents loading, cleaning, tokenization, and the train/val split.

---

## Fine-Tuning Methodology

- **Base model:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), a modern generative model suitable for efficient fine-tuning on Colab’s free GPU.
- **Method:** Parameter-efficient fine-tuning using **LoRA (Low-Rank Adaptation)** via the `peft` library. LoRA is applied to attention layers; rank 8, alpha 16.
- **Training:** Hugging Face `Trainer`. Hyperparameters: learning rate 5e-5 (within the typical 1e-4 to 5e-5 range), batch size 2 with gradient accumulation, 3 epochs. Compatible with Colab’s free GPU resources.
- **Experiment table:** The notebook (Section 9) includes a table documenting experiments—learning rate, batch size, epochs, validation loss, ROUGE-L, BLEU, training time, and GPU memory usage—so the impact of adjustments is documented.

---

## Performance Metrics

- **BLEU score** and **ROUGE score** (e.g. ROUGE-L) on the evaluation set.
- **Perplexity** (from evaluation loss).
- **Qualitative testing:** Interacting with the fine-tuned model on in-domain and out-of-domain queries; the model answers relevant questions and handles out-of-domain queries appropriately.
- **Base vs fine-tuned comparison:** The notebook (Section 8b) compares the base pre-trained model and the fine-tuned version on the same questions to demonstrate improvement and document the value of fine-tuning.

---

## Steps to Run the Model

### Google Colab (recommended)

1. Open the notebook in Colab using the badge or direct link below.
2. **Runtime → Change runtime type → GPU** (e.g. T4). Do this before running any cell.
3. Run all cells in order: install dependencies → imports and config → load dataset → preprocessing → tokenization and train/val split → load base model and apply LoRA → training → evaluation → experiment table → base vs fine-tuned comparison → Gradio UI.
4. The notebook runs end-to-end with minimal setup; training may take on the order of 15–60 minutes on a free Colab GPU.

**Colab badge / direct link:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MariamIssah/MLT-1_Domain-Specific-Assistant_via-LLMs-Fine-Tuning/blob/main/notebooks/domain_assistant_finetune.ipynb)

Direct link (from repo): [notebooks/domain_assistant_finetune.ipynb](https://github.com/MariamIssah/MLT-1_Domain-Specific-Assistant_via-LLMs-Fine-Tuning/blob/main/notebooks/domain_assistant_finetune.ipynb)  
Alternative (Colab/Drive): [Open in Colab](https://colab.research.google.com/drive/18j4rHyiwunTXiRYAlTaW4OF6FIYjcZrB#scrollTo=a1622f1f)

### Deployed interface

The fine-tuned model is deployed via a **Gradio** web interface so users can interact with it: [Hugging Face Space — mariamissah/agriculture-qa-assistant](https://huggingface.co/spaces/mariamissah/agriculture-qa-assistant). The interface allows users to input queries and receive responses from the customized LLM.

### Local

Install dependencies (e.g. `pip install -r requirements.txt`). After training in the notebook, the adapter and tokenizer are saved; you can run the Gradio app (notebook Section 10 or `app/app.py`) pointing to that path for local inference.

---

## Examples of Conversations (Impact of Fine-Tuning)

Example questions to try in the Space or notebook UI. Compare answers from the **base** TinyLlama and the **fine-tuned** model to see the impact of fine-tuning:

- *What are the best practices for preparing soil before planting maize?*
- *How can I control aphid infestation in mustard crops in an eco-friendly way?*
- *How can I improve nitrogen levels in my soil without overusing chemical fertilizers?*
- *What fodder crops are good for feeding dairy cattle in the dry season?*

---

## Repository Structure

```
.
├── README.md
├── requirements.txt              # Dependencies for local / Colab
├── requirements-spaces.txt       # Minimal deps for Hugging Face Space
├── notebooks/
│   ├── domain_assistant_finetune.ipynb      # Main notebook: full pipeline (Colab)
│   ├── domain_assistant_finetune-first.ipynb # Alternative copy of the pipeline
│   ├── domain-assistant-finetune.ipynb      # Alternate notebook
│   ├── domain_assistant_finetune_kaggle.ipynb # Kaggle-oriented version
│   └── agriculture_assistant_lora/          # Created by training (Section 7)
│       ├── adapter_config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── chat_template.jinja
│       ├── README.md
│       ├── checkpoint-113/
│       ├── checkpoint-226/
│       └── checkpoint-339/
└── app/
    ├── app.py          # Gradio app for Space / local (adapter or base)
    └── gradio_app.py   # Standalone Gradio script (expects saved adapter)
```

The main notebook (`domain_assistant_finetune.ipynb`) contains the complete pipeline: Sections 1–2 (setup, config), 3–5 (dataset, preprocessing, tokenization), 6–7 (model, LoRA, training), 8–9 (evaluation, experiment table), 8b (base vs fine-tuned comparison), 10 (Gradio UI).

**`agriculture_assistant_lora` folder:** This folder is **created when you run training** (Section 7). It is written by `trainer.save_model(OUTPUT_DIR)` and `tokenizer.save_pretrained(OUTPUT_DIR)` and contains the LoRA adapter (`adapter_model.safetensors`, `adapter_config.json`), tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.), and optionally training checkpoints. It is not part of the GitHub repo by default. In Colab, after training, run the first notebook cell again to zip this folder, then download the zip and upload its contents to your Hugging Face Space (or use it locally with `app/app.py`).

---

**Submission:** Provide a PDF report that includes links to this repository and the demo video. The demo video (5–10 minutes) should showcase the fine-tuning process, the model’s functionality, user interactions, and key insights, including clear demonstrations of the training workflow and comparisons between the base and fine-tuned models.

**Demo video:** [YouTube](https://youtu.be/NbKSeckC1hc)
