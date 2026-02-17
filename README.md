# Domain-Specific Assistant: Agriculture QA via LLM Fine-Tuning

An **agriculture-focused question-answering assistant** built by fine-tuning [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on the [sowmya14/agriculture_QA](https://huggingface.co/datasets/sowmya14/agriculture_QA) dataset using **LoRA (PEFT)** with PyTorch and Hugging Face Transformers. The project is designed to run end-to-end on **Google Colab** (free GPU).

---

## Project definition and domain

- **Purpose**: Provide accurate, relevant answers to agriculture-related questions (e.g., crops, soil, pests, practices) in a single, easy-to-use assistant.
- **Domain**: Agriculture. The chatbot is specialized for farmers, students, and practitioners who need quick, reliable answers grounded in the training data.
- **Relevance**: A domain-specific model improves answer quality and consistency for agriculture queries compared to a general-purpose chatbot.

---

## Dataset

- **Source**: [sowmya14/agriculture_QA](https://huggingface.co/datasets/sowmya14/agriculture_QA) (Hugging Face Datasets).
- **Size**: ~999 question–answer pairs (within the suggested 1,000–5,000 range).
- **Usage**: Loaded with `load_dataset("sowmya14/agriculture_QA")`. The notebook automatically detects question/answer columns, normalizes text, and formats data into instruction–response pairs for causal language modeling.

---

## Methodology

1. **Preprocessing**: Column detection, text normalization (whitespace, empty/missing handling), instruction–response formatting, tokenization with the base model tokenizer, train/validation split (90/10).
2. **Model**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), loaded in 4-bit (optional) for Colab-friendly memory.
3. **Fine-tuning**: Parameter-efficient fine-tuning with **LoRA** (PEFT) on attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`), rank 8, alpha 16.
4. **Training**: Hugging Face `Trainer` with configurable hyperparameters (learning rate, batch size, gradient accumulation, epochs). An **experiment table** in the notebook (Section 9) records runs with Val loss, ROUGE-L, BLEU, training time, and GPU memory.
5. **Evaluation**: ROUGE, BLEU, perplexity (from eval loss), and qualitative checks. **Section 8b** compares the base (pre-trained) model with the fine-tuned model on the same questions for the report and demo video.

---

## Performance metrics

- **ROUGE** (e.g. ROUGE-L) and **BLEU** on a held-out evaluation set.
- **Perplexity** = exp(eval_loss).
- **Qualitative**: Manual testing on in-domain and out-of-domain questions to assess relevance and appropriateness.

Fill the experiment table in the notebook with your runs and summarize results in your PDF report.

---

## How to run

### Option 1: Google Colab (recommended)

**Checklist — run in this order:**

1. Open the notebook in Colab (badge or link below).
2. **Runtime → Change runtime type → GPU** (e.g. T4). *Do this before running any cell.*
3. **Run all cells from top to bottom** (Runtime → Run all, or run each cell in order):
   - **1** – Install dependencies (`pip install`).
   - **2** – Imports and config.
   - **3** – Load dataset (auto-downloads from Hugging Face).
   - **4** – Preprocessing (instruction–response format).
   - **5** – Tokenization and train/val split.
   - **6** – Load base model and apply LoRA (model auto-downloads).
   - **7** – Training (may take ~15–45 min on T4).
   - **8** – Evaluation (ROUGE, BLEU, perplexity).
   - **8b** – Base vs fine-tuned comparison (for report/demo).
   - **9** – Experiment table (fill with your run results).
   - **10** – Gradio UI (app appears below; use the public URL for your demo).
4. After training, copy **Val loss**, **ROUGE-L**, **BLEU** from Section 8 and paste into the table in Section 9.

**Colab badge** (replace `YOUR_USERNAME` with your GitHub username):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/MLT-1_Domain-Specific-Assistant_via-LLMs-Fine-Tuning/blob/main/notebooks/domain_assistant_finetune.ipynb)

Or use the direct notebook link from your repo:  
`https://github.com/YOUR_USERNAME/MLT-1_Domain-Specific-Assistant_via-LLMs-Fine-Tuning/blob/main/notebooks/domain_assistant_finetune.ipynb` → **Open in Colab**.

### Option 2: Kaggle Notebooks

Use **`notebooks/domain_assistant_finetune_kaggle.ipynb`** (same pipeline, tuned for Kaggle):

1. Upload the notebook to [Kaggle](https://www.kaggle.com) or create a new notebook and paste the contents.
2. **Settings** (right panel) → **Accelerator** → **GPU** (e.g. P100 or T4). Ensure **Internet** is **On**.
3. Run all cells in order. The saved model is written to `/kaggle/working/agriculture_assistant_lora`; use **Save Version** to keep outputs.

### Option 3: Local / standalone UI

- Install: `pip install transformers datasets peft accelerate bitsandbytes evaluate gradio torch`
- After training in the notebook, save the adapter and tokenizer to `./agriculture_assistant_lora`. Then run the Gradio app (see `app/` or the last cells of the notebook) pointing to that path.

---

## Example conversations (impact of fine-tuning)

After fine-tuning, the model should answer agriculture questions more accurately and stay on-topic. Example prompts to try:

- *What are the best practices for soil preparation?*
- *How can I control pests in organic farming?*
- *When is the right time to harvest wheat?*

Compare the same questions with the **base** TinyLlama model (no fine-tuning) vs the **fine-tuned** model and document the differences in your report and demo video.

---

## Repository structure

```
.
├── README.md
├── notebooks/
│   ├── domain_assistant_finetune.ipynb       # Full pipeline for Google Colab
│   └── domain_assistant_finetune_kaggle.ipynb   # Same pipeline for Kaggle
└── app/
    └── gradio_app.py                     # Optional standalone Gradio UI (loads saved adapter)
```

---

## Deliverables checklist

- [x] Well-documented GitHub repo with Colab-ready notebook and README.
- [ ] PDF report (7–10 pages) with links to code and demo video, dataset/preprocessing, methodology, experiment table, metrics, and UI screenshots.
- [ ] 5–10 minute demo video: fine-tuning process, functionality, base vs fine-tuned comparison, and key insights.

---

## Rubric alignment (quick check)

| Criterion | Where it’s covered |
|-----------|--------------------|
| **Project definition & domain** | README “Project definition and domain”; notebook intro. |
| **Dataset & preprocessing** | Section 3–5: load dataset, preprocessing (normalization, instruction–response, tokenization); Section 4 markdown documents tokenization and format. |
| **Model fine-tuning** | Section 6–7: LoRA (PEFT), TrainingArguments; Section 9 experiment table (LR, batch, epochs, LoRA r, val loss, ROUGE-L, BLEU, time, GPU memory). |
| **Performance metrics** | Section 8: ROUGE, BLEU, perplexity; Section 8b: base vs fine-tuned comparison. |
| **UI integration** | Section 10: Gradio UI with instructions in the interface. |
| **Code quality & documentation** | Comments in notebook; README and notebook markdown. |
| **Demo video** | You record 5–10 min covering workflow, metrics, comparison, and UI. |

---

## License and citation

This project is for educational use (MLT-1 assignment). TinyLlama and the dataset have their own licenses; see the linked Hugging Face pages.
