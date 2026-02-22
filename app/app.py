"""
Agriculture QA Assistant — Hugging Face Space.
Uses adapter from agriculture_assistant_lora/ if present; otherwise base TinyLlama only.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# On your Space, adapter + tokenizer files are at the repo root
OUTPUT_DIR = "."

# Lazy-loaded globals
model, tokenizer = None, None
load_error = None
using_adapter = False  # True if we loaded LoRA adapter


def _adapter_available():
    """True if the adapter files exist at OUTPUT_DIR."""
    required = ["adapter_config.json", "adapter_model.safetensors"]
    return all(os.path.isfile(os.path.join(OUTPUT_DIR, f)) for f in required)


def load_model_and_tokenizer():
    global model, tokenizer, load_error, using_adapter
    if model is not None and tokenizer is not None:
        return model, tokenizer
    if load_error is not None:
        raise load_error

    try:
        if _adapter_available():
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(model, OUTPUT_DIR)
            model.eval()
            using_adapter = True
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_ID,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.eval()
            using_adapter = False
        return model, tokenizer
    except Exception as e:
        load_error = e
        raise


def generate_response(model, tokenizer, instruction, max_new_tokens=320):
    prompt = (
        "<|system|>\n"
        "You are a helpful agriculture assistant for farmers. Answer only agriculture-related questions. "
        "Read the question carefully and answer exactly what is asked. Do not copy generic words from instructions; "
        "give specific, relevant advice (e.g. for fodder crops name actual crops like Napier grass, sorghum, maize silage; "
        "for 'without irrigation' suggest mulch and cover crops; for erosion suggest terracing and contour plowing). "
        "Give 2 to 4 numbered steps (1. ... 2. ...). After each number write one or two sentences of practical advice. "
        "Good: '1. Soil testing: Run a soil test to check pH. 2. Tillage: Loosen the top layer.' "
        "Bad: '1. Soil testing: 2. Tillage:' with no explanation. If not about agriculture, say you only answer agriculture questions.\n"
        "<|user|>\n"
        f"{instruction}\n<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.65,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return reply.strip()


def chat(user_input, history=None):
    global model, tokenizer, load_error, using_adapter
    if history is None:
        history = []

    try:
        load_model_and_tokenizer()
    except Exception as e:
        history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": f"❌ Model load failed: {str(e)}"},
        ]
        return history, history

    try:
        reply = generate_response(
            model,
            tokenizer,
            (
                "Address exactly what the question asks. Give 2 to 4 numbered steps with one or two sentences each. "
                "Do not give only a short label. If the question is not about agriculture, say you only answer agriculture questions.\n\n"
                f"Question: {user_input}"
            ),
        )
    except Exception as e:
        reply = f"❌ Generation error: {str(e)}"

    history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply},
    ]
    return history, history


# Build description: show whether adapter is available (will be set after first load)
def get_description():
    if _adapter_available():
        return (
            "## 🌾 Agriculture QA Assistant\n\n"
            "This Space uses **TinyLlama + your fine-tuned LoRA adapter** on "
            "[sowmya14/agriculture_QA](https://huggingface.co/datasets/sowmya14/agriculture_QA).\n\n"
            "Ask any agriculture-related question below."
        )
    return (
        "## 🌾 Agriculture QA Assistant\n\n"
        "Running with the **base** TinyLlama model (adapter folder not found on this Space). "
        "To use your fine-tuned adapter, upload the contents of `agriculture_assistant_lora` "
        "to this repo with paths like `agriculture_assistant_lora/adapter_model.safetensors`. "
        "Training notebook: [GitHub repo](https://github.com/MariamIssah/MLT-1_Domain-Specific-Assistant_via-LLMs-Fine-Tuning).\n\n"
        "Ask any agriculture-related question below."
    )


with gr.Blocks(title="Agriculture Assistant") as demo:
    gr.Markdown(get_description())
    chatbot = gr.Chatbot(label="Chat")
    msg = gr.Textbox(
        placeholder="e.g. How to control aphid infestation in mustard crops?",
        label="Your question",
    )
    submit = gr.Button("Submit")
    clear = gr.Button("Clear")
    state = gr.State([])

    def submit_fn(msg, history):
        if not msg.strip():
            return history, history
        _, new_history = chat(msg, history)
        return new_history, new_history

    submit.click(submit_fn, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])
    msg.submit(submit_fn, [msg, state], [chatbot, state])

demo.launch(server_name="0.0.0.0")
