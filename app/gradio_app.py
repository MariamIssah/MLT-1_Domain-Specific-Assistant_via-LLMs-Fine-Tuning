"""
Standalone Gradio UI for the Agriculture QA assistant.
Run after fine-tuning: expects a saved PEFT adapter and tokenizer in OUTPUT_DIR.

Usage:
  pip install transformers peft accelerate gradio torch
  python gradio_app.py

Set OUTPUT_DIR to the path where trainer.save_model() and tokenizer.save_pretrained() were called.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# Path to the saved fine-tuned adapter and tokenizer (e.g. from Colab download or local training)
OUTPUT_DIR = "./agriculture_assistant_lora"
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, instruction, max_new_tokens=128):
    prompt = (
        "<|system|>\nYou are an agriculture assistant.\n<|user|>\n"
        f"{instruction}\n<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return reply.strip()


def chat(user_input, history=None):
    if history is None:
        history = []
    instruction = (
        "You are an agriculture assistant. Answer the following question.\n\n"
        f"Question: {user_input}"
    )
    reply = generate_response(model, tokenizer, instruction)
    history.append((user_input, reply))
    return history, history


if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    with gr.Blocks(title="Agriculture Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## Agriculture QA Assistant\n\n"
            "Ask any agriculture-related question. The model is fine-tuned on "
            "[sowmya14/agriculture_QA](https://huggingface.co/datasets/sowmya14/agriculture_QA)."
        )
        chatbot = gr.Chatbot(label="Chat")
        msg = gr.Textbox(
            placeholder="e.g. What are the best practices for soil preparation?",
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

    demo.launch(share=False)
