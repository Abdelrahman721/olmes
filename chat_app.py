import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./model_clean"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model (this may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded.")


def _normalize(history: list[dict]) -> list[dict]:
    """Gradio may return content as a list of parts; flatten to plain strings."""
    out = []
    for msg in history:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                p if isinstance(p, str) else p.get("text", "") for p in content
            )
        out.append({"role": msg["role"], "content": str(content)})
    return out


def generate(history: list[dict]) -> str:
    text = tokenizer.apply_chat_template(
        _normalize(history),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def chat(message: str, history: list[dict]) -> gr.ChatMessage:
    history.append({"role": "user", "content": message})
    reply = generate(history)
    history.append({"role": "assistant", "content": reply})
    return history


with gr.Blocks(title="Qwen3 Chat") as demo:
    gr.Markdown("## Qwen3 Chat")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type a message...", show_label=False)
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])
    clear.click(lambda: [], None, [chatbot])

demo.launch(server_name="0.0.0.0", server_port=7860)
