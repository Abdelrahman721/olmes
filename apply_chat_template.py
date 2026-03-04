#!/usr/bin/env python3
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MESSAGES_PAYLOAD_JSON = r"""{"messages": [{"role": "user", "content": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\nHere is the completed function:\n\n```python\n"}]}"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load model from ./model, apply chat_template, and print prompt text."
    )
    parser.add_argument(
        "--model-dir",
        default="./model",
        help="Directory containing Hugging Face model/tokenizer files.",
    )
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Append assistant generation prompt at the end.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()

    payload = json.loads(MESSAGES_PAYLOAD_JSON)
    messages = payload["messages"]

    # Explicitly load both tokenizer and model from ./model as requested.
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=args.add_generation_prompt,
    )
    print("=== Rendered Prompt ===")
    print(rendered_prompt)

    model_inputs = tokenizer(rendered_prompt, return_tensors="pt")
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    
    print("EOS TOKEN: ", tokenizer.eos_token_id)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    completion_ids = generated_ids[0, prompt_length:]
    generated_text = tokenizer.decode(completion_ids)

    print("\n=== Generated Answer ===")
    print(generated_text)


if __name__ == "__main__":
    main()
