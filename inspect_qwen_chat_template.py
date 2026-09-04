from __future__ import annotations

import argparse

from transformers import AutoTokenizer


def render(tokenizer, enable_thinking: bool | None) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a PacMan game agent. Answer with exactly one action token.",
        },
        {
            "role": "user",
            "content": "Grid:\n#####\n#P .#\n#####\nAllowed output tokens: right, stay",
        },
    ]
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def show_case(name: str, prompt: str) -> None:
    print(f"\n===== {name} =====")
    print(prompt)
    print("===== tail repr =====")
    print(repr(prompt[-120:]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print Qwen chat-template rendering for enable_thinking variants."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(f"Tokenizer for {args.model} has no chat_template")

    print(f"model={args.model}")
    print(f"tokenizer_class={tokenizer.__class__.__name__}")
    print(f"chat_template_chars={len(tokenizer.chat_template)}")

    show_case("enable_thinking=None", render(tokenizer, None))
    show_case("enable_thinking=False", render(tokenizer, False))
    show_case("enable_thinking=True", render(tokenizer, True))


if __name__ == "__main__":
    main()
