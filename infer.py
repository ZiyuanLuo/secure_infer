"""Small one-request smoke driver for a configured vLLM installation."""
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    model = os.environ.get("MODEL_PATH", "")
    if not model:
        raise SystemExit("MODEL_PATH is required")
    prompt = os.environ.get("PROMPT", "Explain the runtime in one paragraph.")
    llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")), enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    start = time.perf_counter()
    output = llm.generate([prompt], SamplingParams(max_tokens=int(os.environ.get("MAX_OUTPUT_TOKENS", "64"))))[0]
    elapsed = time.perf_counter() - start
    input_tokens = len(getattr(output, "prompt_token_ids", []) or tokenizer.encode(prompt))
    text = output.outputs[0]
    output_tokens = len(getattr(text, "token_ids", []) or tokenizer.encode(text.text))
    print(f"input_tokens={input_tokens} output_tokens={output_tokens} latency_s={elapsed:.4f}")
    print(text.text)


if __name__ == "__main__":
    main()
