import argparse
import subprocess
import os
import sys
from transformers import AutoTokenizer


def run_inference(tokenizer_path, model_path, engine_path, prompt, max_tokens):
    # Validation
    if not os.path.exists(engine_path):
        print(f"[Error] Engine not found: {engine_path}")
        print("Did you run 'cmake --build build'?")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"[Error] Model not found: {model_path}")
        sys.exit(1)

    # Tokenize (Python Side)
    print(f"[Python] Loading tokenizer: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"[Error] Failed to load tokenizer: {e}")
        sys.exit(1)

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_str = " ".join(map(str, input_ids))

    print(f"[Python] Prompt: '{prompt}'")
    print(f"[Python] Input IDs: {input_ids}")
    print("-" * 40)

    # Execution (C++ Backend)
    cmd = [
        engine_path,
        "--model",
        model_path,
        "--tokens",
        input_str,
        "--max_tokens",
        str(max_tokens),
    ]

    try:
        # Run C++ binary
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Detokenize (Python Side)
        raw_output = result.stdout.strip()
        if raw_output:
            try:
                # Expecting space-separated integers from C++
                output_ids = [int(x) for x in raw_output.split()]
                decoded_text = tokenizer.decode(output_ids)
                print(f"[Python] Output:\n{decoded_text}")
            except ValueError:
                print(f"[Python] Raw output from engine (parsing failed):\n{raw_output}")
        else:
            print("[Python] Warning: Engine returned no output.")

    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Engine crashed (Exit Code {e.returncode})")
        print(f"STDERR: {e.stderr}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max_tokens", type=int, default=32)

    args = parser.parse_args()
    run_inference(args.tokenizer_path, args.model_path, args.engine, args.prompt, args.max_tokens)
