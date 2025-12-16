import argparse
import subprocess
import os
import sys
from transformers import AutoTokenizer


def run_inference(engine_path, tokenizer_dir, model_dir, prompt, max_tokens, temperature=0.0):
    if not os.path.exists(engine_path):
        print(f"[Error] Engine not found: {engine_path}")
        print("Did you run 'cmake --build build'?")
        sys.exit(1)
    if not os.path.isdir(tokenizer_dir):
        print(f"[Error] Tokenizer directory not found: {tokenizer_dir}")
        sys.exit(1)
    if not os.path.isdir(model_dir):
        print(f"[Error] Model directory not found: {model_dir}")
        sys.exit(1)
    required_files = ["config.json", "model.bin"]
    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            print(f"[Error] Missing {f} in model directory: {model_dir}")
            sys.exit(1)

    # Tokenize (Python Side)
    print(f"[Python] Loading tokenizer: {tokenizer_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    except Exception as e:
        print(f"[Error] Failed to load tokenizer: {e}")
        sys.exit(1)

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_str = " ".join(map(str, input_ids))

    print(f"[Python] Prompt: '{prompt}'")
    print(f"[Python] Input IDs ({len(input_ids)}): {input_ids}")
    print("-" * 60)
    print(f"[Python] Output Streaming:\n")

    # Execution (C++ Backend)
    cmd = [
        engine_path,
        "--model",
        model_dir,
        "--tokens",
        input_str,
        "--max_tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
    ]

    # Run C++ binary
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)

    output_ids = []

    # Stream output tokens
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.strip()
                try:
                    current_tokens = [int(x) for x in line.split()]
                    output_ids.extend(current_tokens)
                    text_chunk = tokenizer.decode(current_tokens, skip_special_tokens=True)
                    print(text_chunk, end="", flush=True)
                except ValueError:
                    pass
    except KeyboardInterrupt:
        print("\n[Python] Interrupted by user.")
        process.terminate()
    finally:
        process.stdout.close()
        return_code = process.wait()

    print("\n" + "-" * 60)
    if return_code != 0:
        print(f"[Python] Engine exited with error code {return_code}")
    else:
        full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        # print(f"\n[Python] Full Sequence:\n{full_text}")
        print("[Python] Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()
    run_inference(
        args.engine,
        args.tokenizer_dir,
        args.model_dir,
        args.prompt,
        args.max_tokens,
        args.temperature,
    )
