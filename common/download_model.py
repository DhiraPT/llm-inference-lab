import argparse
import struct
import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def export_model(model_id, data_dir, precision):
    # 1. Setup Paths
    # Clean model ID for folder names (e.g. "meta-llama/Llama-3.1-8B" -> "meta-llama_Llama-3.1-8B")
    safe_name = model_id.replace("/", "_")

    # Path for Tokenizer (Python frontend uses this)
    tokenizer_path = os.path.join(data_dir, "tokenizers", safe_name)

    # Path for Binary Weights (C++ backend uses this)
    filename = f"{safe_name}.{precision}.bin"
    model_output_path = os.path.join(data_dir, "models", filename)

    print(f"--- Processing {model_id} ---")

    # ---------------------------------------------------------------------
    # 2. DOWNLOAD & SAVE TOKENIZER
    # ---------------------------------------------------------------------
    if os.path.exists(tokenizer_path):
        print(f"[Skip] Tokenizer already exists at: {tokenizer_path}")
    else:
        print(f"[Download] Fetching tokenizer for: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(tokenizer_path)
        print(f"         Saved to: {tokenizer_path}")

    # ---------------------------------------------------------------------
    # 3. DOWNLOAD & CONVERT WEIGHTS
    # ---------------------------------------------------------------------
    if os.path.exists(model_output_path):
        print(f"[Skip] Model binary already exists at: {model_output_path}")
        return

    print(f"[Load] Loading model weights: {model_id}")
    config = AutoConfig.from_pretrained(model_id)

    # Map precision string to torch dtype
    torch_dtype = torch.float16 if precision == "fp16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    state_dict = model.state_dict()

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    print(f"[Export] Converting to binary: {model_output_path}")

    with open(model_output_path, "wb") as f:
        # --- HEADER ---
        magic_number = 0x4C4C4D41  # "LLMA"

        dim = config.hidden_size
        hidden_dim = config.intermediate_size
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        vocab_size = config.vocab_size
        norm_eps = config.rms_norm_eps
        rope_theta = config.rope_theta

        header = struct.pack(
            "iiiiiiiff",
            magic_number,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            norm_eps,
            rope_theta,
        )
        f.write(header)

        # --- WEIGHTS ---
        def write_tensor(tensor_name):
            if tensor_name not in state_dict:
                raise ValueError(f"Missing tensor: {tensor_name}")
            t = state_dict[tensor_name].to(dtype=torch_dtype)
            f.write(t.detach().numpy().tobytes())

        # 2a. Embeddings
        write_tensor("model.embed_tokens.weight")

        # 2b. Layers
        for i in range(n_layers):
            layer_prefix = f"model.layers.{i}"
            write_tensor(f"{layer_prefix}.input_layernorm.weight")
            write_tensor(f"{layer_prefix}.self_attn.q_proj.weight")
            write_tensor(f"{layer_prefix}.self_attn.k_proj.weight")
            write_tensor(f"{layer_prefix}.self_attn.v_proj.weight")
            write_tensor(f"{layer_prefix}.self_attn.o_proj.weight")
            write_tensor(f"{layer_prefix}.post_attention_layernorm.weight")
            write_tensor(f"{layer_prefix}.mlp.gate_proj.weight")
            write_tensor(f"{layer_prefix}.mlp.down_proj.weight")
            write_tensor(f"{layer_prefix}.mlp.up_proj.weight")

            if (i + 1) % 4 == 0:
                print(f"  Processed layer {i + 1}/{n_layers}...")

        # 2c. Head
        write_tensor("model.norm.weight")
        write_tensor("lm_head.weight")

    print("Done. Setup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare model data for llm-inference-lab.")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B", help="HuggingFace model ID"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory")
    parser.add_argument(
        "--precision", type=str, choices=["fp16", "fp32"], default="fp16", help="Target precision"
    )

    args = parser.parse_args()
    export_model(args.model, args.data_dir, args.precision)
