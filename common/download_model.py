import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def export_model(model_id, data_dir, precision):
    # Clean model ID for folder names (e.g. "meta-llama/Llama-3.1-8B" -> "meta-llama_Llama-3.1-8B")
    safe_name = model_id.replace("/", "_")

    tokenizer_dir = os.path.join(data_dir, "tokenizers", safe_name)
    model_dir = os.path.join(data_dir, "models", safe_name)
    os.makedirs(model_dir, exist_ok=True)
    config_output_path = os.path.join(model_dir, "config.json")
    weights_output_path = os.path.join(model_dir, f"model.{precision}.bin")

    print(f"--- Processing {model_id} ---")

    # Download and save tokenizer
    if os.path.exists(tokenizer_dir):
        print(f"[Skip] Tokenizer already exists at: {tokenizer_dir}")
    else:
        print(f"[Download] Fetching tokenizer for: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"         Saved to: {tokenizer_dir}")

    # Download and save config
    print(f"[Config] Saving config to: {config_output_path}")
    config = AutoConfig.from_pretrained(model_id)
    config.save_pretrained(model_dir)

    # Download model weights and export to custom binary format
    if os.path.exists(weights_output_path):
        print(f"[Skip] Model binary already exists at: {weights_output_path}")
        return

    print(f"[Load] Loading model weights: {model_id}")

    torch_dtype = torch.float16 if precision == "fp16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map="cpu",
    )

    state_dict = model.state_dict()

    with open(weights_output_path, "wb") as f:

        def write_tensor(tensor_name):
            if tensor_name not in state_dict:
                raise ValueError(f"Missing tensor: {tensor_name}")
            t = state_dict[tensor_name].to(dtype=torch_dtype)
            f.write(t.detach().numpy().tobytes())

        # Embeddings
        write_tensor("model.embed_tokens.weight")

        # Layers
        n_layers = config.num_hidden_layers
        for i in range(n_layers):
            layer_prefix = f"model.layers.{i}"
            # Attention
            write_tensor(f"{layer_prefix}.input_layernorm.weight")
            write_tensor(f"{layer_prefix}.self_attn.q_proj.weight")
            write_tensor(f"{layer_prefix}.self_attn.k_proj.weight")
            write_tensor(f"{layer_prefix}.self_attn.v_proj.weight")
            write_tensor(f"{layer_prefix}.self_attn.o_proj.weight")
            write_tensor(f"{layer_prefix}.post_attention_layernorm.weight")

            # MLP
            write_tensor(f"{layer_prefix}.mlp.gate_proj.weight")
            write_tensor(f"{layer_prefix}.mlp.down_proj.weight")
            write_tensor(f"{layer_prefix}.mlp.up_proj.weight")

            print(f"  Processed layer {i + 1}/{n_layers}...")

        # Head
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
