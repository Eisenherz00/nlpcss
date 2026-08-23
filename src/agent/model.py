"""Model loading for the survey-item agent."""

import sys



def load_model(model_name: str):
    print(f"Loading model: {model_name}")

    if sys.platform == "darwin":
        from mlx_lm import load

        model, tokenizer = load(model_name)
        print("Model loaded on device: Apple Silicon (MLX)")
        return tokenizer, model
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print(f"Model loaded on device: {model.device}")
        return tokenizer, model