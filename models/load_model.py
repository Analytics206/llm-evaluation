from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name="microsoft/DialoGPT-medium"):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device