import json
from models.load_model import load_model
from evaluation.metrics import calculate_bleu, calculate_rouge

def evaluate_model(test_data_path="data/test_data.json", output_path="results/evaluation_results.json"):
    # Load the model, tokenizer, and device
    model, tokenizer, device = load_model()

    # Load the test data
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    results = {
        "avg_bleu": 0,
        "avg_rouge1": 0,
        "avg_rougeL": 0,
        "samples": []
    }

    # Evaluate each sample
    for sample in test_data:
        input_text = sample["input"]
        reference = sample["response"]

        # Tokenize the input and generate attention mask
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate a response with attention mask
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            pad_token_id=tokenizer.eos_token_id  # Use eos_token_id as pad_token_id
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate metrics
        bleu_score = calculate_bleu(reference, generated)
        rouge_scores = calculate_rouge(reference, generated)

        # Store results
        results["samples"].append({
            "input": input_text,
            "reference": reference,
            "generated": generated,
            "bleu": bleu_score,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure
        })

        # Update averages
        results["avg_bleu"] += bleu_score
        results["avg_rouge1"] += rouge_scores["rouge1"].fmeasure
        results["avg_rougeL"] += rouge_scores["rougeL"].fmeasure

    # Compute averages
    num_samples = len(test_data)
    results["avg_bleu"] /= num_samples
    results["avg_rouge1"] /= num_samples
    results["avg_rougeL"] /= num_samples

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {output_path}")