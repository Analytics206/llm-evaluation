from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def calculate_bleu(reference, generated, n_gram=4):
    """
    Calculate BLEU score with smoothing and adjustable n-gram order.
    """
    # Tokenize the reference and generated text
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()

    # Use SmoothingFunction to handle cases with no n-gram overlaps
    smoothing_function = SmoothingFunction().method1

    # Calculate BLEU score with the specified n-gram order
    bleu_score = sentence_bleu(
        reference_tokens,
        generated_tokens,
        weights=(1.0 / n_gram,) * n_gram,  # Equal weights for n-grams
        smoothing_function=smoothing_function
    )

    return bleu_score

def calculate_rouge(reference, generated):
    """
    Calculate ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)