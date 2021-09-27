"""
Next Sentence Prediction
------------------------
Get the probability of critical sentences following settings.
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction
import torch
from tqdm import tqdm


def next_sentence_prob(model, tokenizer, prompt, continuation):
    # Tokenize the input
    input_ids = tokenizer.encode(prompt, continuation, return_tensors='pt')

    # Predict the next sentence classification logits
    with torch.no_grad():
        outputs = model(input_ids)
        prediction_scores = outputs[0]
    
    # Get the probability of the second sentence following the first
    predicted_probability = torch.softmax(
        prediction_scores, dim=1)[0, 0].item()
    return predicted_probability


def main():
    """main"""

    # Load data
    stimuli = pd.read_csv("data/clean/stimuli_analysed.csv")

    print("Running bert_nsp")

    # Load BERT for NSP
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert = AutoModelForNextSentencePrediction.from_pretrained(
        'bert-base-uncased')
    bert.eval()

    probs = []

    for ix in tqdm(range(len(stimuli))):

        row = stimuli.loc[ix]

        probs.append(next_sentence_prob(
            model=bert,
            tokenizer=bert_tokenizer,
            prompt=row.setting,
            continuation=row.critical))

    stimuli["bert_nsp"] = probs

    stimuli.to_csv("data/clean/stimuli_analysed.csv", index=False)


if __name__ == "__main__":
    print("Running nsp.py")
    main()
