"""
Next Sentence Prediction
------------------------
Get the probability of critical sentences following settings.
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel


def find_sublist_index(list, sublist):
    """Find the first occurence of sublist in list.
    Return the start and end indices of sublist in list

    h/t GPT-3-codex for writing this."""

    for i in range(len(list)):
        if list[i] == sublist[0] and list[i:i+len(sublist)] == sublist:
            return i, i+len(sublist)
    return None


def get_embedding(model, tokenizer, sentence, target):
    """Get a token embedding for target in sentence"""

    sentence_enc = tokenizer.encode(sentence, return_tensors="pt")

    target_enc = tokenizer.encode(target, return_tensors="pt",
                                  add_special_tokens=False)

    target_inds = find_sublist_index(
        sentence_enc[0].tolist(), target_enc[0].tolist())

    with torch.no_grad():
        output = model(sentence_enc)
        hidden_states = output.hidden_states

    # second last layer, batch 0
    layer_2l = hidden_states[-2][0]
    token_vecs = layer_2l[target_inds[0]:target_inds[1]]
    embedding = torch.mean(token_vecs, dim=0)
    return embedding


def main():
    """main"""

    # Load BERT
    bert = BertModel.from_pretrained('bert-large-cased',
                                     output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bert.eval()

    # Load data
    stimuli = pd.read_csv("data/clean/stimuli_analysed.csv")

    print("Running bert_emb_ln2")

    # Initialize cols
    stimuli["cd_cosine_bert_ln2"] = np.nan
    stimuli["sc_cosine_bert_ln2"] = np.nan

    for ix in tqdm(stimuli.index):

        row = stimuli.loc[ix]

        """
        Central to distinguishing cosine
        """

        full_stimulus = row.setting + " " + row.critical

        # E2 doesn't (appear to) use central words
        if row.central_word != "-" and row.central_word in full_stimulus:

            central_emb = get_embedding(bert, bert_tokenizer,
                                        full_stimulus, row.central_word)

            distinguishing_emb = get_embedding(bert, bert_tokenizer,
                                               full_stimulus,
                                               row.distinguishing_word)

            stimuli.loc[ix, "cd_cosine_bert_ln2"] = cosine(
                central_emb, distinguishing_emb)

        """
        Setting to critical cosine
        """
        setting_emb = get_embedding(bert, bert_tokenizer,
                                    row.setting, row.setting)

        critical_emb = get_embedding(bert, bert_tokenizer,
                                     row.critical, row.critical)

        stimuli.loc[ix, "sc_cosine_bert_ln2"] = cosine(
            setting_emb, critical_emb)

    stimuli.to_csv("data/clean/stimuli_analysed.csv", index=False)


if __name__ == "__main__":
    print("Running embeddings.py")
    main()
