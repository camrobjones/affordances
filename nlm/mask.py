"""
G&R (2000) | Masking
--------------------
Get the surprisal of distinguishing words from Glenberg & Robertson (2000)
stimuli using masked language modelling
"""

import re

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          # pipeline,
                          AutoModelForMaskedLM)
# import nlm


"""
Masked Candidate functions
--------------------------
Functions to get probabilities of masked candidate words
in the context of a sentence.

"""


def mask_candidates_mask_all(model, tokenizer, text, candidates,
                             out="bits", agg="mean"):
    """Probabilities for candidates as replacement for [MASK] in text.

    Following Kocijan et al. (2019) this function masks all tokens in
    a multi-token candidate and aggregates them. All tokens are masked
    at once.

    Example
    -------
    >>> text = "When the rock fell on the vase, the [MASK] broke."
    >>> candidates = ["rock", "vase"]
    >>> mask_candidates_mask_all(bert, bert_tokenizer, text, candidates)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A huggingface transformers model
    model : transformers.PreTrainedTokenizer
        The relevant tokenizer
    text : str
        Text containing a single [MASK]
    candidates : list of str
        Candidate mask replacements
    out : str {"bits", "prob"}, optional
        Format of output
    agg : str {"mean", "sum"}, optional
        How to aggregate multiple token candidates
    }

    Returns
    -------
    candidates : dict
        {candidate: prob}

    """

    # Check exactly one mask
    masks = sum(np.array(text.split()) == "[MASK]")
    if masks != 1:
        raise ValueError(
            f"Must be exactly one [MASK] in text, {masks} supplied.")

    candidate_probs = {}

    # Loop through candidates and infer probability
    for candidate in candidates:

        # Get candidate ids
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)

        # Add a mask for each token in candidate
        mask_tok = tokenizer.mask_token

        candidate_text = re.sub("\[MASK\]",
                                f"{mask_tok}" * len(candidate_ids),
                                text)

        # Tokenize text
        input_ids = tokenizer.encode(candidate_text, return_tensors="pt")
        mask_inds = np.where(input_ids == tokenizer.mask_token_id)[1]

        # Predict all tokens
        with torch.no_grad():
            outputs = model(input_ids)  # token_type_ids=segments_tensors)
            logits = outputs.logits

        # get predicted tokens
        probs = []

        for (i, mask) in enumerate(mask_inds):
            # prediction for mask
            mask_preds = logits[0, mask.item()]
            mask_preds = torch.softmax(mask_preds, 0)
            prob = mask_preds[candidate_ids[i]].item()
            probs.append(np.log(prob))

        if agg == "mean":
            agg_prob = np.mean(probs)
        elif agg == "sum":
            agg_prob = np.sum(probs)
        else:
            raise ValueError(f'agg must be one of "mean" or "sum", not {agg}')

        if out == "bits":
            candidate_probs[candidate] = -2 * np.mean(agg_prob)
        elif out == "prob":
            candidate_probs[candidate] = np.exp(np.mean(agg_prob))
        else:
            raise ValueError(f'out must be one of "bits" or "prob", not {agg}')

    return candidate_probs


"""
Masking all tokens at once might unnecessarily penalise candidates
where the independent joint probability of tokens is low, but the
conditional probability of each token in the candidate is high
when the other candidate tokens are visible.

Therefore we try another method where we mask one candidate token at
a time
"""

# TODO: Merge these functions


def mask_candidates_mask_one(model, tokenizer, text, candidates,
                             out="bits", agg="mean"):
    """Probabilities for candidates as replacement for [MASK] in text.

    For multi-token candidates, one token is masked at a time and the
    probability for all tokens is aggregated.

    Example
    -------
    >>> text = "When the rock fell on the vase, the [MASK] broke."
    >>> candidates = ["rock", "vase"]
    >>> mask_candidates_mask_one(bert, bert_tokenizer, text, candidates)

    Parameters
    ----------
    Parameters
    ----------
    model : transformers.PreTrainedModel
        A huggingface transformers model
    model : transformers.PreTrainedTokenizer
        The relevant tokenizer
    text : str
        Text containing a single [MASK]
    candidates : list of str
        Candidate mask replacements
    out : str {"bits", "prob"}, optional
        Format of output
    agg : str {"mean", "sum"}, optional
        How to aggregate multiple token candidates

    Returns
    -------
    candidates : dict
        {candidate: prob}

    """

    # Check exactly one mask
    masks = sum(np.array(text.split()) == "[MASK]")
    if masks != 1:
        raise ValueError(
            f"Must be exactly one [MASK] in text, {masks} supplied.")

    mask_tok = tokenizer.mask_token

    candidate_probs = {}

    # Loop through candidates and infer probability
    for candidate in candidates:

        # Tokenize candidate
        cand_toks = tokenizer.tokenize(candidate)

        # get predicted tokens
        token_probs = []

        # Iteratively mask each candidate token
        for (ix, cand_tok) in enumerate(cand_toks):

            # Replace one token with mask
            masked_candidate = cand_toks[:ix] + [mask_tok] + cand_toks[ix+1:]
            masked_candidate = tokenizer.convert_tokens_to_string(
                masked_candidate)

            # Replace text mask with masked candidate string
            candidate_text = re.sub("\[MASK\]", masked_candidate, text)

            # Encode text
            input_ids = tokenizer.encode(candidate_text, return_tensors="pt")

            # Find mask location
            mask_ind = np.where(input_ids == tokenizer.mask_token_id)[1]

            # Predict all tokens
            with torch.no_grad():
                outputs = model(input_ids)  # token_type_ids=segments_tensors)
                logits = outputs.logits

            # prediction for mask
            mask_preds = logits[0, mask_ind.item()]
            mask_preds = torch.softmax(mask_preds, 0)
            cand_tok_id = tokenizer.convert_tokens_to_ids(cand_tok)
            prob = mask_preds[cand_tok_id].item()
            token_probs.append(np.log(prob))

        if agg == "mean":
            agg_prob = np.mean(token_probs)
        elif agg == "sum":
            agg_prob = np.sum(token_probs)
        else:
            raise ValueError(f'agg must be one of "mean" or "sum", not {agg}')

        if out == "bits":
            candidate_probs[candidate] = -2 * np.mean(agg_prob)
        elif out == "prob":
            candidate_probs[candidate] = np.exp(np.mean(agg_prob))
        else:
            raise ValueError(f'out must be one of "bits" or "prob", not {agg}')

    return candidate_probs


"""
Get surprisals
--------------
Load in the dataset, load models, and run functions on the stimuli.
"""


def mask_candidates_df(df, model, tokenizer, agg="mean", out="bits",
                       method="all"):
    """Get mask candidate surprisal for a dataframe of examples"""
    probs = []

    for ix in tqdm(range(len(df))):

        row = df.loc[ix]

        if method == "all":
            prob = mask_candidates_mask_all(
                model=model,
                tokenizer=tokenizer,
                text=row.full_masked,
                candidates=[row.distinguishing_word],
                agg=agg,
                out=out)

        elif method == "one":
            prob = mask_candidates_mask_one(
                model=model,
                tokenizer=tokenizer,
                text=row.full_masked,
                candidates=[row.distinguishing_word],
                agg=agg,
                out=out)

        probs.append(prob[row.distinguishing_word])

    return probs


def main():
    """main"""

    # Load data
    stimuli = pd.read_csv("data/clean/stimuli_processed.csv")

    # Load models
    bert = AutoModelForCausalLM.from_pretrained("bert-large-cased")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    bert.eval()

    roberta = AutoModelForCausalLM.from_pretrained("roberta-large")
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    roberta.eval()

    print("Running bert_mask_all_mean")
    stimuli["bert_mask_all_mean"] = mask_candidates_df(
        stimuli, bert, bert_tokenizer, agg="mean", method="all")

    print("Running bert_mask_one_mean")
    stimuli["bert_mask_one_mean"] = mask_candidates_df(
        stimuli, bert, bert_tokenizer, agg="mean", method="one")

    print("Running bert_mask_all_sum")
    stimuli["bert_mask_all_sum"] = mask_candidates_df(
        stimuli, bert, bert_tokenizer, agg="sum", method="all")

    print("Running bert_mask_one_sum")
    stimuli["bert_mask_one_sum"] = mask_candidates_df(
        stimuli, roberta, roberta_tokenizer, agg="sum", method="one")

    print("Running roberta_mask_all_mean")
    stimuli["roberta_mask_all_mean"] = mask_candidates_df(
        stimuli, roberta, roberta_tokenizer, agg="mean", method="all")

    print("Running roberta_mask_one_mean")
    stimuli["roberta_mask_one_mean"] = mask_candidates_df(
        stimuli, roberta, roberta_tokenizer, agg="mean", method="one")

    print("Running roberta_mask_all_sum")
    stimuli["roberta_mask_all_sum"] = mask_candidates_df(
        stimuli, roberta, roberta_tokenizer, agg="sum", method="all")

    print("Running roberta_mask_one_sum")
    stimuli["roberta_mask_one_sum"] = mask_candidates_df(
        stimuli, roberta, roberta_tokenizer, agg="sum", method="one")

    stimuli.to_csv("data/clean/stimuli_analysed.csv", index=False)


if __name__ == "__main__":
    print("Running mask.py")
    main()
