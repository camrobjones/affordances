"""
Language Models
---------------
General purpose code for getting probabilities & surprisals from nlms

"""

import logging
import re

import numpy as np

import torch
from torch.nn import functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          # pipeline,
                          AutoModelForMaskedLM)


"""
Setup
-----
Logging & parameters

"""

logging.basicConfig(level=logging.INFO)

USE_GPU = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {}  # Store loaded models

MODEL_NAMES_CAUSAL = ["gpt2", "xlnet-base-cased", "distilgpt2",
                      "gtp2-large"]
MODEL_NAMES_MASKED = ["bert-base-uncased", "roberta-base"]


"""
Model setup
-----------
"""


def get_model_and_tokenizer(name):
    """Returns (model, tokenizer) tuple. Loads model if needed.

    Parameters
    ----------
    name : str
        Name of a huggingface/transformers model

    Returns
    -------
    tuple
        (model, tokenizer)
    """

    # If model exists, return model
    if name in MODELS:
        return MODELS[name]

    # Load causal model
    if name in MODEL_NAMES_CAUSAL:
        logging.info("Loading model '%s'...", name)
        model = AutoModelForCausalLM.from_pretrained(name)
        logging.info("Model '%s' loaded", name)

    # Load masked model
    elif name in MODEL_NAMES_MASKED:
        logging.info("Loading model '%s'...", name)
        model = AutoModelForMaskedLM.from_pretrained(name)
        logging.info("Model '%s' loaded", name)

    else:
        raise ValueError("name not in model list: ", name)

    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)

    MODELS[name] = (model, tokenizer)

    return (model, tokenizer)


"""
Inference functions
-------------------
Pass texts to model and get probabilities of different completions.
"""


def mask_predict_n(text, model, tokenizer, n=5):
    """Top n predictions for all masked words in text

    Example
    -------
    >>> text = "[CLS] Sally hit Jenny because [MASK] was angry. [SEP]"
    >>> predict_masked(text, 20)

    Parameters
    ----------
    text : str
        Text containing masks
    n : int, optional
        Number of candidates to produce (5)
    model : TYPE, optional
        language model (BERT)
    tokenizer : TYPE, optional
        lm tokenizer (BERT)

    Returns
    -------
    candidates
        List of tuples  of (index, [(candidate, prob)])
    """
    input_ids = tokenizer.encode(text, return_tensors="pt")
    mask_inds = np.where(input_ids == tokenizer.mask_token_id)[1]

    # Predict all tokens
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # get predicted tokens
    out = []
    for mask in mask_inds:
        print("Predicting mask index: ", mask)

        # prediction for mask
        mask_preds = logits[0, mask.item()]
        mask_preds = torch.softmax(mask_preds, 0)

        predicted_inds = [x.item() for x in 
                          torch.argsort(mask_preds, descending=True)[:n]]
        probs = [mask_preds[i].item() for i in predicted_inds]

        predicted_tokens = []
        for index in predicted_inds:
            predicted_tokens.append(tokenizer.convert_ids_to_tokens([index])[0])

        out.append((mask, list(zip(predicted_tokens, probs))))

    return out


def mask_candidates(model_name, text, candidates, log=True, agg="mean"):
    """Probabilities for candidates as replacement for [MASK] in text
    
    Example
    -------
    >>> text = "[CLS] When the rock fell on the vase, the [MASK] broke. [SEP]"
    >>> candidates = ["rock", "vase"]
    >>> mask_candidates(text, candidates)
    
    Parameters
    ----------
    model_name : str
        The name of a huggingface transformers model
    text : str
        Text containing a single [MASK]
    candidates : list of str
        Candidate mask replacements
    
    Returns
    -------
    candidates : dict
        {candidate: prob}
    
    Raises
    ------
    ValueError
        Description
    """

    # Check exactly one mask
    masks = sum(np.array(text.split()) == "[MASK]")
    if masks != 1:
        raise ValueError(
            f"Must be exactly one [MASK] in text, {masks} supplied.")

    # Load model
    model, tokenizer = get_model_and_tokenizer(model_name)

    candidate_probs = {}

    # Loop through candidates and infer probability
    for candidate in candidates:

        # Get candidate ids
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)

        # Add a mask for each token in candidate
        mask_tok = tokenizer.mask_token
        candidate_text = re.sub("\[MASK\]", f"{mask_tok}" * len(ids), text)

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
            prob = mask_preds[ids[i]].item()
            probs.append(np.log(prob))

        if agg == "mean":
            agg_prob = np.mean(probs)
        elif agg == "sum":
            agg_prob = np.sum(probs)
        else:
            raise ValueError(f'agg must be one of "mean" or "sum", not {agg}')

        if log:
            candidate_probs[candidate] = np.mean(agg_prob)
        else:
            candidate_probs[candidate] = np.exp(np.mean(agg_prob))

    return candidate_probs


def next_seq_prob(seen, unseen, model_name, log=True):
    """Get p(unseen | seen)
    
    Parameters
    ----------
    seen : str
        Preceding context
    unseen : str
        Unseen text to be tokenized
    model_name : str
        Name of transformer model
    """

    # Load model
    model, tokenizer = get_model_and_tokenizer(model_name)

    # Pad unseen
    unseen = unseen if unseen[0] == " " else " " + unseen

    # Get ids for tokens
    input_ids = tokenizer.encode(seen, return_tensors="pt")
    unseen_ids = tokenizer.encode(unseen)

    # Loop through unseen tokens & sum log probs
    probs = []
    for unseen_id in unseen_ids:

        with torch.no_grad():
            logits = model(input_ids).logits

        next_token_logits = logits[0, -1]
        next_token_probs = torch.softmax(next_token_logits, 0)
    
        prob = next_token_probs[unseen_id]
        probs.append(np.log(prob))

        # Add input tokens incrementally to input
        input_ids = torch.cat((input_ids, torch.tensor([[unseen_id]])), 1)

    # Return log or raw prob
    prob = sum(probs) if log else np.exp(sum(probs))
    return prob


"""
Dataframe utilities
-------------------
Perform operations on columns of dataframes
"""


def mask_candidates_df(df, sentence, target):
    """Summary
    
    Parameters
    ----------
    df : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    np1_probs = []
    np2_probs = []

    for i, row in df.iterrows():

        # Extract text
        text = row['text']

        # Enforce CLS and SEP tags
        if text[:5] != "[CLS]":
            text = "[CLS] " + text

        if text[-5:] != "[SEP]":
            text = text + " [SEP]"

        # Extract candidates
        np1 = row['np1'].lower()
        np2 = row['np2'].lower()

        # Get probs
        probs = mask_probability(text, [np1, np2])

        # Add to lists
        np1_probs.append(probs[np1])
        np2_probs.append(probs[np2])

        print("\n\n" + "-" * 30 + "\n\n")
        print(f"{i} / {len(df)}: {text}")
        for k, v in probs.items():
            print(f"{k}: {v:.3g}")

    df["bert_np1_prob"] = np1_probs
    df["bert_np2_prob"] = np2_probs

    return df
