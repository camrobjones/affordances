# Based on James's GPT-3 code

import pandas as pd
import openai
import numpy as np
from tqdm import tqdm

openai.organization = "org-krZWjVYWySsMZqzv0Vg9b9BZ"
openai.api_key = "sk-g1SmnWaPHaiypp5mkWr9liOSachFfcWCROnFHJ0T"


def get_total_surprisal(prompt, model="ada"):
    """Get the probability of unseen following prompt"""

    # Just interested in logprobs, no completion tokens
    output = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=0,
        temperature=1,
        top_p=1,
        n=1,
        stream=False,
        logprobs=1,
        stop="\n",
        echo=True
        )
    logprobs = output.to_dict()['choices'][0].to_dict()['logprobs']["token_logprobs"]
    # Sum logprobs. First is None
    logprob = np.sum(logprobs[1:])
    surprisal = -np.log2(np.exp(logprob))

    return surprisal


def main():
    """main"""

    # Load data
    stimuli = pd.read_csv("data/clean/stimuli_analysed.csv")

    print("Running gpt3_ada")

    surprisals = []

    for ix in tqdm(range(len(stimuli))):

        row = stimuli.loc[ix]
        full_stimulus = row.setting + " " + row.critical
        spl = get_total_surprisal(full_stimulus, model="ada")
        surprisals.append(spl)

    stimuli["gpt3_ada_spl"] = surprisals

    # Davinci
    surprisals = []

    for ix in tqdm(range(len(stimuli))):

        row = stimuli.loc[ix]
        full_stimulus = row.setting + " " + row.critical
        spl = get_total_surprisal(full_stimulus, model="davinci")
        surprisals.append(spl)

    stimuli["gpt3_davinci_spl"] = surprisals

    stimuli.to_csv("data/clean/stimuli_analysed.csv", index=False)

