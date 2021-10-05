"""
Data wrangling
--------------
"""
import re

from wordfreq import word_frequency, zipf_frequency
import pandas as pd
import numpy as np


def get_dw_freqs(stimuli):
    """Get the distinguishing word frequencies for all stimuli:"""

    stimuli["dw_freq"] = [word_frequency(w, lang="en")
                          for w in stimuli.distinguishing_word]

    stimuli["dw_freq_log"] = np.log(stimuli["dw_freq"])

    stimuli["dw_freq_zipf"] = [zipf_frequency(w, lang="en")
                               for w in stimuli.distinguishing_word]

    return stimuli


def main():

    """
    Import data
    -----------
    """

    stimuli = pd.read_csv("data/clean/stimuli.csv")

    """
    Transform stims for mask probability
    ------------------------------------
    Create a version of each critical sentence with a mask token
    in place of the distinguishing word.
    """

    critical_masked_list = []

    for (i, row) in stimuli.iterrows():

        # Replace distinguishing word with mask token
        critical_masked = re.sub(row.distinguishing_word, " [MASK] ", row.critical, flags=re.I)
        # remove double spaces
        critical_masked = re.sub("\s+", " ", critical_masked)
        critical_masked_list.append(critical_masked)

    stimuli["critical_masked"] = critical_masked_list

    stimuli["full_masked"] = stimuli.setting + " " + stimuli.critical_masked

    # Get distinguishing word frequencies
    stimuli = get_dw_freqs(stimuli)

    """
    Write processed data
    --------------------
    """

    stimuli.to_csv("data/clean/stimuli_processed.csv", index=False)


if __name__ == "__main__":
    print("Running wrangle.py")
    main()