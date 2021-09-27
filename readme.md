# Glenberg & Robertson (2000) NLM Baseline Analysis

## Summary

Re-analysis of Glenberg & Robertson (2000) stimuli with Neural Language Model
(NLM) baselines.

The original experiment uses sentences that describe novel situations to test
whether participants are using grounded knowledge which cannot be gleaned from
language experience alone to understand text. For each template sentence,
they modify the object to create three versions: **related** (the 
novel object is related to the new use); **afforded** (the object is unrelated
to the new use but "the affordances of the objects could be meshed to result
in a coherent action that accomplished the character’s goal"); and
**unafforded** (the affordances of the object "cannot (easily) be meshed"
to result in a coherent action.

For example:

> Marissa forgot to bring her pillow on her camping trip. As a substitute
for her pillow, she filled up an old sweater with [object].

Where [object] is one of {clothes (related), leaves (afforded), water
(unafforded)}.

They use cosine distance measures on vector representations of the words
(LSA and HAL) to measure the relatedness of objects to the described scenario.
Objects in the related condition have lower cosine distances (are more
similar to the scenarios) than those in the afforded and unafforded
conditions (which are roughly equal).


They find that participants rate sentences in the afforded condition as more
"sensible" and easier to envision. This difference cannot be explained by
the relatedness of the words based on the distributuion of their use in language 
(operationalised as the LSA and HAL distances). The authors therefore
argue that participants are using grounded, non-linguistic experience to 
understand the sentences. 

### NLM Baseline Analysis

Here we update the distributional baseline used in the analysis, we use
measures from Transformer-based Neural Language Models. We use three
different measures to assess how well a model can distinguish between
afforded and unafforded objects using distributional information alone.

1. Masked Token Probability: We find the probability of the
novel objects used in the stimuli (e.g. leaves vs water).
2. Next Sentence Prediction: We find the probability of the *critical*
sentence following the preceding *setting* sentence in each condition.
3. Embedding Distance: We extract contextualised embeddings of the
tokens and use cosine distance to compare a) the novel object from a *central*
 word in the stimulus and b) the embeddings of the *critical* and *setting*
 sentences.


## File Structure

* `data/`
    - `raw/`
        Raw text files with original stimuli
    - `clean/`
        - `stimuli.csv`: CSV of original data
        - `stimuli_processed.csv`: CSV of stimuli post-`wrangle.py`
        - s`stimuli_analysed.csv`: CSV with NLM predictions
    - `nlm/`
        Python scripts to run NLM analyses
    - `stats/`
        R scripts for statistical analysis


## Todo

### General

* Merge wrangle with `mask.py` [ ]
* E2 analysis [ ]

### NLM

* Try different embedding layers [ ]

* Try other models
    * bert-large-cased [ ]
    * GPT-3 [ ]


### Stats

* Controls
    * Frequency? [ ]
