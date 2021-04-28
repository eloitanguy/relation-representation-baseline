# Classifying SemEval relations by comparing BERT representations

## Method

Given a sentence from SemEval annotated with either Cause-Effect or Component-Whole, for instance "the <e1>radiation</e1> from the atomic <e2>bomb explosion</e2> is a typical acute radiation":

We compute its representation using BERT's [CLS] output, as well as the representations for the following test sentences:

- "radiation causes bomb explosion" for testing Cause-Effect(e1,e2)
- "bomb explosion causes radiation" for testing Cause-Effect(e2,e1)
- "radiation is part of bomb explosion" for testing Component-Whole(e1,e2)
- "bomb explosion is part of radiation" for testing Component-Whole(e2,e1)

Using a similarity score between the original representation and the 4 test representations, we choose the relation with the highest score.

Another method that we test here is the 'exemplar' idea: we take a (disjoint) set of annotated examples, and for each relation we compare the input sentence with the mean of the representations of the exemplar sentences with the corresponding relation label.

## Getting started

In order to run BERT predictions on the train set, run ```python basline.py``` after having changed the 'main' in order to select what you want to do

In order to evaluate the predictions, run ```python analysis.py --f [NAME OF RESULT JSON FILE]```

The processed dataset and exemplars are already in the repo, but the code used to generate them is entirely present.

## Results for Cause-Effect vs Component-Whole Classification

#### Experiment 1

Settings:

- Bert base
- dot similarity
- Cause-Effect(e1,e2) description: "causes"
- Component-Whole(e1,e2) description "is part of"

Resulting F1: **57.3%**

#### Experiment 2: balancing description lengths

Settings:

- Bert base
- dot similarity
- Cause-Effect(e1,e2) description: "is a cause of"
- Component-Whole(e1,e2) description "is a part of"

Resulting F1: **72.8%**

#### Experiment 3: BERT Large

Settings:

- Bert Large
- dot similarity
- Cause-Effect(e1,e2) description: "is a cause of"
- Component-Whole(e1,e2) description "is a part of"

Resulting F1: **50.3%** (very surprising)

#### Experiment 4: balancing description lengths + cosine similarity

Settings:

- Bert base
- cosine similarity
- Cause-Effect(e1,e2) description: "is a cause of"
- Component-Whole(e1,e2) description "is a part of"

Resulting F1: **65.7%**

#### Experiment 5: Exemplar comparison with dot similarity

- Bert base
- dot similarity
- Exemplar comparison

Resulting F1: **65.3%** (always outputs component-whole for some reason)

#### Experiment 6: Exemplar comparison with cosine similarity

- Bert base
- cosine similarity
- Exemplar comparison

Resulting F1: **83.9%**