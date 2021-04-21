# Classifying SemEval relations by comparing BERT representations

## Method

Given a sentence from SemEval annotated with either Cause-Effect or Component-Whole, for instance "the <e1>radiation</e1> from the atomic <e2>bomb explosion</e2> is a typical acute radiation":

We compute its representation using BERT's [CLS] output, as well as the representations for the following test sentences:

- "radiation causes bomb explosion" for testing Cause-Effect(e1,e2)
- "bomb explosion causes radiation" for testing Cause-Effect(e2,e1)
- "radiation is part of bomb explosion" for testing Component-Whole(e1,e2)
- "bomb explosion is part of radiation" for testing Component-Whole(e2,e1)

Using a similarity score between the original representation and the 4 test representations, we choose the relation with the highest score.

## Getting started

Please put the [dataset](https://www.kaggle.com/drtoshi/semeval2010-task-8-dataset) in data/SemEval so that the scripts can access 'data/SemEval/TRAIN_FILE.TXT'.

In order to run BERT predictions on the train set, run ```basline.py```

In order to evaluate the predictions, run ```analysis.py```

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