import torch
from modules import preprocess_sentence

DESCRIPTIONS = [
    {
        "r_name": "Other"
    },
    {
        "r_name": "Cause-Effect(e1,e2)",
        "description": "is a cause of",
    },
    {  # not used in practice since we use Cause-Effect(e1,e2) with a swap instead.
        "r_name": "Cause-Effect(e2,e1)",
        "description": "is caused by",
    },
    {
        "r_name": "Product-Producer(e1,e2)",
        "description": "is produced by",
    },
    {
        "r_name": "Product-Producer(e2,e1)",
        "description": "produces",
    },
    {
        "r_name": "Entity-Origin(e1,e2)",
        "description": "origins from",
    },
    {
        "r_name": "Entity-Origin(e2,e1)",
        "description": "is the origin of",
    },
    {
        "r_name": "Instrument-Agency(e1,e2)",
        "description": "is used by",
    },
    {
        "r_name": "Instrument-Agency(e2,e1)",
        "description": "uses",
    },
    {
        "r_name": "Component-Whole(e1,e2)",
        "description": "is a part of",
    },
    {
        "r_name": "Component-Whole(e2,e1)",
        "description": "includes",
    },
    {
        "r_name": "Content-Container(e1,e2)",
        "description": "is inside",
    },
    {
        "r_name": "Content-Container(e2,e1)",
        "description": "contains",
    },
    {
        "r_name": "Entity-Destination(e1,e2)",
        "description": "goes to",
    },
    {
        "r_name": "Entity-Destination(e2,e1)",
        "description": "is the destination of",
    },
    {
        "r_name": "Member-Collection(e1,e2)",
        "description": "is a member of",
    },
    {
        "r_name": "Member-Collection(e2,e1)",
        "description": "comprises",
    },
    {
        "r_name": "Message-Topic(e1,e2)",
        "description": "is about",
    },
    {
        "r_name": "Message-Topic(e2,e1)",
        "description": "is mentioned in",
    },
]


def create_comparison_sentence(original_sentence, e1, e2, relation_idx):
    if relation_idx % 2 == 0:  # we have a ...(e2, e1) relation: swap the entity order
        e1, e2 = e2, e1
        relation_idx -= 1
    e1_start, e1_end = e1[0], e1[1] + 1  # +1 since we exclude the end index in slicing
    e2_start, e2_end = e2[0], e2[1] + 1
    strings1 = original_sentence.split()[e1_start: e1_end]
    strings2 = original_sentence.split()[e2_start: e2_end]
    comparison_strings = strings1 + DESCRIPTIONS[relation_idx]['description'].split() + strings2
    return ' '.join(comparison_strings).replace('.', '').replace(',', '').replace(':', '').replace(';', '') \
        .replace('!', '').replace('?', '')
