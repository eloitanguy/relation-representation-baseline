import json
from torch.utils.data import Dataset
from modules import preprocess_sentence

RELATION_LIST = {
    0: "Other",
    1: "Cause-Effect(e1,e2)",
    2: "Cause-Effect(e2,e1)",
    3: "Product-Producer(e1,e2)",
    4: "Product-Producer(e2,e1)",
    5: "Entity-Origin(e1,e2)",
    6: "Entity-Origin(e2,e1)",
    7: "Instrument-Agency(e1,e2)",
    8: "Instrument-Agency(e2,e1)",
    9: "Component-Whole(e1,e2)",
    10: "Component-Whole(e2,e1)",
    11: "Content-Container(e1,e2)",
    12: "Content-Container(e2,e1)",
    13: "Entity-Destination(e1,e2)",
    14: "Entity-Destination(e2,e1)",
    15: "Member-Collection(e1,e2)",
    16: "Member-Collection(e2,e1)",
    17: "Message-Topic(e1,e2)",
    18: "Message-Topic(e2,e1)"
}

RELATION_KEY = {k: v for (v, k) in RELATION_LIST.items()}


def process_SemEval():
    """
    Used to pre-process the SemEval data from https://www.kaggle.com/drtoshi/semeval2010-task-8-dataset:\n
    dumps a json file containing a list with entries of the form:\n
    {
        'text': [sentence string],\n
        'e1': (e1_start_idx, e1_end_idx),\n
        'e2': (e2_start_idx, e2_end_idx),\n
        'r_name': [relation string],\n
        'r': [relation index]
    }
    """
    path = 'data/SemEval/TRAIN_FILE.TXT'
    file = open(path, 'r')
    lines = file.readlines()
    n_lines = len(lines)
    entries = []
    read_idx = 0
    success = 0

    while n_lines - read_idx >= 3:
        sentence, relation = lines[read_idx], lines[read_idx + 1]
        start_idx = sentence.find('"') + 1
        end_idx = sentence.find('"', start_idx)

        if -1 in [start_idx - 1, end_idx]:  # incorrect entry: skip
            read_idx += 4
            continue

        processed_sentence = sentence[start_idx: end_idx].lower().replace('<e1>', '').replace('</e1>', '') \
            .replace('<e2>', '').replace('</e2>', '')

        e1_start_idx, e1_end_idx, e2_start_idx, e2_end_idx = -1, -1, -1, -1

        for word_idx, word in enumerate(sentence[start_idx: end_idx].split()):  # assumes unicity of the entity tokens
            if word.find('<e1>') != -1:
                e1_start_idx = word_idx
            if word.find('</e1>') != -1:
                e1_end_idx = word_idx
            if word.find('<e2>') != -1:
                e2_start_idx = word_idx
            if word.find('</e2>') != -1:
                e2_end_idx = word_idx

        if -1 in [e1_start_idx, e1_end_idx, e2_start_idx, e2_end_idx]:  # incorrect entry: skip
            read_idx += 4
            continue

        relation = relation.replace('\n', '')

        try:
            r = RELATION_KEY[relation]
        except KeyError:  # incorrect entry: skip
            read_idx += 4
            continue

        entries.append(
            {
                'text': processed_sentence,
                'e1': (e1_start_idx, e1_end_idx),
                'e2': (e2_start_idx, e2_end_idx),
                'r_name': relation,
                'r': r
            }
        )

        read_idx += 4
        success += 1

    with open('data/semeval_train.json', 'w') as f:
        json.dump(entries[:int(0.666 * success)], f, indent=4)

    with open('data/semeval_val.json', 'w') as f:
        json.dump(entries[int(0.666 * success):], f, indent=4)

    print('Finished processing {} entries'.format(success))


class ProcessedTextDataset(Dataset):
    def __init__(self, text_sentence_list, tokenizer, labels=None):
        self.tokenizer = tokenizer
        self.sentences = text_sentence_list
        self.labels = labels
        self.has_labels = labels is not None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        input_ids, mask = preprocess_sentence(self.sentences[item], self.tokenizer, to_cuda=False)
        # squeeze to have tensors of order 1
        return {'input_ids': input_ids.squeeze(),
                'mask': mask.squeeze(),
                'label': self.labels[item] if self.has_labels else None}


if __name__ == '__main__':
    process_SemEval()
