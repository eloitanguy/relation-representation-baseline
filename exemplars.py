import torch
import json
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from dataset import ProcessedTextDataset
from modules import printProgressBar, preprocess_sentence
import os


def save_exemplar_representations(batch_size=64, workers=8):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained("bert-base-uncased").cuda()

    with open('data/semeval_val.json', 'r') as f:
        data_val = json.load(f)

    dataset = ProcessedTextDataset([e['text'] for e in data_val],
                                   bert_tokenizer,
                                   labels=[e['r'] for e in data_val])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    n_batches = len(loader)
    averages = torch.zeros(19, 768).cuda()  # n_relations * BERT-base hidden size

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            printProgressBar(batch_idx, n_batches, prefix='Processing all exemplars ...')
            input_ids = batch['input_ids'].cuda()
            masks = batch['mask'].cuda()
            labels = batch['label'].cuda()
            model_hidden_states = bert(input_ids, attention_mask=masks).last_hidden_state
            model_output = model_hidden_states[:, 0, :]

            for r in range(19):
                is_r = labels == r  # mask of examples labelled with the relation r
                n_r = is_r.sum()  # number of examples with relation r
                if n_r > 0:  # there might be no example of this class in the batch
                    averages[r, :] = averages[r, :] + torch.sum(model_output[is_r, :], dim=0) / n_r

    file = 'exemplars/all_exemplars.pt'
    if not os.path.exists('exemplars/'):
        os.makedirs('exemplars/')
    torch.save(averages.cpu(), file)


if __name__ == '__main__':
    save_exemplar_representations()
