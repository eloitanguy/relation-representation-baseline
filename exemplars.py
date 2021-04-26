import torch
import json
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from dataset import ProcessedTextDataset
from modules import printProgressBar, preprocess_sentence
import os


def save_exemplar_representations(relations_indices, output_name, batch_size=64, workers=8):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained("bert-base-uncased").cuda()

    with open('semeval_val.json', 'r') as f:
        data_val = json.load(f)

    dataset = ProcessedTextDataset([e['text'] for e in data_val if e['r'] in relations_indices], bert_tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    n_batches = len(loader)
    average = torch.zeros(768).cuda()  # BERT-base hidden size

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            printProgressBar(batch_idx, n_batches, prefix='Processing ' + output_name + ' ...')
            input_ids = batch['input_ids'].cuda()
            masks = batch['mask'].cuda()
            batch_size = batch['mask'].shape[0]
            model_hidden_states = bert(input_ids, attention_mask=masks).last_hidden_state
            model_output = model_hidden_states[:, 0, :]
            average = average + torch.sum(model_output, dim=0) / batch_size

    file = os.path.join('exemplars/', output_name)
    if not os.path.exists('exemplars/'):
        os.makedirs('exemplars/')
    torch.save(average.cpu(), file)


if __name__ == '__main__':
    save_exemplar_representations([1], 'cause_effect_12.pt')
    save_exemplar_representations([2], 'cause_effect_21.pt')
    save_exemplar_representations([9], 'component_whole_12.pt')
    save_exemplar_representations([10], 'component_whole_21.pt')
