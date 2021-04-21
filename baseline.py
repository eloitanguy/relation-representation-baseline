from transformers import BertTokenizer, BertModel
import json
from descriptions import create_comparison_sentence
import torch
from modules import preprocess_sentence, dot_similarity, printProgressBar

with open('semeval_train.json', 'r') as f:
    data_train = json.load(f)

bert_base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_base = BertModel.from_pretrained("bert-base-uncased").cuda()


def cause_vs_component(dataset, model, tokenizer, sim):
    with torch.no_grad():
        relations_indices = [1, 2, 9, 10]  # keep Cause-Effect and Component-Whole, both directions
        data = [e for e in dataset if e['r'] in relations_indices]
        results = []
        for idx, entry in enumerate(data):
            printProgressBar(idx, len(data))
            original_sentence = entry['text']
            og_ids, og_mask = preprocess_sentence(original_sentence, tokenizer)  # shapes (1, 64)

            comparison_sentences = [create_comparison_sentence(original_sentence, entry['e1'], entry['e2'], r_idx)
                                    for r_idx in relations_indices]
            encoded_comparison_sentences = [preprocess_sentence(s, tokenizer) for s in comparison_sentences]

            ids = torch.cat([og_ids] + [e[0] for e in encoded_comparison_sentences])  # stacking input ids
            mask = torch.cat([og_mask] + [e[1] for e in encoded_comparison_sentences])  # stacking attention masks

            model_hidden_states = model(ids, attention_mask=mask).last_hidden_state  # shape (5, sentence_length, h)
            model_output = model_hidden_states[:, 0, :]  # use the CLS output: first hidden state: shape (5, h)

            similarities = sim(model_output).cpu().numpy()
            cause_score = max(similarities[:2])
            component_score = max(similarities[2:])
            is_cause = cause_score > component_score

            result = {
                'Cause-Effect(e1,e2)_score': str(similarities[0]),
                'Cause-Effect(e1,e2)_sentence': str(comparison_sentences[0]),
                'Cause-Effect(e2,e1)_score': str(similarities[1]),
                'Cause-Effect(e2,e1)_sentence': str(comparison_sentences[1]),
                'cause_score': str(cause_score),
                'Component-Whole(e1,e2)_score': str(similarities[2]),
                'Component-Whole(e1,e2)_sentence': str(comparison_sentences[2]),
                'Component-Whole(e2,e1)_score': str(similarities[3]),
                'Component-Whole(e2,e1)_sentence': str(comparison_sentences[3]),
                'component_score': str(component_score),
                'is_cause': str(is_cause),
                'original_sentence': str(original_sentence),
                'e1': entry['e1'],
                'e2': entry['e2'],
                'label': entry['r_name'],
                'is_cause_gt': str(entry['r'] in [1, 2])
            }

            results.append(result)

        with open('cause_component_results.json', 'w') as f:
            json.dump(results, f, indent=4)


cause_vs_component(data_train, bert_base, bert_base_tokenizer, dot_similarity)
