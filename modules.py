import torch


def preprocess_sentence(s, tokenizer, to_cuda=True):
    encoded_dict = tokenizer.encode_plus(s, add_special_tokens=True, max_length=64, padding='max_length',
                                         truncation=True, return_attention_mask=True,
                                         return_tensors='pt')

    if to_cuda:
        return encoded_dict['input_ids'].cuda(), encoded_dict['attention_mask'].cuda()
    else:
        return encoded_dict['input_ids'], encoded_dict['attention_mask']


def dot_similarity(t):
    """
    :param t: tensor of shape (1 + n_comparisons, h)
    :return: the dot-product similarities between the first vector and the others
    """
    return torch.matmul(t[1:, :], t[0, :].T)  # (n_comparison, h) matrix product (h)


def cosine_similarity(t):
    """
    :param t: tensor of shape (1 + n_comparisons, h)
    :return: the cosine similarities between the first vector and the others
    """
    p = torch.matmul(t[1:, :], t[0, :].T)
    n = torch.norm(t[0, :]) * torch.norm(t[1:, :], dim=1)
    return p / n


def printProgressBar(iteration, total, prefix='', suffix='', decimals=2, length=80, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
