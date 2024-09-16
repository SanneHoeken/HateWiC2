from torchmetrics.functional import pairwise_cosine_similarity
import torch
import numpy as np

def sample_triples(id2embeddings, id2data, topk):
    
    # get X and y data (in aligned order)
    data_ids = list(id2data.keys())
    X = [id2embeddings[data_id] for data_id in data_ids]
    y = [id2data[data_id]['label'] for data_id in data_ids]
    assert set(y) == {0, 1}

    # store embeddings of positive and negative examples and their ids
    pos_vecs, neg_vecs, pos_data_ids, neg_data_ids = [], [], [], []
    for i, y in enumerate(y):
        if y == 1:
            pos_vecs.append(X[i])
            pos_data_ids.append(data_ids[i])
        elif y == 0:
            neg_vecs.append(X[i])
            neg_data_ids.append(data_ids[i])
    
    # compute pairwise cosine similarity matrix between positive and negative embeddings
    posneg_pairwise_sim = pairwise_cosine_similarity(torch.stack(pos_vecs), torch.stack(neg_vecs))
    posid2sortednegids = {pos_data_ids[p_id]: [neg_data_ids[x] for x in np.argsort(posneg_pairwise_sim[p_id])] for p_id in range(len(pos_vecs))}

    # compute pairwise cosine similarity matrix between positive embeddings
    pospos_pairwise_sim = pairwise_cosine_similarity(torch.stack(pos_vecs), torch.stack(pos_vecs))
    posid2sortedposids = {pos_data_ids[p_id]: [pos_data_ids[x] for x in np.argsort(pospos_pairwise_sim[p_id])] for p_id in range(len(pos_vecs))}

    # collect a set of triples for every positive embedding as anchor embedding
    sample_size = min([topk, len(pos_vecs), len(neg_vecs)])
    triples = []
    for anchor_id in pos_data_ids:
        # triple positive: top k positive embeddings that are most similar to anchor embedding
        for i in range(sample_size):
            pos_id = posid2sortedposids[anchor_id][-(i+1)]
            if anchor_id != pos_id:
                for j in range(sample_size):
                    # triple negative: top k negative embeddings that are most similar to anchor embedding
                    neg_id = posid2sortednegids[anchor_id][-(j+1)]
                    triples.append((anchor_id, pos_id, neg_id))

                    # triple negative 2: top k positive embeddings that are LEAST similar to anchor embedding
                    #neg_id2 = posid2sortedposids[anchor_id][i]
                    #triples.append((anchor_id, pos_id, neg_id2))

    sentence_triples = list(set([(id2data[id1]['text'], id2data[id2]['text'], id2data[id3]['text']) for (id1, id2, id3) in triples]))
    print('Resulting number of triples:', len(sentence_triples))

    return sentence_triples