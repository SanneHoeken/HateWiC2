import pandas as pd
import numpy as np
import torch, json
from torchmetrics.functional import pairwise_cosine_similarity
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

def get_id2embeddings(id2data, model_path):
    
    sentences = [id2data[data_id]['sentence'] for data_id in id2data]
    data_ids = list(id2data.keys())
    model = SentenceTransformer(model_path).cpu() # if device is mps, because that doesn't work
    embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
    
    return {data_id: emb for data_id, emb in zip(data_ids, embeddings)}


def sample_triples(data_ids, id2embeddings, id2data, topk=10):
    
    # get X and y data (in aligned order)
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
                    neg_id2 = posid2sortedposids[anchor_id][i]
                    triples.append((anchor_id, pos_id, neg_id2))

    sentence_triples = [[id2data[id1]['sentence'], id2data[id2]['sentence'], id2data[id3]['sentence']] for (id1, id2, id3) in triples]
    print('Resulting number of triples:', len(triples))

    return sentence_triples


def train_contrastive_model(train_data, model_dir, pretrained_model_name, batch_size=16, epochs=3, triplet_margin=1):
    
    train_dataset = Dataset.from_dict({"anchor": [t[0] for t in train_data], "positive": [t[1] for t in train_data], "negative": [t[2] for t in train_data]})
    model = SentenceTransformer(pretrained_model_name).cpu() # if device is mps, because that doesn't work
    train_loss = losses.TripletLoss(model=model, triplet_margin=triplet_margin)
    args = SentenceTransformerTrainingArguments(output_dir=model_dir, per_device_train_batch_size=batch_size, num_train_epochs=epochs)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=train_loss) #evaluator=evaluator
    trainer.train()
    model.save_pretrained(model_dir)


def main(input_path, split2ids_path, sentence_column, id_column, label_column, 
         pretrained_model_name, embedding_path, model_path):
    
    # load data
    with open(split2ids_path, 'r') as infile:
        split2ids = json.load(infile)
    train_ids = split2ids['train']
    data = pd.read_csv(input_path)
    id2data = {str(row[id_column]): {'sentence': row[sentence_column].lower(), 
                                     'label': row[label_column]} for i, row in data.iterrows()}
    
    # sample training triples
    id2embeddings = get_id2embeddings(id2data, pretrained_model_name)
    sentence_triples = sample_triples(train_ids, id2embeddings, id2data, topk=10)
    
    # train model
    train_contrastive_model(sentence_triples, model_path, pretrained_model_name)
    
 
if __name__ == '__main__':

    pretrained_model_name = 'sentence-transformers/all-mpnet-base-v2'
    input_path = '../data/WiC-majority-withT5gendefs.csv' 
    split2ids_path = '../data/train70dev10test20-ids.json'
    id_column = 'id'
    label_column = 'majority_annotation'
    sentence_column = 'generated_definition'
    model_path = '../output/models/contrastive70extraneg/'
    
    main(input_path, split2ids_path, sentence_column, id_column, label_column, pretrained_model_name, model_path)
    