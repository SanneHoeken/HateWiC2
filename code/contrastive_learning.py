import pandas as pd
import numpy as np
import torch
from os import path
from tensordict import TensorDict 
from torchmetrics.functional import pairwise_cosine_similarity
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
#os.environ['COMMANDLINE_ARGS'] = '--no-half'

def sample_triples(data_ids, X, y, topk=10):
    
    # works for binary label set {0, 1}
    pos_vecs, neg_vecs, pos_data_ids, neg_data_ids = [], [], [], []
    for i, y in enumerate(y):
        if y == 1:
            pos_vecs.append(X[i])
            pos_data_ids.append(data_ids[i])
        elif y == 0:
            neg_vecs.append(X[i])
            neg_data_ids.append(data_ids[i])
    
    sample_size = min([topk, len(pos_vecs), len(neg_vecs)])
    print('Number of positive examples (= anchor examples):', len(pos_vecs))
    print('Number of negative examples:', len(neg_vecs))
    print('Number of sampled positive & negative examples per anchor example:', sample_size)

    posneg_pairwise_sim = pairwise_cosine_similarity(torch.stack(pos_vecs), torch.stack(neg_vecs))
    posid2sortednegids = {pos_data_ids[p_id]: [neg_data_ids[x] for x in np.argsort(posneg_pairwise_sim[p_id])] for p_id in range(len(pos_vecs))}

    pospos_pairwise_sim = pairwise_cosine_similarity(torch.stack(pos_vecs), torch.stack(pos_vecs))
    posid2sortedposids = {pos_data_ids[p_id]: [pos_data_ids[x] for x in np.argsort(pospos_pairwise_sim[p_id])] for p_id in range(len(pos_vecs))}

    triples = []
    for anchor_id in pos_data_ids:
        for i in range(sample_size):
            pos_id = posid2sortedposids[anchor_id][-(i+1)]
            if anchor_id != pos_id:
                for j in range(sample_size):
                    neg_id = posid2sortednegids[anchor_id][-(j+1)]
                    triples.append((anchor_id, pos_id, neg_id))
                    # alternative: neg_id = posid2sortedposids[anchor_id][i]

    print('Resulting number of triples:', len(triples))

    return triples


def train_contrastive_model(train_data, model_dir, pretrained_model_name, batch_size=16, epochs=3, triplet_margin=1):
    
    train_dataset = Dataset.from_dict({"anchor": [t[0] for t in train_data], "positive": [t[1] for t in train_data], "negative": [t[2] for t in train_data]})
    model = SentenceTransformer(pretrained_model_name).cpu() # if device is mps, because that doesn't work
    train_loss = losses.TripletLoss(model=model, triplet_margin=triplet_margin)
    args = SentenceTransformerTrainingArguments(output_dir=model_dir, per_device_train_batch_size=batch_size, num_train_epochs=epochs)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=train_loss) #evaluator=evaluator
    trainer.train()
    model.save_pretrained(model_dir+"final")


def create_id2embeddings(id2sentences, model_path, output_path):
    
    sentences = list(id2sentences.values())
    data_ids = list(id2sentences.keys())
    model = SentenceTransformer(model_path).cpu() # if device is mps, because that doesn't work
    embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
    id2embeddings = {data_id: emb for data_id, emb in zip(data_ids, embeddings)}
    torch.save(TensorDict(id2embeddings, batch_size=embeddings[0].shape), output_path) 


def main(input_path, sentence_column, id_column, label_column, pretrained_model_name, embedding_path, model_path):
    
    # load dataset
    data = pd.read_csv(input_path)
    data[id_column] = data[id_column].astype(str)
    id2sentences = {data_id: sent.lower() for data_id, sent in zip(data[id_column], data[sentence_column])}

    # get sentence embeddings
    if not path.exists(embedding_path):
        create_id2embeddings(id2sentences, pretrained_model_name, embedding_path)
    id2embeddings = torch.load(embedding_path)
    
    data = data[data[id_column].isin(id2embeddings.keys())].dropna(subset=[label_column]) 
    data = data.drop(data[data[label_column] == "None"].index)

    # sample training triples
    data_ids = list(data[id_column])
    X = [id2embeddings[data_id] for data_id in data_ids]
    y = data[label_column]
    id_triples = sample_triples(data_ids, X, y)
    sentence_triples = [[id2sentences[id1], id2sentences[id2], id2sentences[id3]] for (id1, id2, id3) in id_triples]

    # train model
    train_contrastive_model(sentence_triples, model_path, pretrained_model_name)

 
if __name__ == '__main__':

    pretrained_model_name = 'sentence-transformers/all-mpnet-base-v2'
    input_path = '../data/cohasen_data/cohasen_annotated_plus_T5gendefs.csv' #majority
    id_column = 'id'
    label_column = 'majority_annotation'
    sentence_column = 'generated_definition'
    embedding_path = '../output/cohasen/embeddings/majority/allmpnetbasev2-generated_definitions'
    model_path = '../output/cohasen/models/contrastive1/'

    main(input_path, sentence_column, id_column, label_column, pretrained_model_name, embedding_path, model_path)
    