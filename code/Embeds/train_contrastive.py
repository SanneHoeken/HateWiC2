import pandas as pd
import numpy as np
import torch, os, json
from datasets import Dataset
from torchmetrics.functional import pairwise_cosine_similarity
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

class Params:
    def __init__(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)

def get_id2embeddings(id2data, model_path):
    
    sentences = [id2data[data_id]['text'] for data_id in id2data]
    data_ids = list(id2data.keys())
    model = SentenceTransformer(model_path) #.cpu() if device is mps, because that doesn't work
    embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
    
    return {data_id: emb for data_id, emb in zip(data_ids, embeddings)}

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
    posid2sortednegids = {pos_data_ids[p_id]: [neg_data_ids[x] for x in np.argsort(posneg_pairwise_sim[p_id].cpu())] for p_id in range(len(pos_vecs))}

    # compute pairwise cosine similarity matrix between positive embeddings
    pospos_pairwise_sim = pairwise_cosine_similarity(torch.stack(pos_vecs), torch.stack(pos_vecs))
    posid2sortedposids = {pos_data_ids[p_id]: [pos_data_ids[x] for x in np.argsort(pospos_pairwise_sim[p_id].cpu())] for p_id in range(len(pos_vecs))}

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


def train_contrastive_model(sentence_triples, args):
    
    train_dataset = Dataset.from_dict({
        "anchor": [t[0] for t in sentence_triples], 
        "positive": [t[1] for t in sentence_triples], 
        "negative": [t[2] for t in sentence_triples]})
    model = SentenceTransformer(args.model_name_or_path) #.cpu() if device is mps, because that doesn't work
    train_loss = losses.TripletLoss(model=model, triplet_margin=args.triplet_margin)
    
    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir+'/model', 
        per_device_train_batch_size=args.train_batch_size, 
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        logging_strategy='no', save_strategy='no',
        report_to='none')
    
    trainer = SentenceTransformerTrainer(model=model, args=train_args, train_dataset=train_dataset, loss=train_loss) 
    trainer.train()
    model.save_pretrained(args.output_dir+'/model')


def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 
    with open(args.output_dir+'/train_params.json', 'w') as outfile:
        json.dump(args.__dict__, outfile)

    # load train data
    train_df = pd.read_csv(args.data_path+'train.csv')
    label_column = args.info2column["label"]
    train_df = train_df[train_df[label_column].isin(list(args.label2id.keys()))].replace({label_column: args.label2id})
    train_data = {str(row[args.info2column['id']]): 
                    {'text': row[args.info2column['text']].lower(),
                     'label': row[label_column]} for _, row in train_df.iterrows()}
    
    # sample training triples and train model
    train_embeds = get_id2embeddings(train_data, args.model_name_or_path)
    sentence_triples = sample_triples(train_embeds, train_data, args.topk)
    train_contrastive_model(sentence_triples, args)
    
 
if __name__ == '__main__':

    output_dir= 'individual_wikdef_k=10'
    data_path = 'HateWiC_IndividualAnnos_80-10-10split/'
    info2column = {
        "id": "annotation_id",
        "text": "wiktionary_definition",
        "label": "annotation_label",
        }
    
    label2id = {"Strongly hateful": 1, "Weakly hateful": 1, "Not hateful": 0}
    id2label = {1: "Hateful", 0: "Not hateful"}
    model_name_or_path = 'sentence-transformers/all-mpnet-base-v2'

    train_args = {'output_dir':output_dir, 'data_path':data_path, 'info2column': info2column, 
                'label2id': label2id, 'model_name_or_path':model_name_or_path, 'topk': 10, 
                'triplet_margin': 1, 'num_train_epochs': 3, 'train_batch_size': 32, 
                'learning_rate': 1e-05, 'adam_epsilon': 1e-08, 'seed':42}
    train_args = Params(train_args)
    #os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' # warning: might cause system failure, alternative: lower batch size
    
    main(train_args)