import pandas as pd
import numpy as np
import torch, os, json
from modelling_utils.get_embeddings import get_embeddings
from modelling_utils.traintest_mlp import train_test_mlp

class Params:
    def __init__(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)

def get_id2data(args, split):
    
    df = pd.read_csv(args.data_path+f'{split}.csv')
    label_column = args.info2column["label"]
    df = df[df[label_column].isin(list(args.label2id.keys()))].replace({label_column: args.label2id}) 
    id2data = {
        str(row[info2column['id']]): 
        {
            'text': row[info2column['text']].lower(),
            'target': row[info2column['target']].lower(),
            'label': row[label_column]
            } 
            for _, row in df.iterrows()}
    
    return id2data


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 
    with open(args.output_dir+'/params.json', 'w') as outfile:
        json.dump(args.__dict__, outfile)

    # load data with encoded labels
    train_data = get_id2data(args, 'train')
    test_data = get_id2data(args, 'test')
    
    train_embeds = get_embeddings(train_data, args)
    test_embeds = get_embeddings(test_data, args)

    train_test_mlp(train_data, train_embeds, test_data, test_embeds, args)
    
 
if __name__ == '__main__':

    output_dir=f'../../output/EmbedsCLF/contrastive-example-token'
    model_path = '../../output/ContrastiveModels/example/model' #'bert-base-uncased' #'sentence-transformers/all-mpnet-base-v2' #
    sentence_transformer = True
    embedding_type = 'token'

    data_path = '../../data/HateWiC_IndividualAnnos_80-10-10split/'
    info2column = {
        "id": "annotation_id",
        "text": "example",
        "target": "term",
        "label": "annotation_label"
        }
    
    label2id = {"Strongly hateful": 1, "Weakly hateful": 1, "Not hateful": 0}
    id2label = {1: "Hateful", 0: "Not hateful"}
     
    MLP_params = {'hidden_layer_sizes': (300, 200, 100, 50), 'learning_rate_init': 0.0005, 'max_iter': 10}
    args = {'output_dir':output_dir, 'data_path':data_path, 'info2column': info2column, 'seed': 42, 
            'model_path': model_path, 'sentence_transformer': sentence_transformer, 'embedding_type': embedding_type,
            'label2id': label2id, 'id2label': id2label, 'mlp_params': MLP_params}
    
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' # warning: might cause system failure, alternative: lower batch size
    
    main(Params(args))