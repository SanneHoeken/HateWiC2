import pandas as pd
import torch
from tensordict import TensorDict 
from sentence_transformers import SentenceTransformer

def create_id2embeddings(id2sentences, model_path, output_path):
    
    sentences = list(id2sentences.values())
    data_ids = list(id2sentences.keys())
    model = SentenceTransformer(model_path).cpu() # if device is mps, because that doesn't work
    embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
    #TODO: implement option to only get token embeddings for WiC-embeddings
    id2embeddings = {data_id: emb for data_id, emb in zip(data_ids, embeddings)}
    torch.save(TensorDict(id2embeddings, batch_size=embeddings[0].shape), output_path) 


def main(input_path, sentence_column, id_column, embedding_path, model_path):
    
    # load dataset
    data = pd.read_csv(input_path)
    id2sentences = {str(data_id): sent.lower() for data_id, sent in zip(data[id_column], data[sentence_column])}
    create_id2embeddings(id2sentences, model_path, embedding_path)

    
if __name__ == '__main__':

    input_path = '../data/WiC-majority-withT5gendefs.csv' 
    embedding_path = '../output/embeddings/contrastive70extraneg-allmpnetbasev2-gendef'

    id_column = 'id'
    sentence_column = 'generated_definition'
    model_path = '../output/models/contrastive70extraneg'
    #model_path = 'sentence-transformers/all-mpnet-base-v2'

    main(input_path, sentence_column, id_column, embedding_path, model_path)
    