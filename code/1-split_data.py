from sklearn.model_selection import train_test_split
import pandas as pd
import json

def split_data(data_path, id_column, label_column, output_path, 
               train_size=0.8, dev_size=0.1, test_size=0.1):

    assert train_size + dev_size + test_size == 1

    data = pd.read_csv(data_path)
    data = data[data[label_column].notna()]
    data_ids = list(data[id_column].astype(str))

    train_ids, testdev_ids = train_test_split(data_ids, train_size=train_size, random_state=12)
    test_size = test_size / (dev_size + test_size)
    dev_ids, test_ids = train_test_split(testdev_ids, test_size=test_size, random_state=12)
    split2ids = {'train': train_ids, 'dev': dev_ids, 'test': test_ids}

    with open(output_path, 'w') as outfile:
        json.dump(split2ids, outfile)


if __name__ == '__main__':

    data_path = '../data/HateWiC-majority-withT5gendefs.csv'
    label_column = 'majority_annotation'
    id_column = 'id'
    output_path = '../data/x.json'

    split_data(data_path, id_column, label_column, output_path, 
               train_size=0.7, dev_size=0.1, test_size=0.2)

"""
#converts pickle dumped embedding files to tensordict files

import pickle, torch
from tensordict import TensorDict 

old_path = '../../HateWiC/output/cohasen/embeddings/majority/bert-base-uncased-last-definitions'
new_path = '../output/embeddings/bertbase-def'

with open(old_path, 'rb') as infile:
    old = pickle.load(infile)
new = {str(key): value for key, value in old.items()}
torch.save(TensorDict(new, batch_size=list(new.values())[0].shape), new_path) 
"""