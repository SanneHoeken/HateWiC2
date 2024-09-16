from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict
import os, json

def split_per_annotator(data_path, output_dir, train_size, dev_size, test_size):

    assert train_size + dev_size + test_size == 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_instances, dev_instances, test_instances = [], [], []
    df = pd.read_csv(data_path, sep=';')

    annotator2data = defaultdict(list)
    for _, row in df.iterrows():
        annotator2data[row['annotator_id']].append(row)
    annotator_ids = list(annotator2data.keys())

    for _, annotator_data in annotator2data.items():

        train_data, testdev_data = train_test_split(annotator_data, train_size=train_size, random_state=12)
        test_size = test_size / (dev_size + test_size)
        dev_data, test_data = train_test_split(testdev_data, test_size=test_size, random_state=12)
        train_instances.extend(train_data)
        dev_instances.extend(dev_data)
        test_instances.extend(test_data)
    
    #save all
    train_df = pd.DataFrame(train_instances)
    dev_df = pd.DataFrame(dev_instances)
    test_df = pd.DataFrame(test_instances)

    train_df.to_csv(output_dir+'train.csv', index=False)
    dev_df.to_csv(output_dir+'dev.csv', index=False)
    test_df.to_csv(output_dir+'test.csv', index=False)

    with open(output_dir+'annotator_ids.json', "w") as f:
        json.dump(annotator_ids, f, indent=4)



def split_random(data_path, output_dir, train_size, dev_size, test_size):

    assert train_size + dev_size + test_size == 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = pd.read_csv(data_path, sep=';')

    train_data, testdev_data = train_test_split(data, train_size=train_size, random_state=12)
    test_size = test_size / (dev_size + test_size)
    dev_data, test_data = train_test_split(testdev_data, test_size=test_size, random_state=12)
        
    train_data.to_csv(output_dir+'train.csv', index=False)
    dev_data.to_csv(output_dir+'dev.csv', index=False)
    test_data.to_csv(output_dir+'test.csv', index=False)


if __name__ == '__main__':

    data_path = '../data/HateWiC_IndividualAnnos_GenDefs.csv'
    output_dir = '../data/HateWiC_IndividualAnnos_GenDefs_80-10-10split/'
    train_size = 0.8
    dev_size = 0.1
    test_size = 0.1

    split_per_annotator(data_path, output_dir, train_size, dev_size, test_size)

    data_path = '../data/HateWiC_MajorityLabels_GenDefs.csv'
    output_dir = '../data/HateWiC_MajorityBinary_80-10-10split/'
    train_size = 0.8
    dev_size = 0.1
    test_size = 0.1

    #split_random(data_path, output_dir, train_size, dev_size, test_size)