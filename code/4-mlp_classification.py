import torch, json
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report


@ignore_warnings(category=ConvergenceWarning)
def train_test_MLP(X_train, y_train, X_test, y_test, params):

    X_train = torch.stack(X_train)  
    X_test = torch.stack(X_test)

    clf = MLPClassifier(random_state=12).set_params(**params).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return predictions, accuracy

@ignore_warnings(category=ConvergenceWarning)
def MLP_GridSearch(X_dev, y_dev, 
                   param_grid = {
                       'hidden_layer_sizes':[(300, 200, 100, 50), (200, 100, 50), (100, 50)], 
                       'learning_rate_init':[0.0005, 0.001, 0.005],
                       'max_iter': [10, 20, 40, 80, 100, 200]}):

    mlp = MLPClassifier(random_state=12)
    clf = GridSearchCV(mlp, param_grid)
    clf.fit(torch.stack(X_dev), y_dev)

    return clf.best_params_


def train_test_mlp(data_path, id_column, label_column, split2id_path, embedding_path, preds_path, params):

    # load data
    id2embeddings = torch.load(embedding_path)
    data = pd.read_csv(data_path)
    id2data = {str(row[id_column]): row for i, row in data.iterrows()}
    with open(split2id_path, 'r') as infile:
        split2ids = json.load(infile)

    # get train, dev, test data
    train_X = [id2embeddings[train_id] for train_id in split2ids['train'] if train_id in id2embeddings]
    train_y = [id2data[train_id][label_column] for train_id in split2ids['train'] if train_id in id2embeddings]

    dev_X = [id2embeddings[dev_id] for dev_id in split2ids['dev'] if dev_id in id2embeddings]
    dev_y = [id2data[dev_id][label_column] for dev_id in split2ids['dev'] if dev_id in id2embeddings]

    test_X = [id2embeddings[test_id] for test_id in split2ids['test'] if test_id in id2embeddings]
    test_y = [id2data[test_id][label_column] for test_id in split2ids['test'] if test_id in id2embeddings]
        
    # train and test classification model
    if not params:
        best_params = MLP_GridSearch(dev_X, dev_y)
        predictions, accuracy = train_test_MLP(train_X, train_y, test_X, test_y, best_params)
        print(f"Grid Search Result - Best Hyperparameters: {best_params}")
    else:
        predictions, accuracy = train_test_MLP(train_X, train_y, test_X, test_y, params)

    # save preds
    output_df = pd.DataFrame([id2data[test_id] for test_id in split2ids['test'] if test_id in id2embeddings])
    output_df['prediction'] = predictions
    #output_df.to_csv(preds_path, index=False)

    print(embedding_path)
    print(classification_report(output_df[label_column], output_df['prediction']))
    

if __name__ == '__main__':
    
    data_path = '../data/WiC-majority-withT5gendefs.csv' #majority
    id_column = 'id'
    label_column = 'majority_annotation'
    split2id_path = '../data/train80dev10test10-ids.json'
    embedding_path = '../output/embeddings/bertbase-wic'
    preds_path = '../output/predictions/[].csv'
    #params = dict()
    params = {'hidden_layer_sizes': (300, 200, 100, 50), 'learning_rate_init': 0.0005, 'max_iter': 10}
    
    train_test_mlp(data_path, id_column, label_column, split2id_path, embedding_path, preds_path, params)