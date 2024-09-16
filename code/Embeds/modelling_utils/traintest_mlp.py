import torch
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report


@ignore_warnings(category=ConvergenceWarning)
def train_test_mlp(train_data, train_embeds, test_data, test_embeds, args):
    
    train_ids = [data_id for data_id in train_data if data_id in train_embeds]
    train_X = [train_embeds[data_id] for data_id in train_ids ]
    y_train = [train_data[data_id]['label'] for data_id in train_ids]

    test_ids = [data_id for data_id in test_data if data_id in test_embeds]
    test_X = [test_embeds[data_id] for data_id in test_ids]
       
    X_train = torch.stack(train_X)  
    X_test = torch.stack(test_X)

    clf = MLPClassifier(random_state=12).set_params(**args.mlp_params).fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    y_test = [test_data[data_id]['label'] for data_id in test_ids]
    print(classification_report(y_test, preds))
    
    # save preds
    texts = [test_data[data_id]['text'] for data_id in test_ids]
    output_df = pd.DataFrame({'id': test_ids, 'text': texts, 'label': y_test, 'preds': preds})
    output_df = output_df.replace({"label": args.id2label, "preds": args.id2label})
    output_df.to_csv(args.output_dir + '/preds.csv', index=False)