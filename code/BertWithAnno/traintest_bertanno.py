import torch, json, os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from modelling_utils.clf_module import CLFModule
from modelling_utils.data_module import DataModule

class Params:
    def __init__(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)
            

def train_and_test(args):

    dm = DataModule(args)

    # TRAIN AND TEST
    trainer = pl.Trainer(max_epochs=args.num_train_epochs, enable_checkpointing=args.enable_checkpointing,
                         logger=[TensorBoardLogger(save_dir=args.output_dir, name='parameter_logs')], 
                         callbacks=[TQDMProgressBar()], devices=1, precision=32, gradient_clip_val=1.0, 
                         log_every_n_steps=1, val_check_interval=1.0, check_val_every_n_epoch=10, num_sanity_val_steps=2)
    model = CLFModule(**args.__dict__)
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(args.output_dir+"/model.ckpt")
    trainer.test(model, datamodule=dm)
    
    # TEST ONLY
    #model = CLFModule.load_from_checkpoint(args.output_dir+"/model.ckpt")
    #pl.Trainer().test(model, datamodule=dm)


def main(args):

    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.annotator_id_path, 'r') as f:
        annotator_ids = json.load(f)
    args.annotator_ids = annotator_ids

    train_and_test(args)


if __name__ == "__main__":
    
    use_annotator_embed = True
    output_dir= '../../output/BertWithAnno/binary_wikdef_sentence_annotator'
    data_path = '../../data/HateWiC_IndividualAnnos_80-10-10split/'
    annotator_id_path=f'{data_path}annotator_ids.json'

    info2column = {
        "id": "annotation_id",
         "text": "wiktionary_definition",
         "label": "annotation_label",
         "annotator_id": "annotator_id"
        }
    label2id = {"Strongly hateful": 1, "Weakly hateful": 1, "Not hateful": 0}
    id2label = {1: "Hateful", 0: "Not hateful"}
    
    model_name_or_path = 'bert-base-uncased'

    args = {'output_dir':output_dir, 'data_path':data_path, 'info2column': info2column, 'use_annotator_embed':use_annotator_embed, 
            'annotator_id_path':annotator_id_path, 'label2id':label2id, 'id2label':id2label, 'model_name_or_path':model_name_or_path,
            'enable_checkpointing': False, 'num_train_epochs': 3, 'eval_batch_size': 32, 
            'train_batch_size': 32, 'learning_rate': 1e-05, 'adam_epsilon': 1e-08, 'seed': 42}
    
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' # warning: might cause system failure, alternative: lower batch size
   
    main(Params(args))