from __future__ import annotations

import torch
import pandas as pd
import pytorch_lightning as pl
import torch.multiprocessing
from transformers import AutoTokenizer
from overrides import overrides
from torch.utils.data import DataLoader, Dataset


class DataModule(pl.LightningDataModule):  
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    @overrides
    def train_dataloader(self):

        dataset = AnnoDataset(data_path=self.args.data_path+'train.csv', tokenizer=self.tokenizer, label2id=self.args.label2id,
                                 is_train=True, annotator_ids=self.args.annotator_ids, info2column=self.args.info2column)
        
        return DataLoader(dataset, batch_size=self.args.train_batch_size, drop_last=False, shuffle=True, collate_fn=dataset.collate, pin_memory=True)

    @overrides
    def val_dataloader(self):
        
        dataset = AnnoDataset(data_path=self.args.data_path+'dev.csv', tokenizer=self.tokenizer, label2id=self.args.label2id,
                                 is_train=False, annotator_ids=self.args.annotator_ids, info2column=self.args.info2column)

        return DataLoader(dataset, batch_size=self.args.eval_batch_size, drop_last=False, shuffle=False, collate_fn=dataset.collate, pin_memory=True)
    
    @overrides
    def test_dataloader(self):
        
        dataset = AnnoDataset(data_path=self.args.data_path+'test.csv', tokenizer=self.tokenizer, label2id=self.args.label2id,
                                 is_train=False, annotator_ids=self.args.annotator_ids, info2column=self.args.info2column)

        return DataLoader(dataset, batch_size=self.args.eval_batch_size, drop_last=False, shuffle=False, collate_fn=dataset.collate, pin_memory=True)


class AnnoDataset(Dataset):

    def __init__(self, data_path=None, tokenizer=None, label2id=None, max_len=None, is_train=None, annotator_ids=None, info2column=None):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.is_train = is_train
        self.max_len = max_len
        self.annotator_tokenizer = {annotator_id: i for i, annotator_id in enumerate(annotator_ids)}
        self.info2column = info2column

        df = pd.read_csv(data_path)
        df = df[df[info2column["label"]].isin(list(label2id.keys()))]
        self.instances = df.to_dict('records')

    @overrides
    def __getitem__(self, i):

        instance = self.instances[i]
        output = {
            "id": instance[self.info2column["id"]],  
            "text": instance[self.info2column["text"]].lower(),
            "label": instance[self.info2column["label"]], 
            "annotator_id": instance[self.info2column["annotator_id"]]
        }
        # TODO: process other info in instances

        return output

    def collate(self, instances):
        
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        tokenization = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt")
        batch["text_ids"] = tokenization["input_ids"]
        batch["text_mask"] = tokenization["attention_mask"]
        batch["label_ids"] = torch.Tensor([self.label2id[label] for label in batch["label"]])
        batch["origin_annotator_id"] = batch["annotator_id"]
        batch["annotator_id"] = torch.LongTensor([self.annotator_tokenizer[i] for i in batch["annotator_id"]])
        
        # TODO: batch process other info in instances
        
        return batch

    def __len__(self):
        return len(self.instances)

