from __future__ import annotations
import pytorch_lightning as pl
import torch
import pandas as pd
import torch.nn as nn
from overrides import overrides
from torch.optim import AdamW
from abc import ABC
from sklearn.metrics import classification_report
from modelling_utils.bert_model import BertForClassification


class CLFModule(pl.LightningModule, ABC):

    def __init__(self, **kwargs):  
        super().__init__()

        self.save_hyperparameters()
        
        self.id2label = self.hparams.id2label
        self.num_classes = len(self.id2label)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        model_kwargs = {"annotator_ids": self.hparams.annotator_ids,
                        "use_annotator_embed": self.hparams.use_annotator_embed,
                        "num_classes": self.num_classes}
        self.model = BertForClassification.from_pretrained(self.hparams.model_name_or_path, **model_kwargs)
        self.testing_step_outputs = []


    @overrides(check_signature=False)
    def forward(self, input_ids, attention_mask, **kwargs):
        return self.model(input_ids, attention_mask, **kwargs)
    
    @overrides
    def configure_optimizers(self):
        #optimizer_grouped_parameters = [{"params": [p for n, p in self.model.named_parameters()], "weight_decay": 0.0,}]
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return {"optimizer": optimizer}
    
    @overrides(check_signature=False)
    def training_step(self, batch):
        logits, _ = self(batch["text_ids"], batch["text_mask"], annotator_ids=batch["annotator_id"])
        loss = self.cross_entropy_loss(logits.view(-1, self.num_classes), batch["label_ids"])
        return loss
    
    @overrides(check_signature=False)
    def test_step(self, batch):

        logits, embedding_output = self(batch["text_ids"], batch["text_mask"], annotator_ids=batch["annotator_id"])
        #logits = logits.cpu()
        output = {"logits": logits,
                "predictions": [self.id2label[int(pred_id)] for pred_id in logits.argmax(-1)],
                "probabilities": [float(prob) for prob in torch.max(logits.softmax(dim=-1), 1).values], 
                "ids": batch["id"],
                "texts": batch["text"],
                "golds": [self.id2label[int(label_id)] for label_id in batch["label_ids"]],
                "annotator_ids": batch["origin_annotator_id"],
                "full_embeddings": embedding_output["embeddings"],
                "annotator_embeds": embedding_output["annotator_embed"],
                "sentence_embeds": embedding_output["sentence_embed"]}
        
        self.testing_step_outputs.append(output)
    

    @overrides(check_signature=False)
    def on_test_epoch_end(self):
        
        outputs = self.testing_step_outputs
        texts = [x for xs in [output["texts"] for output in outputs] for x in xs]
        ids = [x for xs in [output["ids"] for output in outputs] for x in xs]
        annotator_ids = [x for xs in [output["annotator_ids"] for output in outputs] for x in xs]
        y_preds = [x for xs in [output["predictions"] for output in outputs] for x in xs]
        y_golds = [x for xs in [output["golds"] for output in outputs] for x in xs]
        pred_probs = [x for xs in [output["probabilities"] for output in outputs] for x in xs]
        
        print(classification_report(y_golds, y_preds))
        output_df = pd.DataFrame({'id': ids, 'text': texts, 'annotator_id': annotator_ids,
                                 'label': y_golds, 'preds': y_preds, 'pred_prob': pred_probs})
        output_df.to_csv(self.hparams.output_dir + '/preds.csv', index=False)
        
        # output pt file for the embeddings
        embedding_info = {
            "full_embeddings": [output["full_embeddings"] for output in outputs], 
            "annotator_embed": [output["annotator_embeds"] for output in outputs], 
            "sentence_embed": [output["sentence_embeds"] for output in outputs],
            # learned embedding???
            "annotator_embed_learned": self.model.embeddings.annotator_embed.state_dict()} 
        
        torch.save(embedding_info, f"{self.hparams.output_dir}/embedding-info.pt")