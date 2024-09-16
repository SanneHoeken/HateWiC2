from __future__ import annotations
import torch
import torch.nn as nn
from overrides import overrides
from collections import OrderedDict
from transformers import BertModel

class EmbeddingOutputs(OrderedDict):

    embeddings=None
    annotator_embed=None
    sentence_embed=None


class CustomizedBertEmbeddings(nn.Module):

    def __init__(self, config, num_annotators):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.sent_W = nn.Parameter(torch.rand(config.hidden_size, config.hidden_size))
        self.annotator_W = nn.Parameter(torch.rand(config.hidden_size, config.hidden_size))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = config.position_embedding_type
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)
        self.annotator_embed = nn.Embedding(num_annotators, config.hidden_size)

    def _calculate_sent_alpha(self, embeddings):
        # calculate the sentence embedding based on the average of the inputs_embeds
        sent_embeds = torch.mean(embeddings, dim=1, keepdim=True)
        sent_embeds = torch.transpose(sent_embeds, 1, 2)
        alpha_sent = torch.einsum('ji, kim->kjm', self.sent_W, sent_embeds)
        return alpha_sent
    
    def _calculate_annotator_alpha(self, ann):
        antr_embed = torch.transpose(self.annotator_embed(ann).unsqueeze(1), 1, 2)
        alpha_ant = torch.einsum('ji, kim->kjm', self.annotator_W, antr_embed)
        return alpha_ant

    def _calculate_alpha(self, alpha_sent, alpha_ant):
        alpha = torch.einsum('bxm,bxn->bmn', alpha_sent, alpha_ant)
        alpha = torch.squeeze(alpha, dim=2) 
        return alpha
    
    def forward(self, input_ids, token_type_ids, annotator_ids=None):
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + token_type_embeddings
        
        position_ids = self.position_ids[:,:input_ids.size()[1]]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        annotator_embed = None  
        sentence_embed = embeddings 

        if annotator_ids is not None:
            alpha_sent = self._calculate_sent_alpha(embeddings)
            alpha_annotator = self._calculate_annotator_alpha(annotator_ids)
            alpha = self._calculate_alpha(alpha_sent, alpha_annotator)
            annotator_embed = alpha * self.annotator_embed(annotator_ids)
            embeddings = embeddings.clone()
            embeddings[:, 0, :] += annotator_embed
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return EmbeddingOutputs(embeddings=embeddings, annotator_embed=annotator_embed, sentence_embed=sentence_embed)


class BertForClassification(BertModel):

    def __init__(self, config, annotator_ids, use_annotator_embed, num_classes):
        super().__init__(config)
        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)  
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
        self.annotator_ids = annotator_ids
        self.use_annotator_embed = use_annotator_embed
        self.embeddings = CustomizedBertEmbeddings(config, num_annotators=len(annotator_ids))

    @overrides(check_signature=False)
    def forward(self, input_ids, attention_mask, annotator_ids=None, **kwargs):
        
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        token_type_ids = self.embeddings.token_type_ids[:, :seq_length].expand(batch_size, seq_length)

        if self.use_annotator_embed:
            embedding_output = self.embeddings(input_ids, token_type_ids, annotator_ids=annotator_ids)
        else:
            embedding_output = self.embeddings(input_ids, token_type_ids)

        attention_mask = torch.ones(((batch_size, seq_length)), device=input_ids.device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        
        encoder_outputs = self.encoder(embedding_output["embeddings"], attention_mask=extended_attention_mask,
                                       head_mask=head_mask, encoder_attention_mask=None, use_cache=False, 
                                       output_attentions=False, output_hidden_states=False, return_dict=True)
        # encoder_outputs: BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=tensor(), past_key_values=None, hidden_states=None, attentions=None, cross_attentions=None)
                                                  
        sequence_output = encoder_outputs[0]  # last hidden state
        
        # HERE WE CAN CHANGE WHAT THE ENCODER OUTPUT WILL BE BEFORE CLASSIFICATION
        pooled_output = self.pooler(sequence_output) # first token = [CLS]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, embedding_output
