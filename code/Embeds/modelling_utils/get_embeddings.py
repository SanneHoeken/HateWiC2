import torch, re
from tqdm import tqdm
from tensordict import TensorDict 
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from difflib import get_close_matches
import torch.nn.functional as F

def find_target_indices(tknzr, text, target):
            
    # encode text and target
    text_encoding = tknzr.encode(text, truncation=True)
    target_encoding = tknzr.encode(target, add_special_tokens=False)
    
    # find indices for target
    target_indices = None
    for i in range(len(text_encoding)):
        if text_encoding[i:i+len(target_encoding)] == target_encoding:
            target_indices = (i, i+len(target_encoding))
    
    # if no exact matches are found
    if not target_indices:
        new_target = None
        new_text = None
        
        # try plural (simple rules)
        if target + 's' in text:
            new_target = target + 's'
        elif target.replace('y', 'ies') in text:
            new_target = target.replace('y', 'ies')
        elif target.replace('man', 'men') in text:
            new_target = target.replace('man', 'men')
        else:
            # try to find the most similar word in the text
            potential_target = get_close_matches(target, text.split(), n=1, cutoff=0.6)
            if len(potential_target) == 1:
                most_similar = re.sub(r'[^\w\s-]','', potential_target[0])
                # replace the most similar word (for which we assume misspelling) with the target
                new_text = text.replace(most_similar, target)
        
        if new_target or new_text:
            # encode new target or text
            if new_target:
                target_encoding = tknzr.encode(new_target, add_special_tokens=False)
            elif new_text:
                text_encoding = tknzr.encode(new_text, truncation=True)
            # try finding indices again
            for i in range(len(text_encoding)):
                if text_encoding[i:i+len(target_encoding)] == target_encoding:
                    target_indices = (i, i+len(target_encoding))
    
    return target_indices


def extract_embedding(sentence_encoding, target_indices, model=None, sentence_embeddings=None):

    if sentence_embeddings != None:
        vecs = sentence_embeddings
    elif model != None:
        input_ids = torch.tensor([sentence_encoding])
        encoded_layers = model(input_ids)[-1]
        vecs = encoded_layers[-1].squeeze(0).detach() # last layer 

    start_idx, end_idx = target_indices
    vecs = vecs[start_idx:end_idx]
    vector = torch.mean(vecs, 0)
    
    return vector



def get_embeddings(id2data, args):
    
    sentences = [id2data[data_id]['text'] for data_id in id2data]
    data_ids = list(id2data.keys())
    
    if args.sentence_transformer:
        model = SentenceTransformer(args.model_path).cpu() # if device is mps, because that doesn't work
        if args.embedding_type == 'sentence':
            embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
        elif args.embedding_type == 'token':
            targets = [id2data[data_id]['target'] for data_id in id2data]
            tknzr = model._first_module().tokenizer
            sent_encodings = [tknzr.encode(sent, truncation=True) for sent in sentences]
            sent_embeddings = model.encode(sentences, output_value='token_embeddings',
                                      convert_to_tensor=True, show_progress_bar=True)
            embeddings = []
            for sent, target, enc, emb in tqdm(zip(sentences, targets, sent_encodings, sent_embeddings)):
                target_indices = find_target_indices(tknzr, sent, target)
                embedding = extract_embedding(enc, target_indices, sentence_embeddings=emb) if target_indices else None
                embeddings.append(embedding)
        else:
            print('Unvalid embedding type, set to "sentence" or "token".')
    else:
        tknzr = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModel.from_pretrained(args.model_path, output_hidden_states=True)
        model.eval()
        sent_encodings = [tknzr.encode(sent, truncation=True) for sent in sentences]
        if args.embedding_type == 'sentence':
            embeddings = [extract_embedding(enc, (0, len(enc)), model=model) for enc in tqdm(sent_encodings)]
        elif args.embedding_type == 'token':
            targets = [id2data[data_id]['target'] for data_id in id2data]
            embeddings = []
            for sent, target, enc in tqdm(zip(sentences, targets, sent_encodings)):
                target_indices = find_target_indices(tknzr, sent, target)
                embedding = extract_embedding(enc, target_indices, model=model) if target_indices else None
                embeddings.append(embedding)
        else:
            print('Unvalid embedding type, set to "sentence" or "token".')

    id2embeddings = {data_id: emb for data_id, emb in zip(data_ids, embeddings) if emb is not None}
    #torch.save(TensorDict(id2embeddings, batch_size=embeddings[0].shape), args.output_dir+'/embeddings')

    return id2embeddings