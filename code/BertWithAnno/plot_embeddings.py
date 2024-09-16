import torch, json, os
import numpy as np
from sklearn.manifold import TSNE
from collections import defaultdict
import matplotlib.pyplot as plt


def plot_embed(ann_list, emb_path, emb_type):
    
    embeddings = torch.load(emb_path)
    embed = torch.cat(embeddings[emb_type], dim=0)
    ann_embeds = list()
    for _, l in ann_list.items():
        idx = l[0]
        ann_embeds.append(embed[idx].cpu().numpy())

    tsne = TSNE(n_components=2, perplexity=40, random_state=12)
    embeddings_tsne = tsne.fit_transform(np.vstack(ann_embeds))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker='.', s=40)


def main():

    pred_path = '../../output/BertWithAnno/experiment1/preds.jsonl'
    with open(pred_path, 'r') as f:
        preds = f.readlines()
    preds = [json.loads(d) for d in preds]

    annotator_list = defaultdict(list)
    for i, pred in enumerate(preds):
        annotator_list[pred["annotator_id"]].append(i)
    
    plot_embed(ann_list=annotator_list,
               emb_path = '../../output/BertWithAnno/experiment1/embedding-info.pt', 
               emb_type="annotator_embed")
    
    plt.savefig(f"../../output/BertWithAnno/experiment1/annotator_embedding_plot.pdf")
    plt.close()


if __name__ == "__main__":
    
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' # warning: might cause system failure, alternative: lower batch size
    main()
