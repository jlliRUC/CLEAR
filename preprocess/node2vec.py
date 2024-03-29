import sys
sys.path.append('..')
import time
import logging
import pickle
import torch
from torch_geometric.nn import Node2Vec


# pyg.node2vec's example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py


def train_node2vec(edge_index, cell_embedding_dim, checkpoint_file, embs_file):
    # edge_index: tensor [2, n]
    logging.info("[node2vec] start.")

    model = Node2Vec(edge_index, embedding_dim=cell_embedding_dim,
                     walk_length=50, context_size=10, walks_per_node=10,
                     num_negative_samples=10, p=1, q=1, sparse=True).cuda()
    loader = model.loader(batch_size=32, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.cuda(), neg_rw.cuda())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def save_checkpoint():
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_file)
        return

    @torch.no_grad()
    def load_checkpoint():
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        return

    @torch.no_grad()
    def save_embeddings():
        embs = model()
        with open(embs_file, 'wb') as fh:
            pickle.dump(embs, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('[save embedding] done.')
        return

    epoch_total = 10
    epoch_train_loss_best = 10000000.0
    epoch_best = 0
    epoch_patience = 10
    epoch_worse_count = 0

    time_training = time.time()
    for epoch in range(epoch_total):
        time_ep = time.time()
        loss = train()
        logging.info("[node2vec] i_ep={}, loss={:.4f} @={}".format(epoch, loss, time.time() - time_ep))

        if loss < epoch_train_loss_best:
            epoch_best = epoch
            epoch_train_loss_best = loss
            epoch_worse_count = 0
            save_checkpoint()
        else:
            epoch_worse_count += 1
            if epoch_worse_count >= epoch_patience:
                break

    load_checkpoint()
    save_embeddings()
    logging.info("[node2vec] @={:.0f}, best_ep={}".format(time.time() - time_training, epoch_best))
    return
