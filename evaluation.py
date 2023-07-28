import torch
from dataset import ExpDataLoader
from train import get_model
import os
import h5py
import sys
sys.path.append("..")


def clear(args, source_file, vec_file):
    # read source sequences from {}_seq.h5 and write the tensor into vec_{}.h5
    # Load Model
    model = get_model(vocab_size=args.vocab_size,
                      embed_size=args.embed_size,
                      hidden_size=args.hidden_size,
                      dropout=args.dropout,
                      num_layers=args.num_layers,
                      bidirectional=args.bidirectional,
                      model_name=args.model_name)
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        return 0
    print(model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # Load exp dataset
    vecs = []
    scaner = ExpDataLoader(source_file, args.batch_size)
    scaner.load()
    i = 0
    while True:
        if i % 100 == 0:
            print("Batch {}: Encoding {} trjs...".format(i, args.batch_size * i))
        i = i + 1
        s, lengths = scaner.get_one_batch()
        if s is None: break
        if torch.cuda.is_available():
            s = s.cuda()
            h, _ = model(s, lengths)  # (batch, hidden_size)
            vecs.append(h.cpu().data)

    vecs = torch.cat(vecs).contiguous()  # (num_seqs, hidden_size)
    path = vec_file
    print("=> saving vectors into {}".format(path))
    with h5py.File(path, "w") as f:
        f['vec'] = vecs.numpy()


def model_generator(model_name):
    if model_name.startswith("clear"):
        return clear


