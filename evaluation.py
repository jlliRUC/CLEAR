import torch
from dataset import ExpDataLoader
from train import get_model
from baseline.t2vec import EncoderDecoder, DataOrderScaner
import os
import pickle
import sys

sys.path.append("..")


def clear(args, source_file, vec_file):
    "read source sequences from {}_seq.h5 and write the tensor into vec_{}.h5"
    # Load Model
    if args.pretrain_mode == "np":
        print("Train without pretraining")
        pretrain_file = None
    else:
        pretrain_file = f"data/{args.dataset_name}/{args.dataset_name}_size-{args.hidden_size}_cellsize-{args.cell_size}_minfreq-{args.minfreq}_node2vec.pkl"

    model = get_model(model_name=args.model_name,
                      vocab_size=args.vocab_size,
                      embed_size=args.embed_size,
                      hidden_size=args.hidden_size,
                      dropout=args.dropout,
                      num_heads=args.num_heads,
                      num_layers=args.num_layers,
                      bidirectional=args.bidirectional,
                      pretrain_mode=args.pretrain_mode,
                      pretrain_file=pretrain_file
                      )
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
            vecs.append(h)

    vecs = torch.cat(vecs).contiguous()  # # (num_seqs, hidden_size)
    path = vec_file
    print("=> saving vectors into {}".format(path))
    with open(path, "wb") as f:
        result = {}
        result["vec"] = vecs
        pickle.dump(result, f)


def model_generator(model_name):
    if model_name.startswith("clear"):
        return clear

