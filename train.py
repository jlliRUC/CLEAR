import torch
from torch.nn.utils import clip_grad_norm_
import time
import os
import shutil
import logging
from model.Embedding import SEmbed
from model.CLEAR import CLEAR
from model.ProjectionHead import ProjectionHead
from losses import get_loss
from dataset import ClearDataLoader
from model.utils_model import load_pretrained
import copy
import torch.nn as nn


def load_pretrain(pretrain_mode, pretrain_file, vocab_size):
    if pretrain_mode == "np":
        embed_c = None
    elif pretrain_mode == "pf":
        pretrained_cell_embeddings = load_pretrained(pretrain_file, vocab_size)
        embed_c = nn.Embedding.from_pretrained(pretrained_cell_embeddings, freeze=True)
        for param in embed_c.parameters():
            param.requires_grad = False
    elif pretrain_mode == "pt":
        pretrained_cell_embeddings = load_pretrained(pretrain_file, vocab_size)
        embed = nn.Embedding.from_pretrained(pretrained_cell_embeddings, freeze=False)
        for param in embed.parameters():
            param.requires_grad = False
    else:
        print("Unknown pretrain marker!")
    return embed_c


def get_model(model_name, vocab_size, embed_size, hidden_size, dropout, num_heads, num_layers, bidirectional, pretrain_mode="np", pretrain_file=None):
    if model_name == "clear-RNN":
        from model.NaiveRNN import RNNEncoder
        embed_s = SEmbed(vocab_size, embed_size)
        embed_c = load_pretrain(pretrain_mode, pretrain_file, vocab_size)
        encoder = RNNEncoder(None, embed_s, embed_c, hidden_size, bidirectional, dropout, num_layers)
        projector = ProjectionHead(embed_size, int(embed_size / 2), int(embed_size / 4), batch_norm=True)
        model = CLEAR(encoder=encoder, projector=projector)
        return model

    elif model_name == "clear-DualRNN":
        from model.DualRNN import DualRNNEncoder
        embed_s = SEmbed(vocab_size, embed_size)
        embed_c = load_pretrain(pretrain_mode, pretrain_file, vocab_size)
        encoder = DualRNNEncoder(None, embed_s, embed_c, hidden_size, bidirectional, dropout, num_layers)
        projector = ProjectionHead(embed_size, int(embed_size / 2), int(embed_size / 4), batch_norm=True)
        model = CLEAR(encoder=encoder, projector=projector)
        return model

    elif model_name == "clear-Trans":
        from model.transformer import TransEncoder, MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, \
            EncoderLayer
        attn = MultiHeadedAttention(num_heads, embed_size)
        ff = PositionwiseFeedForward(embed_size, int(embed_size * 2), dropout)
        position = PositionalEncoding(embed_size, dropout)
        projector = ProjectionHead(embed_size, int(embed_size / 2), int(embed_size / 4), batch_norm=True)
        c = copy.deepcopy

        encoder = TransEncoder(embed_s=SEmbed(vocab_size, embed_size),
                               embed_p=c(position),
                               layer=EncoderLayer(embed_size, c(attn), c(ff), dropout),
                               num_layers=num_layers)
        return CLEAR(encoder=encoder, projector=projector)


def init_parameters(model):
    from torch.nn import init
    for param in model.parameters():
        if len(param.shape) >= 2:
            init.orthogonal_(param.data)
        else:
            init.normal_(param.data)


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint)
    if is_best:
        shutil.copyfile(args.checkpoint, f"data/{args.dataset_name}/{args.suffix}_best.pt")


def validate(args, valid_datasets, model):
    """
    valData (DataLoader)
    """
    model.eval()
    valid_datasets.start = 0
    num_iteration = 0

    total_loss = 0
    while True:
        s_list, lengths_list = valid_datasets.get_one_batch()
        if s_list is None:
            break
        with torch.no_grad():
            num_iteration += 1
            if args.cuda and torch.cuda.is_available():
                s_list = [s.cuda() for s in s_list]
            output_list = [model(s, lengths=lengths) for s, lengths in zip(s_list, lengths_list)]
            h_list, z_list = [output[0] for output in output_list], [output[1] for output in output_list]
            features = torch.cat(z_list, dim=0).to(args.device)
            criteration = get_loss(args.loss)
            total_loss += criteration(args.batch_size, args.n_views, args.temperature, features, args.device)
    # switch back to training mode
    model.train()
    return total_loss / num_iteration


def train(args):
    logging.basicConfig(filename=f"data/{args.dataset_name}/{args.suffix}_training.log", filemode='a',
                        level=logging.INFO)

    train_datasets = ClearDataLoader(dataset_name=args.dataset_name,
                                     num_train=args.num_train,
                                     num_val=args.num_val,
                                     transformer_list=args.transformer_list,
                                     istrain=True,
                                     batch_size=args.batch_size,
                                     partition_method=args.partition_method,
                                     cell_size=args.cell_size,
                                     minfreq=args.minfreq,
                                     shuffle=args.shuffle)
    print("Loading train datasets...")
    train_datasets.load()
    print(f"Train dataset size: {train_datasets.num_s}")

    valid_datasets = ClearDataLoader(dataset_name=args.dataset_name,
                                     num_train=args.num_train,
                                     num_val=args.num_val,
                                     transformer_list=args.transformer_list,
                                     istrain=False,
                                     batch_size=args.batch_size,
                                     partition_method=args.partition_method,
                                     cell_size=args.cell_size,
                                     minfreq=args.minfreq,
                                     shuffle=args.shuffle)
    print("Loading valid data...")
    valid_datasets.load()
    print(f"Valid dataset size: {valid_datasets.num_s}")

    # model
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

    print(f"model {model}")

    # training on GPU
    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        model.cuda()
    else:
        print("=> training with CPU")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_degrade_step, gamma=args.lr_degrade_gamma)

    # load model state and optmizer state
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        logging.info("Restore training @ {}".format(time.time()))
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        start_iteration = checkpoint["iteration"] + 1
        best_valid_loss = checkpoint["best_valid_loss"]
        status = checkpoint["status"]  # The best loss has been stayed for $status iterations
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        logging.info("Start training @ {}".format(time.time()))
        best_valid_loss = float('inf')
        print("=> initializing the parameters...")
        init_parameters(model)
        start_epoch = 0
        start_iteration = 0
        status = 0
    start = time.time()
    epoch_start = time.time()
    iteration_start = time.time()
    for current_epoch in range(start_epoch, args.epochs):
        if current_epoch != start_epoch:
            train_datasets.shuffle = True
            start_iteration = 0
            status = 0
        print(
            f"Epoch {current_epoch} is training. {train_datasets.num_s // args.batch_size - start_iteration} iterations will be done")
        train_datasets.start = start_iteration * args.batch_size
        while True:
            optimizer.zero_grad()
            # batch dataset
            s_list, lengths_list = train_datasets.get_one_batch()
            # print(f"Iteration {start_iteration}: max length is {[max(i) for i in lengths_list]}")
            if s_list is None:
                print(f"Current epoch finished. ")
                break
            if args.cuda and torch.cuda.is_available():
                s_list = [s.cuda() for s in s_list]

            # encoder
            output_list = [model(s, lengths=lengths) for s, lengths in zip(s_list, lengths_list)]
            h_list, z_list = [output[0] for output in output_list], [output[1] for output in output_list]
            features = torch.cat(z_list, dim=0).to(args.device)

            # InfoNCE Loss
            criteration = get_loss(args.loss)
            loss = criteration(args.batch_size, args.n_views, args.temperature, features, args.device)

            # compute the gradients
            loss.backward()

            # clip the gradients
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # print
            if start_iteration % args.print_freq == 0:
                print(
                    f"Iteration {start_iteration}: Contrastive loss {loss:3f}, each iteration costs {(time.time() - iteration_start) / args.print_freq:.4f} s")
                iteration_start = time.time()
            if start_iteration % args.save_freq == 0 and start_iteration > 0:
                valid_loss = validate(args, valid_datasets, model)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    logging.info(
                        f"Best model with loss {best_valid_loss} at iteration {start_iteration}, epoch {current_epoch}")
                    is_best = True
                    status = 0
                else:
                    is_best = False
                    status += args.save_freq
                    if status >= args.best_threshold:
                        print(f"No improvement after {args.best_threshold}, training stops")
                        break
                print(
                    f"Saving the model at iteration {start_iteration}, epoch {current_epoch}, validation loss {valid_loss}")
                save_checkpoint({"epoch": current_epoch,
                                 "iteration": start_iteration,
                                 "best_valid_loss": best_valid_loss,
                                 "status": status,
                                 "model": model.state_dict(),
                                 "optimizer": optimizer.state_dict()
                                 }, is_best, args)
            start_iteration += 1

        scheduler.step()
        print(f"Epoch {current_epoch} costs: {time.time() - epoch_start:.4f} s")
        epoch_start = time.time()
    print(f"Training costs: {time.time() - start:.4f}s")
