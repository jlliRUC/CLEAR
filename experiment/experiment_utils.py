import sys
sys.path.append("..")
import os
import numpy as np
from preprocess.augmentation import data_augmentation
import pickle
import functools
import random
random.seed(0)
import torch
from evaluation import model_generator
import copy
from scipy.stats import spearmanr


def seq2str(seq):
    result = ''
    for item in seq:
        result += (str(item) + ' ')
    result += '\n'
    return result


def uniformsplit(traj):
    n = traj.shape[0]
    idx1 = [i for i in range(0, n, 2)]
    idx2 = [i + 1 for i in range(0, n - 1, 2)]

    return traj[idx1, :], traj[idx2, :]


def multi_encode(args, exps):
    with torch.no_grad():
        model = model_generator(args.model_name)
        # for exp_type in ["self-similarity", "cross-similarity", "KNN"]:
        for exp_type in exps:
            if args.partition_method == "grid":
                source_path = f"{exp_type}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_original_token.pkl"
                vec_path = f"{exp_type}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_original_vec_{args.suffix}.pkl"
            model(args, source_path, vec_path)
            for name in ["distort", "downsampling"]:
                # for name in ["distort"]:
                for rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    if args.partition_method == "grid":
                        source_path = f"{exp_type}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{name}_rate_{str(rate)}_token.pkl"
                        vec_path = f"{exp_type}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{name}_rate_{str(rate)}_vec_{args.suffix}.pkl"
                    model(args, source_path, vec_path)


def createQueryKey(traj_file, start, query_size, key_size, transformer, querykey_file, label_file, min_length, max_length):

    num_query, num_key = 0, 0
    with open(traj_file, "rb") as f:
        traj_df = pickle.load(f)
    query_list = []
    key_list = []
    query_label_list = []
    key_label_list = []

    for i in range(start, traj_df.shape[0]):
        location = traj_df.loc[i, "Locations"]
        if num_query < query_size:
            if min_length <= location.shape[0] <= max_length:
                location_q, location_k = uniformsplit(location)
                query_list.append(transformer(location_q))
                key_list.append(transformer(location_k))
                query_label_list.append(i)
                key_label_list.append(i)
                num_query += 1
                num_key += 1
        elif num_key < key_size:
            if min_length <= location.shape[0] <= max_length:
                location_q, location_k = uniformsplit(location)  ## Here we drop out half of key_pair to make the sequence length be similar.
                key_list.append(transformer(location_k))
                key_label_list.append(i)
                num_key += 1
        else:
            break

    with open(querykey_file, "wb") as f:
        pickle.dump((query_list + key_list), f)
    with open(label_file, "wb") as f:
        pickle.dump((query_label_list + key_label_list), f)

    return num_query, num_key


def createToken(region, traj_file, token_file):
    with open(traj_file, "rb") as f:
        traj_list = pickle.load(f)
    token_list = [region.spatial_encoding(traj) for traj in traj_list]

    with open(token_file, "wb") as f:
        pickle.dump(token_list, f)

    return


def data_self_similarity(args, region):
    exp_label = "self-similarity"
    for name in args.name_list:
        if name == "original":
            transformer = data_augmentation(name)
            suffix = name
            if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                os.makedirs(f"{exp_label}/{args.dataset_name}")
            if args.partition_method == "grid":
                querykey_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_querykey.pkl"
                token_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_token.pkl"
                label_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_label.pkl"
            print(f"CreateQueryKey for {suffix}")
            createQueryKey(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.pkl',
                           start=args.exp_start,
                           query_size=args.num_query_ss,
                           key_size=args.num_key_ss,
                           transformer=transformer,
                           querykey_file=querykey_file,
                           label_file=label_file,
                           min_length=args.min_length,
                           max_length=args.max_length)
            print(f"CreateToken for {suffix}")
            createToken(region, querykey_file, token_file)
        else:
            for rate in args.rate_list:
                transformer = functools.partial(data_augmentation(name), rate=rate)
                suffix = name + "_" + "rate" + "_" + str(rate)
                if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                    os.makedirs(f"{exp_label}/{args.dataset_name}")
                if args.partition_method == "grid":
                    querykey_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_querykey.pkl"
                    token_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_token.pkl"
                    label_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_label.pkl"
                print(f"CreateQueryKey for {suffix}")
                createQueryKey(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.pkl',
                               start=args.exp_start,
                               query_size=args.num_query_ss,
                               key_size=args.num_key_ss,
                               transformer=transformer,
                               querykey_file=querykey_file,
                               label_file=label_file,
                               min_length=args.min_length,
                               max_length=args.max_length)
                print(f"CreateToken for {suffix}")
                createToken(region, querykey_file, token_file)


def exp_self_similarity(args):
    results = {}
    transformer_list = ["original"]
    name_list = copy.deepcopy(args.name_list)
    name_list.remove("original")
    for name in name_list:
        for rate in args.rate_list:
            transformer_list.append(f"{name}_rate_{str(rate)}")
    for suffix in transformer_list:
        results[suffix] = {}
        if args.partition_method == "grid":
            vec_file = f"self-similarity/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_vec_{args.suffix}.pkl"
        with open(vec_file, "rb") as f:
            vec_f = pickle.load(f)
        vecs = vec_f['vec']
        querys, keys = vecs[0:args.num_query_ss, :], vecs[args.num_query_ss:, :]
        dists = torch.cdist(querys, keys, p=2)  # [num_query, num_key]
        targets = torch.diag(dists)  # [num_query]

        for key_size in args.key_sizes:
            rank = torch.sum(torch.le(dists[:, 0:key_size].T, targets)).item() / args.num_query_ss
            results[suffix][str(key_size)] = round(rank, 4)
            print(f"{suffix}: Mean rank {round(rank, 4)} with key size {key_size}")
    return results


def createPair(traj_file, start, num_pair, transformer, pair_file, min_length, max_length):
    with open(traj_file, "rb") as f:
        traj_df = pickle.load(f)
    num = 0
    list1 = []
    while num < num_pair:
        location1 = traj_df.loc[start + num, "Locations"]
        if min_length <= location1.shape[0] <= max_length:
            list1.append(transformer(location1))
            num += 1
    num = 0
    list2 = []
    while num < num_pair:
        location2 = traj_df.loc[traj_df.shape[0] - 1 - num, "Locations"]
        if min_length <= location2.shape[0] <= max_length:
            list2.append(transformer(location2))
            num += 1

    with open(pair_file, "wb") as f:
        pickle.dump(list1+list2, f)
    return num_pair


def data_cross_similarity(args, region):
    exp_label = "cross-similarity"
    for name in args.name_list:
        if name == "original":
            transformer = data_augmentation(name)
            suffix = name
            if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                os.makedirs(f"{exp_label}/{args.dataset_name}")
            if args.partition_method == "grid":
                pair_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_pair.pkl"
                token_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_token.pkl"
            print(f"CreatePair for {suffix}")
            createPair(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.pkl',
                       start=args.exp_start,
                       num_pair=args.num_pair,
                       transformer=transformer,
                       pair_file=pair_file,
                       min_length=args.min_length,
                       max_length=args.max_length)
            print(f"CreatePairToken for {suffix}")
            createToken(region, pair_file, token_file)
        else:
            for rate in args.rate_list:
                transformer = functools.partial(data_augmentation(name), rate=rate)
                suffix = name + "_" + "rate" + "_" + str(rate)
                if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                    os.makedirs(f"{exp_label}/{args.dataset_name}")
                if args.partition_method == "grid":
                    pair_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_pair.pkl"
                    token_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_token.pkl"
                print(f"CreatePair for {suffix}")
                createPair(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.pkl',
                           start=args.exp_start,
                           num_pair=args.num_pair,
                           transformer=transformer,
                           pair_file=pair_file,
                           min_length=args.min_length,
                           max_length=args.max_length)
                print(f"CreatePairToken for {suffix}")
                createToken(region, pair_file, token_file)


def exp_cross_similarity(args):
    if args.partition_method == "grid":
        ori_file = f"cross-similarity/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_original_vec_{args.suffix}.pkl"
    with open(ori_file, "rb") as f:
        ori_f = pickle.load(f)
    vecs_ori = ori_f['vec']
    vecs_ori1, vecs_ori2 = vecs_ori[0:args.num_pair, :], vecs_ori[args.num_pair:, :]
    transformer_list = []
    name_list = copy.deepcopy(args.name_list)
    name_list.remove("original")
    for name in name_list:
        for rate in args.rate_list:
            transformer_list.append(f"{name}_rate_{str(rate)}")
    results = {f"{args.num_pair}": {}}
    for transformer in transformer_list:
        if args.partition_method == "grid":
            transformed_file = f"cross-similarity/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{transformer}_vec_{args.suffix}.pkl"
        with open(transformed_file, "rb") as f:
            transformed_f = pickle.load(f)
        vecs_transformed = transformed_f['vec']
        vecs_transformed1, vecs_transformed2 = vecs_transformed[0:args.num_pair, :], vecs_transformed[args.num_pair:, :]
        print(vecs_transformed1.shape, vecs_transformed2.shape)
        score = []
        for i in range(args.num_pair):
            distance_transformed = args.distance(vecs_transformed1[i].detach().cpu().numpy(), vecs_transformed2[i].detach().cpu().numpy())
            distance_ori = args.distance(vecs_ori1[i].detach().cpu().numpy(), vecs_ori2[i].detach().cpu().numpy())
            score.append(abs(distance_transformed - distance_ori) / distance_ori)
        print(f"{transformer}, {len(score)}, {np.mean(score)}")
        results[f"{args.num_pair}"][transformer] = round(np.mean(score), 4)
        print(f"cross_similarity of {transformer} is {np.mean(score)}.")
    return results


def createKNN(traj_file, start, query_size, key_size, transformer, querykey_file, label_file, min_length, max_length):
    """
    :param traj_file:
    :param start: start 是前面训练数据集跟测试数据集的终点，保证这部分数据不参与训练或验证
    :param query_size:
    :param key_size:
    :param transformer:
    :param querydb_file:
    :param min_length:
    :param max_length:
    :return:
    """
    num_query, num_key = 0, 0
    with open(traj_file, "rb") as f:
        traj_df = pickle.load(f)
    query_list = []
    key_list = []
    query_label_list = []
    key_label_list = []

    for i in range(start, traj_df.shape[0]):
        location = traj_df.loc[i, "Locations"]
        if num_query < query_size:
            if min_length <= location.shape[0] <= max_length:
                query_list.append(transformer(location))
                query_label_list.append(i)
                num_query += 1
        elif num_key < key_size:
            if min_length <= location.shape[0] <= max_length:
                key_list.append(transformer(location))
                key_label_list.append(i)
                num_key += 1
        else:
            break

    with open(querykey_file, "wb") as f:
        pickle.dump((query_list + key_list), f)
    with open(label_file, "wb") as f:
        pickle.dump((query_label_list + key_label_list ), f)

    return num_query, num_key


def data_KNN(args, region):
    exp_label = "knn"
    for name in args.name_list:
        if name == "original":
            transformer = data_augmentation(name)
            suffix = name
            if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                os.makedirs(f"{exp_label}/{args.dataset_name}")
            if args.partition_method == "grid":
                querykey_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_querykey.pkl"
                token_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_token.pkl"
                label_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_label.pkl"
            print(f"CreateKNN for {suffix}")
            createKNN(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.pkl',
                      start=args.exp_start,
                      query_size=args.num_query_knn,
                      key_size=args.num_key_knn,
                      transformer=transformer,
                      querykey_file=querykey_file,
                      label_file=label_file,
                      min_length=args.min_length,
                      max_length=args.max_length)
            print(f"CreateToken for {suffix}")
            createToken(region, querykey_file, token_file)
        else:
            for rate in args.rate_list:
                transformer = functools.partial(data_augmentation(name), rate=rate)
                suffix = name + "_" + "rate" + "_" + str(rate)
                if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                    os.makedirs(f"{exp_label}/{args.dataset_name}")
                if args.partition_method == "grid":
                    querykey_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_querykey.pkl"
                    token_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_token.pkl"
                    label_file = f"{exp_label}/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_label.pkl"
                print(f"CreateKNN for {suffix}")
                createKNN(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.pkl',
                          start=args.exp_start,
                          query_size=args.num_query_knn,
                          key_size=args.num_key_knn,
                          transformer=transformer,
                          querykey_file=querykey_file,
                          label_file=label_file,
                          min_length=args.min_length,
                          max_length=args.max_length)
                print(f"CreateToken for {suffix}")
                createToken(region, querykey_file, token_file)


def exp_KNN(args):
    if args.partition_method == "grid":
        ori_file = f"knn/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_original_vec_{args.suffix}.pkl"
    with open(ori_file, "rb") as f:
        ori_f = pickle.load(f)
    vecs_ori = ori_f['vec']
    query_ori, key_ori = vecs_ori[0:args.num_query_knn, :], vecs_ori[args.num_query_knn:, :]
    result_ori = knnsearch(query_ori, key_ori, args.max_k)
    print(f"query {query_ori.shape}, key {key_ori.shape}")
    transformer_list = []
    name_list = copy.deepcopy(args.name_list)
    name_list.remove("original")
    for name in name_list:
        for rate in args.rate_list:
            transformer_list.append(f"{name}_rate_{str(rate)}")

    results = {}
    for transformer in transformer_list:
        if args.partition_method == "grid":
            transformed_file = f"knn/{args.dataset_name}/cellsize-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{transformer}_vec_{args.suffix}.pkl"
        results[transformer] = {}
        with open(transformed_file, "rb") as f:
            transformed_f = pickle.load(f)
        vecs_transformed = transformed_f['vec']
        query_transformed, key_transformed = vecs_transformed[0:args.num_query_knn, :], vecs_transformed[
                                                                                        args.num_query_knn:, :]
        result_transformed = knnsearch(query_transformed, key_transformed, args.max_k)
        for k in args.k_list:
            score1, score2 = topkscore(result_ori, result_transformed, k)
            print(f"{k}NN score of {transformer} is {round(score1, 4), round(score2, 4)}")
            results[transformer][str(k)] = (round(score1, 4), round(score2, 4))
    return results


def knnsearch(query, key, k):
    dist = torch.cdist(query, key, p=2)
    sorted, idxs = torch.sort(dist)

    return idxs[:, :k]


def topkscore(result1, result2, k):
    result1 = result1.detach().cpu().numpy()
    result2 = result2.detach().cpu().numpy()
    intersection_score = [np.intersect1d(result1[i, :k], result2[i, :k]).shape[0] / k for i in range(result1.shape[0])]
    spearman_score = [spearmanr(result1[i, :k], result2[i, :k]) for i in range(result1.shape[0])]
    return np.array(intersection_score).mean(), np.array(spearman_score).mean()
