import sys
sys.path.append("..")
import os
import h5py
import numpy as np
from augmentation.data_augmentation import data_augmentation
import functools
import random
random.seed(0)
import torch
from evaluation import model_generator
import copy


def seq2str(seq):
    result = ''
    for item in seq:
        result += (str(item) + ' ')
    result += '\n'
    return result


def uniformsplit(traj, timestamp):
    n = traj.shape[0]
    idx1 = [i for i in range(0, n, 2)]
    idx2 = [i + 1 for i in range(0, n - 1, 2)]

    return traj[idx1, :], timestamp[idx1], traj[idx2, :], timestamp[idx2]


def multi_encode(args, exps):
    with torch.no_grad():
        model = model_generator(args.model_name)
        for exp_type in exps:
            if args.spatial_type == "grid":
                source_path = f"{exp_type}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_original_seq.h5"
                vec_path = f"{exp_type}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_original_vec_{args.file_suffix}.h5"
            model(args, source_path, vec_path)
            for name in ["distort", "downsampling", "interpolation"]:
                for rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    if args.spatial_type == "grid":
                        source_path = f"{exp_type}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{name}_rate_{str(rate)}_seq.h5"
                        vec_path = f"{exp_type}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{name}_rate_{str(rate)}_vec_{args.file_suffix}.h5"
                    model(args, source_path, vec_path)


def createQueryKey(traj_file, start, query_size, key_size, transformer, querykey_file, min_length, max_length):
    """

    :param traj_file:
    :param start: start
    :param query_size:
    :param key_size:
    :param transformer:
    :param querykey_file:
    :param min_length:
    :param max_length:
    :return:
    """

    num_query, num_key = 0, 0
    traj_df = h5py.File(traj_file, "r")
    querykey_df = h5py.File(querykey_file, 'w')
    num = traj_df.attrs['num']
    query_flag = True
    key_flag = True
    for i in range(start, num):
        location = traj_df[f"/Locations/{i}"][:]
        timestamp = traj_df[f"/Timestamps/{i}"][:]
        if num_query < query_size:
            if 2 * min_length <= location.shape[0] <= 2 * max_length:
                location_q, timestamp_q, location_k, timestamp_k = uniformsplit(location, timestamp)
                querykey_df[f"/query/Locations/{num_query}"], querykey_df[f"/query/Timestamps/{num_query}"] = transformer(
                    location_q, timestamp_q)
                querykey_df[f"/query/names/{num_query}"] = i
                querykey_df[f"/key/Locations/{num_key}"], querykey_df[f"/key/Timestamps/{num_key}"] = transformer(
                    location_k, timestamp_k)
                querykey_df[f"/key/names/{num_key}"] = i
                num_query += 1
                query_flag = True
                num_key += 1
                key_flag = True
        elif num_key < key_size:
            if 2 * min_length <= location.shape[0] <= 2 * max_length:
                location_q, timestamp_q, location_k, timestamp_k = uniformsplit(location,
                                                                                timestamp)  ## Here we drop out half of key_pair to make the sequence length be similar.
                querykey_df[f"/key/Locations/{num_key}"], querykey_df[f"/key/Timestamps/{num_key}"] = transformer(
                    location_k, timestamp_k)
                querykey_df[f"/key/names/{num_key}"] = i
                num_key += 1
                key_flag = True
        else:
            break
        if num_query % 100 == 0 and query_flag:
            print(f"{num_query} query saved.")
            query_flag = False
        if num_key % 1000 == 0 and key_flag:
            print(f"{num_key} key saved.")
            key_flag = False
    querykey_df.attrs["num_query"] = num_query
    querykey_df.attrs["num_key"] = num_key
    querykey_df.close()

    return num_query, num_key


def createToken(region, querykey_file, token_file, label_file):
    querykey_df = h5py.File(querykey_file, "r")
    labels = []
    with h5py.File(token_file, 'w') as f:
        num_query, num_key = querykey_df.attrs["num_query"], querykey_df.attrs["num_key"]
        for i in range(0, num_query + num_key):
            if i < num_query:
                label = "query"
                idx = i
            else:
                label = "key"
                idx = i - num_query
            f[f"/Locations/{i}"] = region.spatial_encoding(querykey_df[f"{label}/Locations/{idx}"])
            f[f"/Timestamps/{i}"] = np.array(querykey_df[f"{label}/Timestamps/{idx}"])
            name = querykey_df[f"{label}/names/{idx}"]
            labels.append(np.array(name))
        f.attrs["num"] = num_query + num_key
    f.close()
    with open(label_file, 'w') as f:
        for item in labels:
            f.write(str(item) + '\n')
    f.close()
    querykey_df.close()
    return len(labels)


def rank_search_vec(query, querylabel, key, key_label, distance):
    """
    for each traj in query, compute the rank of its twin traj in keys.
    :param query:
    :param querylabel:
    :param key:
    :param key_label:
    :param distance:
    :return:
    """
    assert query.shape[0] == len(querylabel), "unmatched query and label"
    assert key.shape[0] == len(key_label), "unmatched key and label"

    def rank(x, xlabel):
        dists = []
        for key_vec in key:
            dists.append(distance(x, key_vec))
        idxs = sorted(range(len(dists)), key=lambda k: dists[k])
        neighbors = [key_label[item] for item in idxs]
        return neighbors.index(xlabel) + 1

    ranks = []
    for i in range(0, len(querylabel)):
        ranks.append(rank(query[i, :], querylabel[i]))
    return ranks


def data_self_similarity(args, region):
    exp_label = "self-similarity"
    for name in args.name_list:
        if name == "original":
            transformer = data_augmentation(name)
            suffix = name
            if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                os.makedirs(f"{exp_label}/{args.dataset_name}")
            if args.spatial_type == "grid":
                querykey_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_querykey.h5"
                token_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_seq.h5"
                label_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}.label"
            print(f"CreateQueryKey for {suffix}")
            createQueryKey(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.h5',
                           start=args.start,
                           query_size=args.num_query_ss,
                           key_size=args.num_key_ss,
                           transformer=transformer,
                           querykey_file=querykey_file,
                           min_length=args.min_length,
                           max_length=args.max_length)
            print(f"CreateToken for {suffix}")
            createToken(region, querykey_file, token_file, label_file)
        else:
            for rate in args.rate_list:
                if name == "distort":
                    f = h5py.File(f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.h5', 'r')
                    transformer = functools.partial(data_augmentation(name), radius_ts=f.attrs["mean_interval"],
                                                    rate=rate)
                else:
                    transformer = functools.partial(data_augmentation(name), rate=rate)
                suffix = name + "_" + "rate" + "_" + str(rate)
                if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                    os.makedirs(f"{exp_label}/{args.dataset_name}")
                if args.spatial_type == "grid":
                    querykey_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_querykey.h5"
                    token_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_seq.h5"
                    label_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}.label"
                print(f"CreateQueryKey for {suffix}")
                createQueryKey(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.h5',
                               start=args.start,
                               query_size=args.num_query_ss,
                               key_size=args.num_key_ss,
                               transformer=transformer,
                               querykey_file=querykey_file,
                               min_length=args.min_length,
                               max_length=args.max_length)
                print(f"CreateToken for {suffix}")
                createToken(region, querykey_file, token_file, label_file)


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
        if args.spatial_type == "grid":
            vec_file = f"self-similarity/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_vec_{args.file_suffix}.h5"
            label_file = f"self-similarity/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}.label"
        vec_f = h5py.File(vec_file, "r")
        vecs = np.array(vec_f['vec'])
        with open(label_file) as label_f:
            labels = label_f.readlines()
        query, key = vecs[0:args.num_query_ss, :], vecs[args.num_query_ss:, :]
        query_label, key_label = labels[0:args.num_query_ss], labels[args.num_query_ss:]
        for key_size in args.key_sizes:
            ranks = rank_search_vec(query, query_label, key[:key_size], key_label[:key_size],
                                    lambda x, y: np.linalg.norm(x - y))
            results[suffix][str(key_size)] = round(np.mean(ranks), 4)
            print(f"{suffix}: Mean rank {round(np.mean(ranks), 4)} with key size {key_size}")
        vec_f.close()
        label_f.close()
    return results


def createPair(traj_file, start, num_pair, transformer, pair_file, min_length, max_length):
    traj_df = h5py.File(traj_file, "r")
    num = traj_df.attrs['num']
    pair_df = h5py.File(pair_file, 'w')
    for i in range(num_pair):
        location1 = traj_df[f"/Locations/{start + i}"][:]
        timestamp1 = traj_df[f"/Timestamps/{start + i}"][:]
        pair_df[f"/Locations1/{i}"], pair_df[f"/Timestamps1/{i}"] = transformer(location1, timestamp1)
        pair_df[f"/names1/{i}"] = i
        location2 = traj_df[f"/Locations/{num - 10 - i}"][:]
        timestamp2 = traj_df[f"/Timestamps/{num - 10 - i}"][:]
        pair_df[f"/Locations2/{i}"], pair_df[f"/Timestamps2/{i}"] = transformer(location2, timestamp2)
        pair_df[f"/names2/{i}"] = num - 10 - i
        if i % 100 == 0:
            print(f"{i} pairs saved.")
    pair_df.attrs["num"] = num_pair
    pair_df.close()
    return num_pair


def createPairToken(region, pair_file, token_file, label_file):
    pair_df = h5py.File(pair_file, "r")
    label = []
    with h5py.File(token_file, 'w') as f:
        num = pair_df.attrs["num"]
        for i in range(0, num * 2):
            if i < num:
                location_label = "Locations1"
                timestamp_label = "Timestamps1"
                name_label = "names1"
                idx = i
            else:
                location_label = "Locations2"
                timestamp_label = "Timestamps2"
                name_label = "names2"
                idx = i - num
            f[f"/Locations/{i}"] = region.spatial_encoding(pair_df[f"{location_label}/{idx}"])
            f[f"/Timestamps/{i}"] = np.array(pair_df[f"{timestamp_label}/{idx}"])
            name = pair_df[f"{name_label}/{idx}"]
            label.append(np.array(name))
        f.attrs["num"] = num * 2
    f.close()
    with open(label_file, 'w') as f:
        for item in label:
            f.write(str(item) + '\n')
    f.close()
    pair_df.close()
    return len(label)


def data_cross_similarity(args, region):
    exp_label = "cross-similarity"
    for name in args.name_list:
        if name == "original":
            transformer = data_augmentation(name)
            suffix = name
            if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                os.makedirs(f"{exp_label}/{args.dataset_name}")
            if args.spatial_type == "grid":
                pair_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_pair.h5"
                token_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_seq.h5"
                label_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}.label"
            print(f"CreatePair for {suffix}")
            createPair(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.h5',
                       start=args.start,
                       num_pair=args.num_pair,
                       transformer=transformer,
                       pair_file=pair_file,
                       min_length=args.min_length,
                       max_length=args.max_length)  # 100 for porto, 10000 for geolife
            print(f"CreatePairToken for {suffix}")
            createPairToken(region, pair_file, token_file, label_file)
        else:
            for rate in args.rate_list:
                if name == "distort":
                    f = h5py.File(f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.h5', 'r')
                    transformer = functools.partial(data_augmentation(name), radius_ts=f.attrs["mean_interval"],
                                                    rate=rate)
                else:
                    transformer = functools.partial(data_augmentation(name), rate=rate)
                suffix = name + "_" + "rate" + "_" + str(rate)
                if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                    os.makedirs(f"{exp_label}/{args.dataset_name}")
                if args.spatial_type == "grid":
                    pair_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_pair.h5"
                    token_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_seq.h5"
                    label_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}.label"
                print(f"CreatePair for {suffix}")
                createPair(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.h5',
                           start=args.start,
                           num_pair=args.num_pair,
                           transformer=transformer,
                           pair_file=pair_file,
                           min_length=args.min_length,
                           max_length=args.max_length)
                print(f"CreatePairToken for {suffix}")
                createPairToken(region, pair_file, token_file, label_file)


def exp_cross_similarity(args):
    if args.spatial_type == "grid":
        ori_file = f"cross-similarity/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_original_vec_{args.file_suffix}.h5"
    ori_f = h5py.File(ori_file, "r")
    vecs_ori = np.array(ori_f['vec'])
    vecs_ori1, vecs_ori2 = vecs_ori[0:args.num_pair, :], vecs_ori[args.num_pair:, :]
    transformer_list = []
    name_list = copy.deepcopy(args.name_list)
    name_list.remove("original")
    for name in name_list:
        for rate in args.rate_list:
            transformer_list.append(f"{name}_rate_{str(rate)}")
    results = {f"{args.num_pair}": {}}
    for transformer in transformer_list:
        if args.spatial_type == "grid":
            transformed_file = f"cross-similarity/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{transformer}_vec_{args.file_suffix}.h5"
        transformed_f = h5py.File(transformed_file, "r")
        vecs_transformed = np.array(transformed_f['vec'])
        vecs_transformed1, vecs_transformed2 = vecs_transformed[0:args.num_pair, :], vecs_transformed[args.num_pair:, :]
        print(vecs_transformed1.shape, vecs_transformed2.shape)
        score = []
        for i in range(args.num_pair):
            distance_transformed = args.distance(vecs_transformed1[i], vecs_transformed2[i])
            distance_ori = args.distance(vecs_ori1[i], vecs_ori2[i])
            score.append(abs(distance_transformed - distance_ori) / distance_ori)
        transformed_f.close()
        print(f"{transformer}, {len(score)}, {np.mean(score)}")
        results[f"{args.num_pair}"][transformer] = round(np.mean(score), 4)
        print(f"cross_similarity of {transformer} is {np.mean(score)}.")
    ori_f.close()
    return results


def createKNN(traj_file, start, query_size, key_size, transformer, querykey_file, min_length, max_length):
    """
    :param traj_file:
    :param start: start
    :param query_size:
    :param key_size:
    :param transformer:
    :param querydb_file:
    :param min_length:
    :param max_length:
    :return:
    """
    num_query, num_key = 0, 0
    traj_df = h5py.File(traj_file, "r")
    querykey_df = h5py.File(querykey_file, 'w')
    num = traj_df.attrs['num']
    query_flag = True
    key_flag = True
    for i in range(start, num):
        location = traj_df[f"/Locations/{i}"][:]
        timestamp = traj_df[f"/Timestamps/{i}"][:]
        if num_query < query_size:
            if min_length <= location.shape[0] <= max_length:
                querykey_df[f"/query/Locations/{num_query}"], querykey_df[f"/query/Timestamps/{num_query}"] = transformer(
                    location, timestamp)
                querykey_df[f"/query/names/{num_query}"] = i
                num_query += 1
                query_flag = True
        elif num_key < key_size:
            if min_length <= location.shape[0] <= max_length:
                querykey_df[f"/key/Locations/{num_key}"], querykey_df[f"/key/Timestamps/{num_key}"] = transformer(
                    location, timestamp)
                querykey_df[f"/key/names/{num_key}"] = i
                num_key += 1
                key_flag = True
        else:
            break
        if num_query % 100 == 0 and query_flag:
            print(f"{num_query} query saved.")
            query_flag = False
        if num_key % 1000 == 0 and key_flag:
            print(f"{num_key} key saved.")
            key_flag = False
    querykey_df.attrs["num_query"] = num_query
    querykey_df.attrs["num_key"] = num_key
    querykey_df.close()

    return num_query, num_key


def data_KNN(args, region):
    exp_label = "knn"
    for name in args.name_list:
        if name == "original":
            transformer = data_augmentation(name)
            suffix = name
            if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                os.makedirs(f"{exp_label}/{args.dataset_name}")
            if args.spatial_type == "grid":
                querykey_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_querykey.h5"
                token_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_seq.h5"
                label_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}.label"
            print(f"CreateKNN for {suffix}")
            createKNN(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.h5',
                      start=args.start,
                      query_size=args.num_query_knn,
                      key_size=args.num_key_knn,
                      transformer=transformer,
                      querykey_file=querykey_file,
                      min_length=args.min_length,
                      max_length=args.max_length)
            print(f"CreateToken for {suffix}")
            createToken(region, querykey_file, token_file, label_file)
        else:
            for rate in args.rate_list:
                if name == "distort":
                    f = h5py.File(f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.h5', 'r')
                    transformer = functools.partial(data_augmentation(name), radius_ts=f.attrs["mean_interval"],
                                                    rate=rate)
                else:
                    transformer = functools.partial(data_augmentation(name), rate=rate)
                suffix = name + "_" + "rate" + "_" + str(rate)
                if not os.path.exists(f"{exp_label}/{args.dataset_name}"):
                    os.makedirs(f"{exp_label}/{args.dataset_name}")
                if args.spatial_type == "grid":
                    querykey_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_querykey.h5"
                    token_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}_seq.h5"
                    label_file = f"{exp_label}/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{suffix}.label"
                print(f"CreateKNN for {suffix}")
                createKNN(traj_file=f'{args.data_path}/{args.dataset_name}/{args.dataset_name}.h5',
                          start=args.start,
                          query_size=args.num_query_knn,
                          key_size=args.num_key_knn,
                          transformer=transformer,
                          querykey_file=querykey_file,
                          min_length=args.min_length,
                          max_length=args.max_length)
                print(f"CreateToken for {suffix}")
                createToken(region, querykey_file, token_file, label_file)


def exp_KNN(args):
    if args.spatial_type == "grid":
        ori_file = f"knn/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_original_vec_{args.file_suffix}.h5"
    ori_f = h5py.File(ori_file, "r")
    vecs_ori = np.array(ori_f['vec'])
    query_ori, key_ori = vecs_ori[0:args.num_query_knn, :], vecs_ori[args.num_query_knn:, :]
    result_ori = knnsearch(query_ori, key_ori, args.max_k, args.distance)
    print(f"query {query_ori.shape}, key {key_ori.shape}")
    transformer_list = []
    name_list = copy.deepcopy(args.name_list)
    name_list.remove("original")
    for name in name_list:
        for rate in args.rate_list:
            transformer_list.append(f"{name}_rate_{str(rate)}")

    results = {}
    for transformer in transformer_list:
        if args.spatial_type == "grid":
            transformed_file = f"knn/{args.dataset_name}/cell-{args.cell_size}_minfreq-{args.minfreq}/{args.dataset_name}_{transformer}_vec_{args.file_suffix}.h5"
        results[transformer] = {}
        transformed_f = h5py.File(transformed_file, "r")
        vecs_transformed = np.array(transformed_f['vec'])
        query_transformed, key_transformed = vecs_transformed[0:args.num_query_knn, :], vecs_transformed[
                                                                                        args.num_query_knn:, :]
        result_transformed = knnsearch(query_transformed, key_transformed, args.max_k, args.distance)
        for k in args.k_list:
            score = topkscore(result_ori, result_transformed, k)
            print(f"{k}NN score of {transformer} is {score}")
            results[transformer][str(k)] = round(score, 4)
        transformed_f.close()
    ori_f.close()
    return results


def knnsearch(query, key, k, distance):
    num_query = query.shape[0]
    results = []
    for i in range(0, num_query):
        dists = []
        for item in key:
            dists.append(distance(query[i], item))
        idxs = sorted(range(len(dists)), key=lambda k: dists[k])
        results.append(idxs[:k])
    return results


def topkscore(result1, result2, k):
    num = len(result1)
    score = []
    for i in range(num):
        score.append(len(list(set(result1[i][:k]).intersection(set(result2[i][:k])))) / k)
    return np.mean(score)

