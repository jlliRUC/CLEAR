import sys
sys.path.append("..")
import os
import pickle
import time
import pandas as pd
import math
import numpy as np
from utils import meters2lonlat, lonlat2meters
from collections import Counter
from sklearn.neighbors import BallTree
import argparse
import multiprocessing
from node2vec import train_node2vec
import torch
from config import Config
import argparse


def parse_args():
    # Set some key configs from screen

    parser = argparse.ArgumentParser(description="preprocess.py")

    parser.add_argument("-dataset_name",  # ["porto", "geolife", "tdrive", "aisus"]
                        help="Name of dataset")

    parser.add_argument("-partition_method",  # ["grid", "quadtree"]
                        help="The way of space partitioning for spatial token_generation")

    parser.add_argument("-aug1_name", type=str)

    parser.add_argument("-aug1_rate")

    parser.add_argument("-aug2_name", type=str)

    parser.add_argument("-aug2_rate")

    parser.add_argument("-combination", type=str,
                        help="Different ways for combine pair or multiple augmentations")

    parser.add_argument("-model_name", type=str,
                        help="The name of clear's variant")

    parser.add_argument("-model_settings", type=str,
                        help="suffix to mark special cases")

    parser.add_argument("-loss", type=str,
                        help="Contrastive Loss Function")

    parser.add_argument("-pretrain_mode", type=str,
                        help="np means no pretrain (by default), pf means pretrain-freeze, pt means pretrain-train")

    parser.add_argument("-pretrain_method", type=str,
                        help="node2vec or word2vec")

    parser.add_argument("-batch_size", type=int,
                        help="The batch size")

    parser.add_argument("-cell_size", type=int)

    parser.add_argument("-minfreq", type=int)

    parser.add_argument("-exp_list", nargs='+')


    args = parser.parse_args()

    params = {}
    for param, value in args._get_kwargs():
        if value is not None:
            params[param] = value

    return params


def rle(seq):
    import itertools
    return [k for k, v in itertools.groupby(seq)]


class GridPartition:
    def __init__(self, traj_file, dataset_name, min_lon, min_lat, max_lon, max_lat, x_step, y_step, minfreq,
                 maxvocab_size, built):
        """
        :param traj_file: path for the trajectory file.
        :param dataset_name: dataset name
        :param min_lon: Boundary, GPS coordinate
        :param min_lat: Boundary, GPS coordinate
        :param max_lon: Boundary, GPS coordinate
        :param max_lat: Boundary, GPS coordinate
        :param x_step: Cell size x (meters)
        :param y_step: Cell size y (meters)
        :param minfreq: hot cell threshold, a hot cell must at least has minfreq points
        :param maxvocab_size: threshold of vocabulary
        :param built: If build ball-tree for hot cell, then True, else False.
        """
        self.traj_file = traj_file
        self.dataset_name = dataset_name
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.min_x, self.min_y = lonlat2meters(min_lon, min_lat)
        self.max_x, self.max_y = lonlat2meters(max_lon, max_lat)
        self.num_x = math.ceil(round((self.max_x - self.min_x), 6) / x_step)
        self.num_y = math.ceil(round((self.max_x - self.min_x), 6) / y_step)
        self.x_step = x_step
        self.y_step = y_step
        self.minfreq = minfreq
        self.maxvocab_size = maxvocab_size
        self.built = built

    def coord2cell(self, x, y):
        """
        Calculate the cell ID of point (x, y) based on the corresponding offsets.
        :param x: spherical meters
        :param y: spherical meters
        :return:
        """
        x_offset = math.floor(round((x - self.min_x), 6) / self.x_step)
        y_offset = math.floor(round((y - self.min_y), 6) / self.y_step)

        return y_offset * self.num_x + x_offset

    def cell2coord(self, cell):
        """
        From cell ID to the coordinates of point in spherical meters. Here the point is actually the cell center.
        :param cell:
        :return:
        """
        y_offset = cell // self.num_x
        x_offset = cell % self.num_x
        y = self.min_y + (y_offset + 0.5) * self.y_step
        x = self.min_x + (x_offset + 0.5) * self.x_step

        return x, y

    def gps2cell(self, lon, lat):
        """
        From point (lon, lat) to cell ID
        :param lon:
        :param lat:
        :return:
        """
        x, y = lonlat2meters(lon, lat)

        return self.coord2cell(x, y)

    def cell2gps(self, cell):
        """
        From cell ID to the coordinates of point in GPS coordinates. Here the point is actually the cell center.
        :param cell:
        :return:
        """
        x, y = self.cell2coord(cell)

        return meters2lonlat(x, y)

    def point_inregion(self, lon, lat):
        """
        If the point (lon, lat) is contained in this domain
        :param lon:
        :param lat:
        :return:
        """
        return self.min_lon <= lon < self.max_lon and self.min_lat <= lat < self.max_lat

    def makeVocab(self):
        """
        Build the vocabulary from the raw trajectories stored in the hdf5 file.
        For one trajectory, each point lies in a row if reading with Python.
        :return:
        """

        vocab_file = f"/home/jiali/clear/data/{self.dataset_name}/{self.dataset_name}_cell-{self.x_step}_minfreq-{self.minfreq}_grid_vocab.pkl"

        if not os.path.exists(vocab_file):
            start = time.time()
            num_out_region = 0  # useful for evaluating the size of region bounding box
            with open(self.traj_file, "rb") as f:
                df = pickle.load(f)

            num = df.shape[0]

            # cell_generation
            cell_list = []
            print("Cell generation begins")
            for i in range(0, num):
                traj = df.loc[i, "Locations"]
                for j in range(0, len(traj)):
                    lon, lat = traj[j, :]
                    if not self.point_inregion(lon, lat):
                        num_out_region += 1
                    else:
                        cell = self.gps2cell(lon, lat)
                        cell_list.append(cell)
                if i % 100000 == 0:
                    print(f"Processed {i} trips")
            print(f"Cell generation ends, we got {len(cell_list)} cells from the trajectories")

            self.num_out_region = num_out_region

            # Find out all hot cells
            self.cellcounter = Counter(cell_list)
            self.hotcell = []
            for item in self.cellcounter.most_common():
                if item[1] >= self.minfreq:
                    self.hotcell.append(item[0])
            # Further restrain the number of hot cells by self.maxvocab_size
            max_num_hotcells = min(self.maxvocab_size, len(self.hotcell))
            print(
                f"We have {len(self.hotcell)} hot cells, maxvocab_size is {self.maxvocab_size}, max_num_hotcells is {max_num_hotcells}")
            self.hotcell = self.hotcell[0:max_num_hotcells]

            print(f"Attention! We have {len(self.cellcounter)} cells, only {len(self.hotcell)} of them are hot cells")

            print("Build the map between cell and vocab id")
            self.hotcell2vocab = dict([(v, k) for k, v in enumerate(self.hotcell)])
            self.vocab2hotcell = dict([(v, k) for k, v in self.hotcell2vocab.items()])

            print("Calculate vocabulary size")
            self.vocab_size = len(self.hotcell)
            print(f"vocab_size is {self.vocab_size}")

            print("Build the hot cell balltree to facilitate search")
            coord = []
            for item in self.hotcell:
                # coord.append(self.cell2coord(item))
                coord.append(self.cell2gps(item))
            self.hotcell_tree = BallTree(coord, metric='haversine')
            self.built = True
            print("Save extra information to region")
            self.saveregion(vocab_file)
            print(f"Time cost: {time.time() - start} s")
        else:
            self.built = True
            print(f"Loading Vocabulary from {vocab_file}")
            num_out_region = self.loadregion(vocab_file)

        return len(self.cellcounter.keys()), len(self.hotcell)

    def bin_p(self):
        df = pd.DataFrame()
        bins = [0, 1000, 2000, 3000, 4000, 5000, 10000, 1000000]
        len_list = []
        for cell in self.hotcell:
            len_list.append(self.cellcounter[cell])
        df["Length"] = len_list
        print(df["Length"].describe())
        df_length = df["Length"]
        cats = pd.cut(df_length, bins)
        print(pd.value_counts(cats, sort=False))

    def density_p(self):
        """
        Since we put all points into hot cells, then then density should be increased.
        :return:
        """
        sum = 0
        for k, v in self.cellcounter.items():
            sum += v
        sum_hot = 0
        for k in self.hotcell:
            sum_hot += self.cellcounter[k]

        # return the density_p, density_p only for hotcells, density_p for mapped hotcells

        return sum / len(self.cellcounter.keys()), sum_hot / len(self.hotcell), sum / len(self.hotcell)

    def density_t(self):
        """
        Since we put all points into hot cells, then then density should be increased.
        :return:
        """
        p_cell = []
        t_cell = []
        p_hotcell = []
        t_hotcell = []
        with open(self.traj_file, "rb") as f:
            df = pickle.load(f)

        num = df.shape[0]
        for i in range(0, num):
            p_cell_temp = []
            p_hotcell_temp = []

            traj = df.loc[i, "Locations"]
            for j in range(0, len(traj)):
                lon, lat = traj[j, :]
                if self.point_inregion(lon, lat):
                    cell_id = self.gps2cell(lon, lat)
                    hotcell_id = self.cell2vocab(cell_id)
                    p_cell_temp.append(cell_id)
                    p_hotcell_temp.append(hotcell_id)
            p_cell += [item for item in p_cell_temp]
            t_cell += [item for item in list(set(p_cell_temp))]  # For traj, it can have multiple points pass through one region
            p_hotcell += [item for item in p_hotcell_temp]
            t_hotcell += [item for item in list(set(p_hotcell_temp))]
        # Without mapping to hotcell
        p_counter = Counter(p_cell)
        t_counter = Counter(t_cell)

        print(len(self.cellcounter.keys()), len(t_counter.keys()), len(p_counter.keys()))
        for k in p_counter.keys():
            if p_counter[k] < t_counter[k]:
                print(k, p_counter[k], t_counter[k])

        # density_t
        sum_t = 0
        sum_t_hotcell = 0
        for k, v in t_counter.items():
            sum_t += v
        # density_t for hot cell
        for k in self.hotcell:
            sum_t_hotcell += t_counter[k]
        # density_p/t
        density_counter = {}
        for k in p_counter.keys():
            density_counter[k] = p_counter[k] / t_counter[k]
        sum_density_cell = 0
        sum_density_hotcell = 0
        for k, v in density_counter.items():
            sum_density_cell += v
            if k in self.hotcell:
                sum_density_hotcell += v

        # With mapping to hotcell
        p_hc_counter = Counter(p_hotcell)
        t_hc_counter = Counter(t_hotcell)
        # density_t
        sum_t_hc = 0
        for k, v in t_hc_counter.items():
            sum_t_hc += v
        # density_p/t
        sum_density_hc = 0
        for k in p_hc_counter.keys():
            sum_density_hc += p_hc_counter[k] / t_hc_counter[k]

        result1 = sum_t / len(self.cellcounter.keys())
        result2 = sum_t_hotcell / len(self.hotcell)
        result3 = sum_t_hc / len(self.hotcell)

        result4 = sum_density_cell / len(self.cellcounter.keys())
        result5 = sum_density_hotcell / len(self.hotcell)
        result6 = sum_density_hc / len(self.hotcell)

        return result1, result2, result3, result4, result5, result6

    def knearestHotcells(self, cell, k):
        assert self.built, "Build index for region first"
        coord = self.cell2gps(cell)
        dists, idxs = self.hotcell_tree.query(np.array(coord).reshape(1, -1), k=k)
        dists = dists[0]
        idxs = idxs[0]
        result_cell = []
        for index in list(idxs):
            result_cell.append(self.hotcell[index])
        return result_cell, dists

    def nearestHotcell(self, cell):
        assert self.built, "Build index for region first"
        hotcell, _ = self.knearestHotcells(cell, 1)

        return hotcell[0]

    def cell2vocab(self, cell):
        """
        Return the vocab id for a cell in the region.
        If the cell is not hot cell, the function will first search its nearest hotcell and return the corresponding vocab id
        :param region:
        :param cell:
        :return:
        """
        assert self.built, "Build index for region first"
        if cell in self.hotcell2vocab.keys():
            return self.hotcell2vocab[cell]
        else:
            hotcell = self.nearestHotcell(cell)
            return self.hotcell2vocab[hotcell]

    def gps2vocab(self, lon, lat):
        """
        Mapping a gps point to the vocab id in the vocabulary consists of hot cells,
        each hot cell has an unique vocab id (hotcell2vocab)
        If the point falls out of the region, 'UNK' will be returned.
        If the point falls into the region, but out of the hot cells, its nearest hot cell will be used.
        :param region:
        :param lon:
        :param lat:
        :return:
        """
        if self.point_inregion(lon, lat):
            return self.cell2vocab(self.gps2cell(lon, lat))
        else:
            return self.cell2vocab(self.gps2cell(max(min(lon, self.max_lon), self.min_lon), max(min(lat, self.max_lat), self.min_lat)))

    def traj2seq(self, traj):
        seq = []
        for i in range(0, traj.shape[0]):
            lon, lat = traj[i, :]
            seq.append(self.gps2vocab(lon, lat))

        return seq

    def seq2str(self, seq):
        """
        :param seq:
        :return:
        """
        result = ''
        for item in seq:
            result += (str(item) + ' ')
        result += '\n'
        return result

    def spatial_encoding(self, traj):

        return np.array(self.traj2seq(traj))

    def saveregion(self, param_file):
        """
        :param param_file:
        :return:
        """
        with open(param_file, 'wb') as f:
            pickle.dump({"num_out_region": self.num_out_region,
                         "cellcounter": self.cellcounter,
                         "hotcell": self.hotcell,
                         "hotcell2vocab": self.hotcell2vocab,
                         "vocab2hotcell": self.vocab2hotcell,
                         "hotcell_tree": self.hotcell_tree,
                         "vocab_size": self.vocab_size}, f)
        f.close()

    def loadregion(self, param_file):
        with open(param_file, 'rb') as f:
            region_temp = pickle.load(f)
            self.num_out_region = region_temp["num_out_region"]
            self.cellcounter = region_temp["cellcounter"]
            self.hotcell = region_temp["hotcell"]
            self.hotcell2vocab = region_temp["hotcell2vocab"]
            self.vocab2hotcell = region_temp["vocab2hotcell"]
            self.hotcell_tree = region_temp["hotcell_tree"]
            self.vocab_size = region_temp["vocab_size"]
            self.built = True
        f.close()
        return self.num_out_region


def grid_cell_stats(dataset_name, cell_size, minfreq, regionps):
    cell_size = int(cell_size)
    minfreq = int(minfreq)
    region = GridPartition(traj_file=f"../data/{dataset_name}/{dataset_name}_full.pkl",
                           dataset_name=dataset_name,
                           min_lon=regionps["min_lon"],
                           min_lat=regionps["min_lat"],
                           max_lon=regionps["max_lon"],
                           max_lat=regionps["max_lat"],
                           x_step=cell_size,
                           y_step=cell_size,
                           minfreq=minfreq,
                           maxvocab_size=1000000,
                           built=False)

    result = {}
    result[f"cell-{cell_size}_minfreq-{minfreq}"] = []

    total_num, hot_num = region.makeVocab()

    density_p, density_p_hotcell, density_p_hc = region.density_p()

    density_t, density_t_hotcell, density_t_hc, d, d_hotcell, d_hc = region.density_t()
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(total_num)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(hot_num)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(density_p)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(density_p_hotcell)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(density_p_hc)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(density_t)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(density_t_hotcell)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(density_t_hc)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(d)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(d_hotcell)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(d_hc)
    indexes = []
    indexes.append(f"{dataset_name}_num_cell")
    indexes.append(f"{dataset_name}_num_hotcell")
    indexes.append(f"{dataset_name}_density_p")
    indexes.append(f"{dataset_name}_density_p_hotcell")
    indexes.append(f"{dataset_name}_density_p_hc")
    indexes.append(f"{dataset_name}_density_t")
    indexes.append(f"{dataset_name}_density_t_hotcell")
    indexes.append(f"{dataset_name}_density_t_hc")
    indexes.append(f"{dataset_name}_d")
    indexes.append(f"{dataset_name}_d_hotcell")
    indexes.append(f"{dataset_name}_d_hc")
    df = pd.DataFrame(result, index=indexes)
    df.to_csv(f"cell-{cell_size}_minfreq-{minfreq}_density_{dataset_name}.csv")


def region_grid(dataset_name, cell_size, minfreq, regionps):
    cell_size = int(cell_size)
    minfreq = int(minfreq)

    region = GridPartition(traj_file=f"../data/{dataset_name}/{dataset_name}_full.pkl",
                           dataset_name=dataset_name,
                           min_lon=regionps["min_lon"],
                           min_lat=regionps["min_lat"],
                           max_lon=regionps["max_lon"],
                           max_lat=regionps["max_lat"],
                           x_step=cell_size,
                           y_step=cell_size,
                           minfreq=minfreq,
                           maxvocab_size=100000,
                           built=False)
    # creating vocabulary of trajectory dataset
    region.makeVocab()
    print(f"Vocabulary size {region.vocab_size} with cellsize {cell_size} (meters)")

    return region


def token_grid_single(result_path, dataset_name, region):
    marker = result_path.split('/')[-1].split('_token.pkl')[-2]
    if os.path.exists(result_path):
        print(f"Spatial token for {marker} already exists")
        return
    else:
        print(f"Creating spatial token for {marker}")
        if "original" in marker:
            file_name = f"../data/{dataset_name}/{dataset_name}.pkl"
        else:
            file_name = f"../data/{dataset_name}/augmentation/{marker}.pkl"

        with open(file_name, "rb") as f:
            df = pickle.load(f)
        token_list = []
        num = df.shape[0]
        start = time.time()
        for i in range(num):
            traj = df.loc[i, "Locations"]
            token_list.append(region.spatial_encoding(traj))
            if i % 100000 == 0 and i != 0:
                print(f"Processed {i} trajectories in {marker}, avg time is {(time.time() - start) / i},"
                      f" job is expected to be finished in {((time.time() - start) / i) * (num - i)} s")
        df["token"] = token_list
        df_token = df[["token"]]
        with open(result_path, "wb") as f:
            pickle.dump(df_token, f)
        print(f"Creating spatial token based on grid partitioning for {marker} cost: {time.time() - start} s")


def token_grid(dataset_name, encoded_folder, cell_size, minfreq, region):
    cell_size = int(cell_size)
    minfreq = int(minfreq)
    encoded_folder = f"{encoded_folder}/cellsize-{cell_size}_minfreq-{minfreq}"
    if not os.path.exists(encoded_folder):
        os.makedirs(encoded_folder)
    result_path_list = [os.path.join(encoded_folder, f"{file.split('.pkl')[0]}_token.pkl") for file in os.listdir(f"../data/{dataset_name}/augmentation")]
    result_path_list.insert(0, os.path.join(encoded_folder, f"{dataset_name}_original_token.pkl"))

    params = []
    for result_path in result_path_list:
        params.append((result_path, dataset_name, region))
    pool = multiprocessing.Pool(processes=8)
    pool.starmap(token_grid_single, params)
    pool.close()
    pool.join()


def cell_connection_build(dataset_name, cell_size, minfreq, region, k=20):
    result_file = f"../data/{dataset_name}/{dataset_name}_edges_cellsize-{cell_size}_minfreq-{minfreq}.pkl"
    cell2idx = {v: k for k, v in enumerate(region.hotcell)}
    if not os.path.exists(result_file):
        edges = []
        for cell_id in region.hotcell:
            neighbors_id, _ = region.knearestHotcells(cell_id, k)
            for neighbor_id in neighbors_id:
                edges.append((cell2idx[cell_id], cell2idx[neighbor_id]))
        print(f"We have {len(edges)} edges")
        with open(result_file, "wb") as f:
            pickle.dump(edges, f)
    else:
        print(f"Load edges from {result_file}")
        with open(result_file, "rb") as f:
            edges = pickle.load(f)

    return edges


if __name__ == "__main__":
    configs = Config()
    configs.default_update(parse_args())
    configs.config_dataset()
    print(f"{configs.dataset_name}")
    dataset_name = configs.dataset_name
    cell_size = configs.cell_size
    minfreq = configs.minfreq
    region_settings = {"min_lon": configs.min_lon,
                       "max_lon": configs.max_lon,
                       "min_lat": configs.min_lat,
                       "max_lat": configs.max_lat}

    # Grid partitioning
    t1 = time.time()
    region = region_grid(dataset_name, cell_size, minfreq, region_settings)
    t2 = time.time()
    print(f"Grid paritioning costs {t2 - t1} s")
    # Token generation
    token_folder = f"../data/{dataset_name}/token"
    if not os.path.exists(token_folder):
        os.mkdir(token_folder)
    token_grid(dataset_name, token_folder, cell_size, minfreq, region)
    print(f"Token generation costs: {time.time() - t2} s")

    # Connection build
    t3 = time.time()
    edges = cell_connection_build(dataset_name, cell_size, minfreq, region, k=8)
    print(f"Connection build costs: {time.time() - t3} s")

    # Node2vec
    print("Node2vec starts")
    t4 = time.time()
    edges = torch.tensor(edges, dtype=torch.long).T
    cell_embedding_dim = configs.hidden_size
    dataset_prefix = f"{dataset_name}_cellsize-{cell_size}_minfreq-{minfreq}"
    embs_file = f"../data/{dataset_name}/{dataset_name}_size-{cell_embedding_dim}_cellsize-{cell_size}_minfreq-{minfreq}_node2vec.pkl"
    checkpoint_file = f"../data/{dataset_name}/pretrain/{dataset_prefix}_node2vec_cell_best.pt"

    #train_node2vec(edges, cell_embedding_dim, checkpoint_file, embs_file)
    print(f"Node2vec costs {time.time() - t4} s")






