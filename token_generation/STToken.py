import sys
sys.path.append("..")
import os
import pickle
import time
import pandas as pd
import h5py
import math
import numpy as np
import json
from utils import meters2lonlat, lonlat2meters
from collections import Counter
from sklearn.neighbors import BallTree
from constants import *
import argparse
import multiprocessing

parser = argparse.ArgumentParser(description="SpatialEmbedding.py")

parser.add_argument("-dataset_name")


def rle(seq):
    import itertools
    return [k for k, v in itertools.groupby(seq)]


class GridPartition:
    def __init__(self, traj_file, dataset_name, min_lon, min_lat, max_lon, max_lat, x_step, y_step, minfreq,
                 maxvocab_size, vocab_start, built):
        """
        :param traj_file: path for the trajectory file, hdf5 file.
        :param dataset_name: dataset name
        :param min_lon: Boundary, GPS coordinate
        :param min_lat: Boundary, GPS coordinate
        :param max_lon: Boundary, GPS coordinate
        :param max_lat: Boundary, GPS coordinate
        :param x_step: Cell size x (meters)
        :param y_step: Cell size y (meters)
        :param minfreq: hot cell threshold, a hot cell must at least has minfreq points
        :param maxvocab_size: threshold of vocabulary
        :param vocab_start: The first several numbers are reserved for other purposes such as Padding, See constants.py for details.
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
        self.vocab_start = vocab_start
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
            f = h5py.File(self.traj_file)
            num = f.attrs["num"]

            # cell_generation
            cell_list = []
            print("Cell generation begins")
            for i in range(0, num):
                traj = f[f'Locations/{i}'][:]
                for j in range(0, len(traj)):
                    lon, lat = traj[j, :]
                    if not self.point_inregion(lon, lat):
                        num_out_region += 1
                    else:
                        cell = self.gps2cell(lon, lat)
                        cell_list.append(cell)
                if i % 100000 == 0:
                    print(f"Processing {i} trips")
            print(f"Cell generation ends, {len(cell_list)} cells")

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
            self.hotcell2vocab = dict([(v, k + self.vocab_start) for k, v in enumerate(self.hotcell)])
            self.vocab2hotcell = dict([(v, k) for k, v in self.hotcell2vocab.items()])

            print("Calculate vocabulary size")
            self.vocab_size = self.vocab_start + len(self.hotcell)
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

    def hotcell_density(self):
        scales = [self.cellcounter[cellID] for cellID in self.hotcell]

        return np.array(scales).mean()

    def abandoned_points(self):
        num = 0
        for cellID in self.cellcounter.keys():
            if not cellID in self.hotcell:
                num += self.cellcounter[cellID]

        return num

    def trajectory_density(self):
        hotcells_density = {}
        point_token = []
        traj_token = []
        with h5py.File(self.traj_file, "r") as f:
            num = f.attrs["num"]
            for i in range(0, num):
                point_temp = []
                traj = f[f'Locations/{i}'][:]
                for j in range(0, len(traj)):
                    lon, lat = traj[j, :]
                    if self.point_inregion(lon, lat):
                        cell_id = self.gps2cell(lon, lat)
                        if cell_id in self.hotcell2vocab.keys():
                            point_temp.append(cell_id)
                point_token += [item for item in point_temp]
                traj_token += [item for item in list(set(point_temp))]
        point_counter = Counter(point_token)
        traj_counter = Counter(traj_token)
        values = []
        for k in point_counter.keys():
            hotcells_density[k] = point_counter[k] / traj_counter[k]
            values.append(point_counter[k] / traj_counter[k])
        # print(f"We have {len(point_counter.keys())} hot cells, the average density is {np.array(values).mean()}")
        return np.array(values).mean()

    def knearestHotcells(self, cell, k):
        assert self.built, "Build index for region first"
        #coord = self.cell2coord(cell)
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
            return UNK

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


def grid_cell_check(dataset_name, cell_size, minfreq):
    cell_size = int(cell_size)
    minfreq = int(minfreq)
    print(f"{dataset_name}: cell size {cell_size}, minfreq {minfreq}")
    result = {}
    result[f"cell-{cell_size}_minfreq-{minfreq}"] = []
    ori_datafile = f"../data/{dataset_name}/{dataset_name}_full.h5"
    if not os.path.exists(ori_datafile):
        print(f"Please provide the correct hdf5 file: {ori_datafile}")
    # load fundamental hyper-parameters from json_file
    with open(os.path.join(
            f"../data/{dataset_name}/{dataset_name}_hyper-parameters.json")) as f:
        param = json.load(f)
        regionps = param["region"]
    f.close()
    region = GridPartition(traj_file=ori_datafile,
                           dataset_name=regionps["cityname"],
                           min_lon=regionps["minlon"],
                           min_lat=regionps["minlat"],
                           max_lon=regionps["maxlon"],
                           max_lat=regionps["maxlat"],
                           x_step=cell_size,
                           y_step=cell_size,
                           minfreq=minfreq,
                           maxvocab_size=100000000000,
                           vocab_start=4,
                           built=False)
    total_num, hot_num = region.makeVocab()
    point_density = region.hotcell_density()
    traj_density = region.trajectory_density()
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(total_num)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(hot_num)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(point_density)
    result[f"cell-{cell_size}_minfreq-{minfreq}"].append(traj_density)
    indexes = []
    indexes.append(f"{dataset_name}_total")
    indexes.append(f"{dataset_name}_hot")
    indexes.append(f"{dataset_name}_point-density")
    indexes.append(f"{dataset_name}_traj-density")
    df = pd.DataFrame(result, index=indexes)
    df.to_csv(f"cell-{cell_size}_minfreq-{minfreq}_density_{dataset_name}.csv")


def region_grid(dataset_name, cell_size=100, minfreq=50):
    cell_size = int(cell_size)
    minfreq = int(minfreq)
    # check the ori data file
    ori_datafile = f"/home/jiali/clear/data/{dataset_name}/{dataset_name}_full.h5"
    if not os.path.exists(ori_datafile):
        print(f"Please provide the correct hdf5 file: {ori_datafile}")

    # load fundamental hyper-parameters from json_file
    with open(os.path.join(f"/home/jiali/clear/data/{dataset_name}/{dataset_name}_hyper-parameters.json")) as f:
        param = json.load(f)
        regionps = param["region"]
    f.close()
    region = GridPartition(traj_file=ori_datafile,
                           dataset_name=regionps["cityname"],
                           min_lon=regionps["minlon"],
                           min_lat=regionps["minlat"],
                           max_lon=regionps["maxlon"],
                           max_lat=regionps["maxlat"],
                           x_step=cell_size,
                           y_step=cell_size,
                           minfreq=minfreq,
                           maxvocab_size=100000,
                           vocab_start=4,
                           built=False)
    # creating vocabulary of trajectory dataset
    region.makeVocab()
    print(f"Vocabulary size {region.vocab_size} with cellsize {regionps['cellsize']} (meters)")

    return region


def token_grid_single(result_path, dataset_name, region):
    marker = result_path.split('/')[-1].split('_seq.h5')[-2]
    print(f"Creating spatial token for {marker}")
    if os.path.exists(result_path):
        f_temp = h5py.File(result_path, 'r')
        flag = 'Locations' in f_temp.keys()
        f_temp.close()
        if flag:
            f_new = None
        else:
            f_new = h5py.File(result_path, 'a')
    else:
        f_new = h5py.File(result_path, 'w')

    if f_new is None:
        print(f"Spatial token for {marker} already exists")
    else:
        if "original" in marker:
            file_name = f"../data/{dataset_name}/{dataset_name}.h5"
        else:
            file_name = f"../data/{dataset_name}/augmentation/{marker}.h5"
        with h5py.File(file_name) as f:
            num = f.attrs["num"]
            start = time.perf_counter()
            for i in range(num):
                locations = np.array(f[f"/Locations/{i}"])
                v_s = region.spatial_encoding(locations)
                f_new[f"/Locations/{i}"] = v_s
                if i % 100000 == 0 and i != 0:
                    print(f"Processed {i} trajectories, avg time is {(time.perf_counter() - start) / i},"
                          f" job is expected to be finished in {((time.perf_counter() - start) / i) * (num - i)} s")
            f_new.attrs["num"] = num
        f_new.close()
        print(f"Creating spatial token based on grid partitioning for {marker} cost: {time.perf_counter() - start} s")


def token_grid(dataset_name, encoded_folder, cell_size=100, minfreq=50):
    cell_size = int(cell_size)
    minfreq = int(minfreq)
    region = region_grid(dataset_name, cell_size, minfreq)
    if not os.path.exists(os.path.join(encoded_folder, f"cell-{cell_size}_minfreq-{minfreq}")):
        os.mkdir(os.path.join(encoded_folder, f"cell-{cell_size}_minfreq-{minfreq}"))
    encoded_folder = f"{encoded_folder}/cell-{cell_size}_minfreq-{minfreq}"
    result_path_list = [os.path.join(encoded_folder, f"{file.split('.h5')[0]}_seq.h5") for file in os.listdir(f"../data/{dataset_name}/augmentation")]
    result_path_list.insert(0, os.path.join(encoded_folder, f"{dataset_name}_original_seq.h5"))

    params = []
    for result_path in result_path_list:
        params.append((result_path, dataset_name, region))
    pool = multiprocessing.Pool(processes=8)
    pool.starmap(token_grid_single, params)
    pool.close()
    pool.join()






