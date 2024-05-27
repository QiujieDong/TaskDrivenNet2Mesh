import shutil
import os
import sys
import random
import numpy as np
import math

import torch
from torch.utils.data import Dataset

import trimesh
from utils.prepare import compute_all_operators
import utils.util as utils


class Manifold40ClsDataset(Dataset):

    def __init__(self, args, is_train=True):

        self.args = args
        self.is_train = is_train

        self.data_path = os.path.join('data', self.args.dataset_name)
        self.cache_dir = os.path.join(self.data_path, "cache")
        self.dataset_cache_dir = os.path.join(self.data_path, "dataset_cache")

        self.class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                            'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar',
                            'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano',
                            'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent',
                            'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.labels_list = []
        self.mesh_path_list = []

        load_cache = None
        if self.args.use_cache:
            print('---use cache: loading {} ---'.format('train_cache' if self.is_train else 'test_cache'))
            train_set_cache = os.path.join(self.cache_dir, 'train_cache.pt')
            test_set_cache = os.path.join(self.cache_dir, 'test_cache.pt')
            load_cache = train_set_cache if self.is_train else test_set_cache
            if os.path.exists(load_cache):
                self.verts_list, self.faces_list, self.verts_normals_list, self.evals_list, self.evecs_list, self.verts_dihedralAngles_list, self.hks_list, self.mass_list, self.labels_list, self.mesh_path_list = torch.load(
                    load_cache)
                return
            print("---do not find data in cache, recomputing---")

        subdataset_name = 'train' if self.is_train else 'test'
        for class_idx, class_name in enumerate(self.class_names):
            files = os.listdir(os.path.join(self.data_path, class_name, subdataset_name))
            select_files = self.samples_files_list = random.sample(files,
                                                                   math.ceil(len(files) / self.args.select_num))
            for f in files:
                full_f = os.path.join(self.data_path, class_name, subdataset_name, f)
                self.mesh_path_list.append(full_f)
                self.labels_list.append(class_idx)

        for ind, label in enumerate(self.labels_list):
            self.labels_list[ind] = torch.tensor(label)

        # Precompute operators
        self.verts_list, self.faces_list, self.verts_normals_list, self.evals_list, self.evecs_list, \
            self.verts_dihedralAngles_list, self.hks_list, self.mass_list = compute_all_operators(
            self.args, self.mesh_path_list, self.dataset_cache_dir)

        # save to cache
        if self.args.use_cache:
            utils.ensure_folder_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.verts_normals_list, self.evals_list, self.evecs_list,
                        self.verts_dihedralAngles_list, self.hks_list, self.mass_list, self.labels_list,
                        self.mesh_path_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.verts_normals_list[idx], self.evals_list[idx], \
            self.evecs_list[idx], self.verts_dihedralAngles_list[idx], self.hks_list[idx], \
            self.mass_list[idx], self.labels_list[idx], self.mesh_path_list[idx]