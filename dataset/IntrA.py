import shutil
import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import trimesh
from utils.prepare import compute_all_operators
import utils.util as utils


class AneurysmSegDataset(Dataset):

    def __init__(self, args, is_train=True):
        self.args = args
        self.is_train = is_train

        self.data_path = os.path.join('data', self.args.dataset_name, self.args.dataset_name, 'annotated')
        self.cache_dir = os.path.join(self.data_path, "cache")
        self.dataset_cache_dir = os.path.join(self.data_path, "dataset_cache")

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

        # store in memory
        self.labels_list = []  # per-faces labels
        self.mesh_path_list = []

        # Load the meshes & labels
        if not os.path.exists(os.path.join(self.data_path, "train.txt")):
            utils.split_dataset_and_generate_labels(self.data_path)

        if self.is_train:
            with open(os.path.join(self.data_path, "train.txt")) as f:
                this_files = [line.rstrip() for line in f]
        else:
            with open(os.path.join(self.data_path, "test.txt")) as f:
                this_files = [line.rstrip() for line in f]

        subset_data_path = os.path.join(self.data_path, 'obj')
        labels_path = os.path.join(self.data_path, "seg")
        for f in this_files:
            mesh_path = os.path.join(subset_data_path, f)
            label_file = os.path.join(labels_path, f[:-9] + ".eseg")

            labels = np.loadtxt(label_file).astype(int)
            labels[labels[:] == 2] = 1  # the boundary lines are grouped into aneurysm segments
            labels = torch.tensor(np.ascontiguousarray(labels))

            self.labels_list.append(labels)
            self.mesh_path_list.append(mesh_path)

        # Precompute operators
        self.verts_list, self.faces_list, self.verts_normals_list, self.evals_list, self.evecs_list, self.verts_dihedralAngles_list, self.hks_list, self.mass_list = compute_all_operators(
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
        # print("IntrA.py_75_verts_list", self.verts_list[idx].size())
        return self.verts_list[idx], self.faces_list[idx], self.verts_normals_list[idx], self.evals_list[idx], \
               self.evecs_list[idx], self.verts_dihedralAngles_list[idx], self.hks_list[idx], \
               self.mass_list[idx], self.labels_list[idx], self.mesh_path_list[idx]
