import shutil
import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import trimesh
from utils.prepare import compute_all_operators, get_all_operators
import utils.util as utils


class Shrec11MeshDataset_Simplify(Dataset):

    # NOTE: Remeshed data from MeshCNN authors.
    # Can be downloaded here (link from the MeshCNN authors): https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz. Note that despite the filename, this really is the shapes from the SHREC 2011 dataset. Extract it to the `[ROOT_DIR]/raw/` directory.

    def __init__(self, args, is_train=True, exclude_dict=None):
        self.args = args
        self.is_train = is_train  # bool
        self.exclude_dict = exclude_dict

        self.split_size = None
        if self.is_train:
            self.split_size = self.args.split_size

        self.data_path = os.path.join('data', self.args.dataset_name)
        self.dataset_cache_dir = os.path.join(self.data_path, "dataset_cache")

        self.class_names = ['alien', 'ants', 'armadillo', 'bird1', 'bird2', 'camel', 'cat', 'centaur', 'dinosaur',
                            'dino_ske', 'dog1', 'dog2', 'flamingo', 'glasses', 'gorilla', 'hand', 'horse', 'lamp',
                            'laptop', 'man', 'myScissor', 'octopus', 'pliers', 'rabbit', 'santa', 'shark', 'snake',
                            'spiders', 'two_balls', 'woman']

        self.entries = {}

        # store in memory
        self.labels_list = []
        self.mesh_path_list = []

        for class_idx, class_name in enumerate(self.class_names):

            # load both train and test subdirectories; we are manually regenerating random splits to do multiple trials
            mesh_files = []
            for t in ['test', 'train']:
                files = os.listdir(os.path.join(self.data_path, class_name, t))
                for f in files:
                    full_f = os.path.join('data', self.args.dataset_name, class_name, t, f)
                    mesh_files.append(full_f)

            # print("123456", mesh_files)
            # Randomly grab samples for this split. If given, disallow any samples in commmon with exclude_dict (ie making sure train set is distinct from test).
            order = np.random.permutation(len(mesh_files))
            added = 0
            self.entries[class_name] = set()
            for ind in order:
                if (self.split_size is not None and added == self.split_size): continue

                path = mesh_files[ind]
                if self.exclude_dict is not None and path in exclude_dict[class_name]:
                    continue

                self.labels_list.append(class_idx)
                self.entries[class_name].add(path)
                self.mesh_path_list.append(path)

                added += 1

            if (self.split_size is not None and added < self.split_size):
                raise ValueError("could not find enough entries to generate requested split")

        for ind, label in enumerate(self.labels_list):
            self.labels_list[ind] = torch.tensor(label)
        # Precompute operators
        self.verts_list, self.faces_list, self.verts_normals_list, self.evals_list, self.evecs_list, \
        self.verts_dihedralAngles_list, self.hks_list, self.mass_list = compute_all_operators(
            self.args, self.mesh_path_list, self.dataset_cache_dir)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.verts_normals_list[idx], self.evals_list[idx], \
               self.evecs_list[idx], self.verts_dihedralAngles_list[idx], self.hks_list[idx], \
               self.mass_list[idx], self.labels_list[idx], self.mesh_path_list[idx]


class Shrec11MeshDataset_Original(Dataset):

    # NOTE: Original data from the challenge, not simplified models

    # The original SHREC11 models were previously distributed via NIST [here](https://www.nist.gov/itl/iad/shrec-2011-datasets), but that page seems to have been lost to the sands of time. We provide a zip of the old dataset page here: https://drive.google.com/uc?export=download&id=1O_P03aAxhjCOKQH2n71j013-EfSmEp5e. The relevant files are in `SHREC11_test_database_new.zip`, which is password protected with the password `SHREC11@NIST`.

    # Unzip it like
    # unzip -P SHREC11@NIST SHREC11_test_database_new.zip -d [DATA_ROOT]/raw

    def __init__(self, args, is_train=True, exclude_dict=None):
        self.args = args
        self.is_train = is_train  # bool
        self.exclude_dict = exclude_dict

        self.split_size = None
        if self.is_train:
            self.split_size = self.args.split_size

        self.data_path = os.path.join('data', self.args.dataset_name)
        self.dataset_cache_dir = os.path.join(self.data_path, "dataset_cache")

        self.class_names = []
        self.entries = {}

        # store in memory
        self.labels_list = []
        self.mesh_path_list = []

        ## Parse the categories file
        cat_path = os.path.join(self.data_path, 'categories.txt')
        with open(cat_path) as cat_file:
            cat_file.readline()  # skip the first two lines
            cat_file.readline()
            for i_class in range(self.args.num_classes):
                cat_file.readline()
                class_name, _, count = cat_file.readline().strip().split()
                count = int(count)
                self.class_names.append(class_name)
                mesh_files = []
                for j in range(count):
                    mesh_files.append(cat_file.readline().strip())

                # Randomly grab samples for this split. If given, disallow any samples in commmon with exclude_dict (ie making sure train set is distinct from test).
                order = np.random.permutation(len(mesh_files))
                added = 0
                self.entries[class_name] = set()

                for ind in order:
                    if (self.split_size is not None and added == self.split_size): continue

                    name = mesh_files[ind]
                    if self.exclude_dict is not None and name in exclude_dict[class_name]:
                        continue
                    path = os.path.join(self.data_path, "T{}.off".format(name))

                    self.labels_list.append(i_class)
                    self.entries[class_name].add(name)
                    self.mesh_path_list.append(path)

                    added += 1

                if (self.split_size is not None and added < self.split_size):
                    raise ValueError("could not find enough entries to generate requested split")

        for ind, label in enumerate(self.labels_list):
            self.labels_list[ind] = torch.tensor(label)

        # Precompute operators
        self.verts_list, self.faces_list, self.verts_normals_list, self.evals_list, self.evecs_list, \
        self.verts_dihedralAngles_list, self.hks_list, self.mass_list = compute_all_operators(
            self.args, self.mesh_path_list, self.dataset_cache_dir)

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        mesh_path = self.mesh_path_list[idx]

        verts, faces, verts_normals, evals, evecs, \
        verts_dihedralAngles, hks, mass = get_all_operators(
            self.args, mesh_path, self.dataset_cache_dir)

        return verts, faces, verts_normals, evals, \
               evecs, verts_dihedralAngles, hks, \
               mass, self.labels_list[idx], mesh_path

