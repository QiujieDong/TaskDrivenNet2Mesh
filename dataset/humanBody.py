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


class segHumanBody_simplify(Dataset):
    """
    """

    def __init__(self, args, is_train=True):

        self.args = args
        self.is_train = is_train  # bool

        self.data_path = os.path.join('data', self.args.dataset_name)
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

        # Load the actual files
        subset_data_path = os.path.join(self.data_path, 'train' if self.is_train else 'test')
        labels_path = os.path.join(self.data_path, 'seg')  # per_faces label path

        for f in os.listdir(subset_data_path):
            mesh_path = os.path.join(subset_data_path, f)
            label_file = os.path.join(labels_path, f[:-4] + ".eseg")

            labels = np.loadtxt(label_file).astype(int)
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
        return self.verts_list[idx], self.faces_list[idx], self.verts_normals_list[idx], self.evals_list[idx], \
               self.evecs_list[idx], self.verts_dihedralAngles_list[idx], self.hks_list[idx], \
               self.mass_list[idx], self.labels_list[idx], self.mesh_path_list[idx]



class segHumanBody_origin(Dataset):
    """
    """

    def __init__(self, args, is_train=True):
        super(segHumanBody_origin, self).__init__()

        self.args = args
        self.is_train = is_train

        self.data_path = os.path.join('data', self.args.dataset_name)
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


        # Get all the files
        label_files = []
        mesh_path = []

        # Train test split
        if self.is_train:

            # adobe
            mesh_dirpath = os.path.join(self.data_path, "meshes", "train", "adobe")
            label_dirpath = os.path.join(self.data_path, "segs", "train", "adobe")
            for fname in os.listdir(mesh_dirpath):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                label_fullpath = os.path.join(label_dirpath, fname[:-4] + ".txt")
                mesh_path.append(mesh_fullpath)
                label_files.append(label_fullpath)

            # faust
            mesh_dirpath = os.path.join(self.data_path, "meshes", "train", "faust")
            label_dirpath = os.path.join(self.data_path, "segs", "train", "faust")
            for fname in os.listdir(mesh_dirpath):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                label_fullpath = os.path.join(label_dirpath, "faust_corrected.txt")
                mesh_path.append(mesh_fullpath)
                label_files.append(label_fullpath)

            # mit
            mesh_dirpath_patt = os.path.join(self.data_path, "meshes", "train", "MIT_animation", "meshes_{}", "meshes")
            label_dirpath = os.path.join(self.data_path, "segs", "train", "mit")
            pose_names = ['bouncing', 'handstand', 'march1', 'squat1', 'crane', 'jumping', 'march2', 'squat2']
            for pose in pose_names:
                mesh_dirpath = mesh_dirpath_patt.format(pose)
                for fname in os.listdir(mesh_dirpath):
                    mesh_fullpath = os.path.join(mesh_dirpath, fname)
                    label_fullpath = os.path.join(label_dirpath, "mit_{}_corrected.txt".format(pose))
                    mesh_path.append(mesh_fullpath)
                    label_files.append(label_fullpath)

            # scape
            mesh_dirpath = os.path.join(self.data_path, "meshes", "train", "scape")
            label_dirpath = os.path.join(self.data_path, "segs", "train", "scape")
            for fname in os.listdir(mesh_dirpath):
                mesh_fullpath = os.path.join(mesh_dirpath, fname)
                label_fullpath = os.path.join(label_dirpath, "scape_corrected.txt")
                mesh_path.append(mesh_fullpath)
                label_files.append(label_fullpath)

        else:

            # shrec
            mesh_dirpath = os.path.join(self.data_path, "meshes", "test", "shrec")
            label_dirpath = os.path.join(self.data_path, "segs", "test", "shrec")
            for iShrec in range(1, 21):
                if iShrec == 16 or iShrec == 18: continue  # why are these messing from the dataset? so many questions...
                if iShrec == 12:
                    mesh_fname = "12_fix_orientation.off"
                else:
                    mesh_fname = "{}.off".format(iShrec)
                label_fname = "shrec_{}_full.txt".format(iShrec)
                mesh_fullpath = os.path.join(mesh_dirpath, mesh_fname)
                label_fullpath = os.path.join(label_dirpath, label_fname)
                mesh_path.append(mesh_fullpath)
                label_files.append(label_fullpath)

        print("loading {} meshes".format(len(mesh_path)))


        self.labels_list = []  # per-faces labels
        self.mesh_path_list = []

        for iFile in range(len(mesh_path)):
            labels = np.loadtxt(label_files[iFile]).astype(int) - 1

            self.labels_list.append(labels)
            self.mesh_path_list.append(mesh_path[iFile])

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
        return self.verts_list[idx], self.faces_list[idx], self.verts_normals_list[idx], self.evals_list[idx], \
               self.evecs_list[idx], self.verts_dihedralAngles_list[idx], self.hks_list[idx], \
               self.mass_list[idx], self.labels_list[idx], self.mesh_path_list[idx]
