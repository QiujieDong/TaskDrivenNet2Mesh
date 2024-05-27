import os

import random
import json
import hashlib
import scipy

import numpy as np
import trimesh
from sklearn.model_selection import train_test_split

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from timm.loss import LabelSmoothingCrossEntropy
from scipy.spatial.transform import Rotation

import math


class RunningAverage:
    """
    Example:
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def same_seed(seed):
    """

    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def save_dict_to_json(d, json_path):
    """
    :param d:
    :param json_path:
    :return:
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_logging(mode, test_acc, test_loss, save_path=None, model=None, train_acc=0.0, train_loss=0.0, epoch=None,
                 save_name='best'):
    if mode == 'train':
        save_dict_to_json(
            {
                'epoch': epoch,
                'Train': {
                    'epoch_loss': train_loss,
                    'epoch_acc': train_acc,
                },
                'Val': {
                    'epoch_loss': test_loss,
                    'epoch_acc': test_acc
                }
            }, os.path.join(save_path, '{}.json'.format(save_name))
        )
        torch.save(model.state_dict(), os.path.join(save_path, '{}.pth'.format(save_name)))
    else:
        save_dict_to_json(
            {
                'Metric': {
                    'loss': test_loss,
                    'face_acc': test_acc
                }
            }, os.path.join(save_path, 'metric.json')
        )


def save_logging_for_correspondence(mode, test_loss, test_geodesic_error, save_path=None, model=None, train_loss=0.0,
                                    epoch=None, save_name='best'):
    if mode == 'train':
        save_dict_to_json(
            {
                'epoch': epoch,
                'Train': {
                    'epoch_loss': train_loss,
                },
                'Val': {
                    'epoch_loss': test_loss,
                    'epoch_geodesic_error': test_geodesic_error
                }
            }, os.path.join(save_path, '{}.json'.format(save_name))
        )
        torch.save(model.state_dict(), os.path.join(save_path, '{}.pth'.format(save_name)))
    else:
        save_dict_to_json(
            {
                'Metric': {
                    'loss': test_loss,
                    'epoch_geodesic_error': test_geodesic_error
                }
            }, os.path.join(save_path, 'metric.json')
        )


class LossAdjacency(nn.Module):
    def __init__(self, verts, adj_matrix, bandwidth):
        super(LossAdjacency, self).__init__()
        self.verts = verts
        self.adj_matrix = adj_matrix
        self.bandwidth = bandwidth

    def forward(self, pred, gt):
        pred = torch.log_softmax(pred, dim=-1)

        gt = Tensor2Array(gt)
        idx = gt != -1
        pred_in_gt_position = pred[idx.squeeze(), gt[idx]].unsqueeze(1)

        pred_cdist = torch.cdist(pred_in_gt_position, pred_in_gt_position, p=2,
                                 compute_mode='use_mm_for_euclid_dist_if_necessary')

        # Euclidean_distance
        euclidean_distance = torch.cdist(self.verts, self.verts, p=2,
                                         compute_mode='use_mm_for_euclid_dist_if_necessary').float()

        euc_dis_filter = torch.exp(- euclidean_distance / (2 * self.bandwidth))

        pred_cdist = euc_dis_filter * pred_cdist
        verts_adj = self.adj_matrix.to_dense()
        pred_adjacency = (pred_cdist * verts_adj).sum(1)
        adj_num = verts_adj.sum(1)
        adj_num[adj_num[:] == 0] = 1
        loss_neighbor = (pred_adjacency.div(adj_num)).sum() / gt.shape[0]

        return loss_neighbor


def AttentionFeatures(features, feature_rate=0.5):
    num = features.shape[0]
    norm_features = features - torch.mean(features, dim=-1).reshape(-1, 1)

    weight = torch.zeros(num, num).to(features.device)
    for i in range(num):
        weight[i, :] = F.cosine_similarity(norm_features[i, :], norm_features, dim=1)

    features_weight = torch.mm(weight, features) / torch.tensor(num)

    features = (1 - feature_rate) * features + feature_rate * features_weight

    return features


# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, weight, smoothing=0.1):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         self.weight = weight
#         self.smoothing = smoothing
#     def forward(self, x, target):
#         confidence = 1. - self.smoothing
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = self.weight * (confidence * nll_loss + self.smoothing * smooth_loss)
#         return loss.mean()

def cluster_faces(pred_labels, mesh_path, num_class):
    mesh = trimesh.load_mesh(mesh_path, process=False)

    face_adj = mesh.face_adjacency
    device = pred_labels.device
    pred_labels = Tensor2Array(pred_labels)

    pred_cluster_A = np.zeros((face_adj.shape[0], 2))
    pred_cluster_B = np.zeros((face_adj.shape[0], 2))

    pred_cluster_A[:, 0] = face_adj[:, 0]
    pred_cluster_A[:, 1] = pred_labels[face_adj[:, 1]]
    pred_cluster_B[:, 0] = face_adj[:, 1]
    pred_cluster_B[:, 1] = pred_labels[face_adj[:, 0]]

    pred_cluster = np.concatenate((pred_cluster_A, pred_cluster_B), axis=0)

    _, conut = np.unique(pred_cluster[:, 0], return_counts=True)

    pred = np.zeros((pred_labels.shape[0], num_class))

    if np.count_nonzero(conut - 3) == 0:
        ind = np.argsort(pred_cluster[:, 0])
        ind = ind.reshape(-1, 3)

        one_hot = np.eye(num_class)[pred_cluster[:, 1].astype(int)]

        for i in range(3):
            pred += one_hot[ind[:, i], :]
        pred = torch.from_numpy(np.argmax(pred, axis=1)).to(device)
    else:
        for i in range(face_adj.shape[0]):
            face_i = face_adj[i, 0]
            face_j = face_adj[i, 1]
            pred[face_i, pred_labels[face_j]] += 1
            pred[face_j, pred_labels[face_i]] += 1
        pred = torch.from_numpy(np.argmax(pred, axis=1)).to(device)

    return pred


def cluster_faces_neighbor(pred_labels, mesh_path, num_class):
    mesh = trimesh.load_mesh(mesh_path, process=False)

    face_neigh = mesh.face_neighborhood
    device = pred_labels.device
    pred_labels = Tensor2Array(pred_labels)

    pred_cluster = np.zeros((face_neigh.shape[0], 2))
    pred_cluster[:, 0] = face_neigh[:, 0]
    pred_cluster[:, 1] = pred_labels[face_neigh[:, 1]]

    one_hot = np.eye(num_class)[pred_cluster[:, 1].astype(int)]

    _, indices, counts = np.unique(pred_cluster[:, 0], return_index=True, return_counts=True)

    pred = np.zeros((pred_labels.shape[0], num_class))
    for i in range(pred_labels.shape[0]):
        start = indices[i]
        num = counts[i]
        for j in range(num):
            ind = start + j
            pred[i] += one_hot[ind]
    pred = torch.from_numpy(np.argmax(pred, axis=1)).to(device)

    return pred


# class AdjacencyLoss(nn.Module):
#
#     def __init__(self, mesh_path):
#         super(AdjacencyLoss, self).__init__()
#         self.mesh_path = mesh_path
#         self.mesh = trimesh.load_mesh(self.mesh_path, process=False)
#         self.face_adj = self.mesh.face_adjacency
#
#     def forward(self, pred_faces, labels):
#         logprobs = F.softmax(pred_faces, dim=-1)
#         probs_max = torch.max(logprobs, dim=-1)
#         probs_max_index = probs_max.indices
#         probs_max = probs_max.values
#
#         faces_adj = torch.from_numpy(np.concatenate((self.face_adj, self.face_adj[:, [1, 0]]), axis=0))
#
#         diff_i2j = probs_max[faces_adj[:, 0]] - logprobs[faces_adj[:, 1], probs_max_index[faces_adj[:, 0]]]
#         diff_j2i = probs_max[faces_adj[:, 1]] - logprobs[faces_adj[:, 0], probs_max_index[faces_adj[:, 1]]]
#         loss_adj = 0.5 * torch.norm((diff_i2j + diff_j2i), p=1) / 3
#
#         # diff_ij = 0.5 * (torch.norm(diff_i2j, p=1) + torch.norm(diff_j2i, p=1))
#         # loss_adj = diff_ij / 3
#
#         probs_in_gt_pos = logprobs[:, labels[:]]
#         loss_gt = torch.mean(torch.exp(-probs_in_gt_pos) - 1 / torch.exp(torch.tensor(1)))
#
#         loss = 0.5 * (loss_adj + loss_gt)
#
#         return loss


class HardClusteringLoss(nn.Module):
    def __init__(self, alpha=0.1, mesh_path=None):
        super(HardClusteringLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.mesh_path = mesh_path
        self.mesh = trimesh.load_mesh(self.mesh_path, process=False)
        self.face_adj = self.mesh.face_adjacency
        self.adj_angles = self.mesh.face_adjacency_angles

    def forward(self, pred_faces):
        device = pred_faces.device
        face_adj = torch.tensor(self.face_adj).to(device)
        adj_angles = torch.tensor(self.adj_angles).to(device)
        self.alpha = self.alpha.to(device)

        eps = torch.tensor(1e-8).to(device)

        logprobs = F.softmax(pred_faces, dim=-1)
        probs_max = torch.max(logprobs, dim=-1)
        probs_max_index = probs_max.indices
        probs_max = probs_max.values

        probs_max[probs_max[:] < eps] = eps
        e_1 = -torch.log(probs_max)

        adj_angles[adj_angles[:] == 0] = 0.1
        e_2 = -torch.log(adj_angles / torch.pi)
        e_2[probs_max_index[face_adj[:, 0]] == probs_max_index[face_adj[:, 1]]] = 0

        loss = torch.mean(e_1) + self.alpha * torch.mean(e_2) * 2

        return loss


# class segmentation_loss(nn.Module):
#     def __init__(self, smoothing=0.1, loss_rate=0.1, bandwidth=1.0, use_adj_loss=False, num_classes=None, iter_num=5):
#         super(segmentation_loss, self).__init__()
#         self.smoothing = smoothing
#         self.loss_rate = loss_rate
#         self.bandwidth = bandwidth
#         self.use_adj_loss = use_adj_loss
#         self.iter_num = iter_num
#         self.num_classes = num_classes
#
#     def forward(self, pred_faces, labels, mesh_path=None):
#         preds = torch.log_softmax(pred_faces, dim=-1)
#         pred_labels = torch.max(preds, dim=1).indices
#
#         # for i in range(self.iter_num):
#         cluster_labels = cluster_faces_neighbor(pred_labels, mesh_path, self.num_classes)
#
#         ind = (pred_labels - cluster_labels) != 0
#         pred_faces_clone = pred_faces.clone()
#         pred_faces[ind, cluster_labels[ind]] = pred_faces_clone[ind, pred_labels[ind]]
#         pred_faces[ind, pred_labels[ind]] = pred_faces_clone[ind, cluster_labels[ind]]
#
#         criterion = LabelSmoothingCrossEntropy(smoothing=self.smoothing).to(device=pred_faces.device)
#         loss = criterion(pred_faces, labels)
#
#         if self.use_adj_loss:
#             criterion_hard = HardClusteringLoss(mesh_path=mesh_path).to(device=pred_faces.device)
#             loss = loss + self.loss_rate * criterion_hard(pred_faces)
#
#         return loss

class segmentation_loss(nn.Module):
    def __init__(self, smoothing=0.1, loss_rate=0.1, bandwidth=1.0, use_adj_loss=False, num_classes=None,
                 iter_num=5):
        super(segmentation_loss, self).__init__()
        self.smoothing = smoothing
        self.loss_rate = loss_rate
        self.bandwidth = bandwidth
        self.use_adj_loss = use_adj_loss
        self.iter_num = iter_num
        self.num_classes = num_classes

    def forward(self, pred_faces, labels, mesh_path=None):

        criterion = LabelSmoothingCrossEntropy(smoothing=self.smoothing).to(device=pred_faces.device)
        loss = criterion(pred_faces, labels)

        return loss


def normalize_positions(pos, faces=None, method='mean', scale_method='max_rad'):
    # center and unit-scale positions
    if method == 'mean':
        # center using the average point position
        pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    elif method == 'bbox':
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, dim=-2).values
        bbox_max = torch.max(pos, dim=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center.unsqueeze(-2)
    else:
        raise ValueError("unrecognized method")

    if scale_method == 'max_rad':
        scale = torch.max(torch.norm(pos, dim=-1), dim=-1, keepdim=True).values.unsqueeze(-1)
        pos = pos / scale
    elif scale_method == 'area':
        if faces is None:
            raise ValueError("must pass faces for area normalization")
        coords = pos[faces]
        vec_A = coords[:, 1, :] - coords[:, 0, :]
        vec_B = coords[:, 2, :] - coords[:, 0, :]
        face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
        total_area = torch.sum(face_areas)
        scale = (1. / torch.sqrt(total_area))
        pos = pos * scale
    else:
        raise ValueError("unrecognized scale method")
    return pos


def vertices_normalize(vertices):
    verts = torch.sub(vertices, torch.min(vertices, dim=0, keepdim=True)[0])
    verts = torch.div(verts, torch.max(verts, dim=0, keepdim=True)[0])
    return verts


def normalize_mesh(mesh):
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)

    return torch.from_numpy(mesh.vertices).float()


# normalization the data to [-1, 1]
def normalize_data(X, min=-1, max=1):
    data_max = torch.max(X)
    data_min = torch.min(X)

    k = (max - min) / (data_max - data_min)

    Y = min + k * (X - data_min)

    return Y


# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()


# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = Tensor2Array(A.indices())
    values = Tensor2Array(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()


def Tensor2Array(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()


def normalization(x):
    x = (x - torch.min(x, dim=-2, keepdim=True).values) / (
            torch.max(x, dim=-2, keepdim=True).values - torch.min(x, dim=-2, keepdim=True).values)
    return x


def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randgen is None:
        randgen = np.random.RandomState()

    theta, phi, z = tuple(randgen.rand(3).tolist())

    theta = theta * 2.0 * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def random_rotate_points(pts, verts_normals=None, randgen=None):
    R = random_rotation_matrix(randgen)
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    if verts_normals != None:
        return torch.matmul(pts, R), torch.matmul(verts_normals, R)
    else:
        return torch.matmul(pts, R)


def random_rotate_points_axis_y(pts, verts_normals=None):
    pts_device = pts.device
    pts_dtype = pts.dtype

    # random rotation 3-Dim direction
    # axis_seq = ''.join(random.sample('xyz', 3))
    # angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
    # rotation = Rotation.from_euler(axis_seq, angles, degrees=True)

    # random rotation 1-Dim (y-axis) direction
    angles = [random.choice([0, 90, 180, 270])]
    rotation = Rotation.from_euler('y', angles, degrees=True)

    pts = rotation.apply(Tensor2Array(pts))

    if verts_normals != None:
        verts_normals = rotation.apply(Tensor2Array(verts_normals))
        return torch.from_numpy(pts).to(device=pts_device, dtype=pts_dtype), torch.from_numpy(verts_normals).to(
            device=pts_device, dtype=pts_dtype)
    else:
        return torch.from_numpy(pts).to(device=pts_device, dtype=pts_dtype)


def random_rotate_points_axis_x(pts, verts_normals=None):
    pts_device = pts.device
    pts_dtype = pts.dtype

    # random rotation 3-Dim direction
    # axis_seq = ''.join(random.sample('xyz', 3))
    # angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
    # rotation = Rotation.from_euler(axis_seq, angles, degrees=True)

    # random rotation 1-Dim (y-axis) direction
    angles = [random.choice([0, 90])]
    rotation = Rotation.from_euler('x', angles, degrees=True)

    pts = rotation.apply(Tensor2Array(pts))

    if verts_normals != None:
        verts_normals = rotation.apply(Tensor2Array(verts_normals))
        return torch.from_numpy(pts).to(device=pts_device, dtype=pts_dtype), torch.from_numpy(verts_normals).to(
            device=pts_device, dtype=pts_dtype)
    else:
        return torch.from_numpy(pts).to(device=pts_device, dtype=pts_dtype)


# Python string/file utilities
def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def warm_up_with_cosine_lr(warm_up_epochs, eta_min, base_lr, epochs, T_max=0):
    if T_max == 0:
        T_max = epochs
    warm_up_with_cosine_lr = lambda \
            epoch: (eta_min + (
            base_lr - eta_min) * epoch / warm_up_epochs) / base_lr if epoch <= warm_up_epochs else (eta_min + 0.5 * (
            base_lr - eta_min) * (math.cos(
        (epoch - warm_up_epochs) * math.pi / (T_max - warm_up_epochs)) + 1)) / base_lr
    # warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * (
    #             math.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * math.pi) + 1)
    return warm_up_with_cosine_lr


# Spectral Reconstructing mesh
# The color encodes the error of reconstructed coordinates.
def visualization_spectral_reconstruct_mesh(mesh_path, save_path, K_values):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    mesh: trimesh.Trimesh

    # the path of saving the result of reconstruct
    reconstruct_save_path = os.path.join(save_path, os.path.splitext(mesh_path.split('/')[-1])[0])
    if not os.path.exists(reconstruct_save_path):
        os.makedirs(reconstruct_save_path)

    # eigenvectors
    cot = -igl.cotmatrix(mesh.vertices, mesh.faces).toarray()
    cot = torch.from_numpy(cot).float().to('cuda')
    eigen_values, eigen_vectors = torch.linalg.eigh(cot)
    ind = torch.argsort(eigen_values)[:]
    eigen_vector = eigen_vectors[:, ind].cpu().numpy()

    for k in K_values:
        mesh_trans = mesh.copy()
        vertices_new = eigen_vector[:, :k] @ (eigen_vector[:, :k].T @ mesh_trans.vertices)

        error_reconstructed_coordinates = np.sqrt(np.power((mesh_trans.vertices - vertices_new), 2).sum(axis=1))
        max_error = error_reconstructed_coordinates.max()
        colors = plt.get_cmap("viridis")(error_reconstructed_coordinates / (max_error if max_error > 0 else 1))

        # mesh show or save
        mesh_trans.visual.vertex_colors = colors[:, :3]
        mesh_trans.vertices = vertices_new
        # mesh.show(smooth=False)
        mesh_trans.export(os.path.join(reconstruct_save_path,
                                       os.path.splitext(mesh_path.split('/')[-1])[0] + '_kValue{0}.obj'.format(k)))


def compute_hks_autoscale(evals, evecs, count=16):
    # these scales roughly approximate those suggested in the hks paper
    """
            Inputs:
              - evals: (K) eigenvalues
              - evecs: (V,K) values
              - count: num step
            Outputs:
              - (V,S) hks values
            """
    # scales: (S) times
    scales = torch.logspace(-2, 0., steps=count, device=evals.device, dtype=evals.dtype)

    # expand batch
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    power_coefs = torch.exp(-evals.unsqueeze(1) * scales.unsqueeze(-1)).unsqueeze(1)  # (B,1,S,K)
    terms = power_coefs * (evecs * evecs).unsqueeze(2)  # (B,V,S,K)

    out = torch.sum(terms, dim=-1)  # (B,V,S)

    if expand_batch:
        return out.squeeze(0)
    else:
        return out


def split_dataset_and_generate_labels(path):
    meshes_path = os.path.join(path, 'obj')
    old_gt_path = os.path.join(path, 'ad')
    new_gt_path = os.path.join(path, 'seg')

    ensure_folder_exists(new_gt_path)

    files_list = list()

    for file in sorted(os.listdir(meshes_path)):
        file.strip()
        if os.path.isdir(os.path.join(meshes_path, file)):
            continue

        files_list.append(file)

        old_labels_path = os.path.join(old_gt_path, file.split('_')[0] + '-_norm.ad')
        new_labels_path = os.path.join(new_gt_path, file.split('_')[0] + '.eseg')

        old_labels = np.loadtxt(old_labels_path)[:, -1]

        np.savetxt(new_labels_path, old_labels, fmt='%1u')

    train_set, test_set = train_test_split(files_list, random_state=0, train_size=0.8)

    np.savetxt(os.path.join(path, 'train.txt'), train_set, fmt='%s')
    np.savetxt(os.path.join(path, 'test.txt'), test_set, fmt='%s')


def compute_IoU(pred_labels, labels, n_class=2):
    NUMCLASS = n_class
    pred_labels = pred_labels.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    pred_labels += 1
    labels += 1
    intersection = pred_labels * (pred_labels == labels)
    area_inter, _ = np.histogram(intersection, bins=NUMCLASS, range=(1, NUMCLASS))
    area_pred, _ = np.histogram(pred_labels, bins=NUMCLASS, range=(1, NUMCLASS))
    area_label, _ = np.histogram(labels, bins=NUMCLASS, range=(1, NUMCLASS))
    area_union = area_pred + area_label - area_inter
    iou = area_inter / area_union

    return iou[0], iou[1]


def compute_SD_coefficient(pred_labels, labels, n_class=2):
    NUMCLASS = n_class
    pred_labels = pred_labels.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    pred_labels += 1
    labels += 1
    intersection = pred_labels * (pred_labels == labels)
    area_inter, _ = np.histogram(intersection, bins=NUMCLASS, range=(1, NUMCLASS))
    area_pred, _ = np.histogram(pred_labels, bins=NUMCLASS, range=(1, NUMCLASS))
    area_label, _ = np.histogram(labels, bins=NUMCLASS, range=(1, NUMCLASS))
    DSC = (2 * area_inter) / (area_pred + area_label)

    return DSC[0], DSC[1]


def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,D,K) transformed values
    """
    valuesT = values.permute(0, -1, -2)
    return torch.matmul(valuesT * massvec, basis)


def corresponds_loss(C_pred, C_gt):
    return torch.mean(torch.square(C_pred - C_gt))



class LossClassInOut(nn.Module):
    def __init__(self):
        super(LossClassInOut, self).__init__()

    def forward(self, pred, pseudo, labels=None, used_labels=False):

        pseudo_labels = torch.max(pseudo, dim=-1).indices
        pred_labels = torch.max(pred, dim=-1).indices

        self.pred_loss_list = list()
        self.pseudo_loss_list = list()
        self.pred_diff_list = list()
        self.pseudo_diff_list = list()

        if used_labels:
            pred_loss_nllLoss = torch.nn.functional.nll_loss(pred, labels)
            for i, i_class in enumerate(labels.unique()):
                # intra-class gap
                index = (labels[:] == i_class)

                pred_iClass = pred[index, :]
                pred_iClass_mean = torch.mean(pred_iClass, dim=0)
                pred_loss = torch.abs(pred_iClass[:] - pred_iClass_mean).mean()

                pseudo_iClass = pseudo[index, :]
                pseudo_iClass_mean = torch.mean(pseudo_iClass, dim=0)
                pseudo_loss = torch.abs(pseudo_iClass[:] - pseudo_iClass_mean).mean()

                # gap between classes
                pred_other_class_mean = torch.mean(pred[~index, :], dim=0)
                pseudo_other_class_mean = torch.mean(pseudo[~index, :], dim=0)
                pred_diff = torch.exp(-5 * (torch.abs(pred_iClass_mean - pred_other_class_mean).mean()))
                pseudo_diff = torch.exp(-5 * (torch.abs(pseudo_iClass_mean - pseudo_other_class_mean).mean()))

                self.pred_loss_list.append(pred_loss)
                self.pseudo_loss_list.append(pseudo_loss)
                self.pred_diff_list.append(pred_diff)
                self.pseudo_diff_list.append(pseudo_diff)

            loss = 0.5 * (pred_loss_nllLoss + torch.stack(
                (self.pred_loss_list + self.pseudo_loss_list + self.pred_diff_list + self.pseudo_diff_list),
                dim=0).mean())

        else:
            for i, i_class in enumerate(pseudo_labels.unique()):
                index = (pseudo_labels[:] == i_class)

                pred_iClass = pred[index, :]
                pred_iClass_mean = torch.mean(pred_iClass, dim=0)
                pred_loss = torch.abs(pred_iClass[:] - pred_iClass_mean).mean()

                if len(pseudo_labels.unique()) > 1:
                    pred_other_class_mean = torch.mean(pred[~index, :], dim=0)
                    pred_diff = torch.exp(-5 * torch.abs(pred_iClass_mean - pred_other_class_mean).mean())
                else:
                    pred_diff = torch.exp(torch.tensor(0).to(pred.device))

                self.pred_loss_list.append(pred_loss)
                self.pred_diff_list.append(pred_diff)

            for i, i_class in enumerate(pred_labels.unique()):
                index = (pred_labels[:] == i_class)

                pseudo_iClass = pseudo[index, :]
                pseudo_iClass_mean = torch.mean(pseudo_iClass, dim=0)
                pseudo_loss = torch.abs(pseudo_iClass[:] - pseudo_iClass_mean).mean()

                if len(pred_labels.unique()) > 1:
                    pseudo_other_class_mean = torch.mean(pseudo[~index, :], dim=0)
                    pseudo_diff = torch.exp(-5 * torch.abs(pseudo_iClass_mean - pseudo_other_class_mean).mean())
                else:
                    pseudo_diff = torch.exp(torch.tensor(0).to(pred.device))

                self.pseudo_loss_list.append(pseudo_loss)
                self.pseudo_diff_list.append(pseudo_diff)

            loss = torch.stack((self.pred_loss_list + self.pseudo_loss_list + self.pred_diff_list + self.pseudo_diff_list),
                       dim=0).mean()

        return loss

def log_string(out_str, log_file):
    # helper function to log a string to file and print it
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)



if __name__ == '__main__':
    import igl
    import matplotlib.pyplot as plt

    meshes_path = 'data/paper_data'

    # The origin mesh is simplified to 2k faces (about 1k vertices) by Quadric Edge Collapse Decimation.
    K_values = [1000, 800, 600, 500, 400, 200, 100, 80, 60, 40, 20, 10]

    save_path = 'data/paper_data/reconstruct'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file in sorted(os.listdir(meshes_path)):
        file.strip()
        if os.path.splitext(file)[1] not in ['.obj', '.off', '.ply']:
            continue
        # if file != 'homer_faces2k_vertice1002.obj':
        #     continue
        print(os.path.join(meshes_path, file))

        mesh_path = os.path.join(meshes_path, file)

        visualization_spectral_reconstruct_mesh(mesh_path, save_path, K_values)
