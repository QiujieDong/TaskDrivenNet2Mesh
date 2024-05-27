import os.path

import numpy as np

import torch

import trimesh
import igl
import potpourri3d as pp3d
import robust_laplacian

import scipy
import scipy.sparse.linalg as sla
import scipy.spatial

import sklearn.neighbors

import utils.util as utils


# import robust_laplacian


def compute_all_operators(args, mesh_path_list, dataset_cache_dir, is_pointCloud=False):
    num_mesh = len(mesh_path_list)

    verts_list = [None] * num_mesh
    faces_list = [None] * num_mesh
    verts_normals_list = [None] * num_mesh
    evals_list = [None] * num_mesh
    evecs_list = [None] * num_mesh
    verts_dihedralAngles_list = [None] * num_mesh
    hks_list = [None] * num_mesh
    mass_list = [None] * num_mesh

    for i in range(num_mesh):
        print("processing mesh: {} / {}, {:.3f}%".format((i + 1), num_mesh, (i + 1) / num_mesh * 100))

        operators = get_all_operators(args, mesh_path_list[i], dataset_cache_dir, is_pointCloud)

        verts_list[i] = operators[0]
        faces_list[i] = operators[1]
        verts_normals_list[i] = operators[2]
        evals_list[i] = operators[3]
        evecs_list[i] = operators[4]
        verts_dihedralAngles_list[i] = operators[5]
        hks_list[i] = operators[6]
        mass_list[i] = operators[7]

    return verts_list, faces_list, verts_normals_list, evals_list, evecs_list, verts_dihedralAngles_list, hks_list, mass_list


def get_all_operators(args, mesh_path, dataset_cache_dir=None, is_pointCloud=False):
    """
        See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
        All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
        """
    if not is_pointCloud:
        mesh = trimesh.load_mesh(mesh_path, process=False)
        verts_np = mesh.vertices
        faces_np = mesh.faces
        if (np.isnan(verts_np).any()):
            raise RuntimeError("tried to construct operators from NaN verts")
    else:
        verts_np = mesh_path
        faces_np = np.zeros(1)  # just for the code to run

    max_k_eig = np.max(args.k_eig_list)
    device = args.device
    dtype = torch.float32

    # Check the cache directory
    # Note 1: Collisions here are exceptionally unlikely, so we could probably just use the hash...
    #         but for good measure we check values nonetheless.
    # Note 2: There is a small possibility for race conditions to lead to bucket gaps or duplicate
    #         entries in this cache. The good news is that that is totally fine, and at most slightly
    #         slows performance with rare extra cache misses.
    get_cache = False
    if dataset_cache_dir is not None:
        utils.ensure_folder_exists(dataset_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                dataset_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if ("verts" not in npzfile) or ("faces" not in npzfile) or ("verts_normals" not in npzfile) or (
                        "evals" not in npzfile) or ("evecs" not in npzfile) or (
                        "verts_dihedralAngles" not in npzfile) or ("hks" not in npzfile) or ("mass" not in npzfile):
                    print("---overwriting---entries are absent")
                    os.remove(search_path)
                    break

                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"]

                # If the cache doesn't match, keep looking
                if (not np.array_equal(faces_np, cache_faces)):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    # searching 10 times, preventing infinite loop
                    if i_cache_search == 10:
                        print("---overwriting---Have searching 10 times!")
                        os.remove(search_path)
                        break
                    continue

                if cache_k_eig < max_k_eig:
                    print("---overwriting---not enough eigenvalues")
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                verts_normals = npzfile["verts_normals"]
                evals = npzfile["evals"][:max_k_eig]
                evecs = npzfile["evecs"][:, :max_k_eig]
                verts_dihedralAngles = npzfile["verts_dihedralAngles"]
                hks = npzfile["hks"]
                mass = npzfile["mass"]


                verts = torch.from_numpy(cache_verts).to(device=device, dtype=dtype)
                faces = torch.from_numpy(cache_faces).to(device=device)
                verts_normals = torch.from_numpy(verts_normals).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                verts_dihedralAngles = torch.from_numpy(verts_dihedralAngles).to(device=device,
                                                                                 dtype=dtype)
                hks = torch.from_numpy(hks).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)

                get_cache = True

                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break

            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not get_cache:

        # No matching entry found; recompute.
        utils.ensure_folder_exists(dataset_cache_dir)

        faces = torch.tensor(np.ascontiguousarray(faces_np))
        if not is_pointCloud:
            verts = normalize_verts(verts_np, args, faces, mesh)

            # vertices normals
            verts_normals = mesh.vertex_normals / np.linalg.norm(mesh.vertex_normals, axis=-1, keepdims=True)

            verts_dihedralAngles = compute_dihedral_angles_of_vertex(mesh.vertex_faces, mesh.face_adjacency,
                                                                     mesh.face_normals).to(device=device, dtype=dtype)
        else:
            verts = normalize_verts(verts_np, args)

            # vertices normals
            verts_normals = pointCloud_normal(verts, args.n_neighbors_cloud)

            verts_dihedralAngles = torch.zeros(1)  # just for the code to run

        verts_normals = torch.tensor(np.ascontiguousarray(verts_normals)).to(device=device, dtype=dtype)

        # eigen decomposition for large mesh
        evals, evecs, mass = compute_L_eigenvalue_decomposition_for_large_mesh_from_PP3d(verts, faces,
                                                                                           max_k_eig, device,
                                                                                           is_pointCloud)
        # for the bigger-mesh， we compute HKS using only partial engenvectors.
        hks = compute_hks_autoscale(evals, evecs, count=args.hks_count)


        # Store it in the cache
        if dataset_cache_dir is not None:
            dtype_np = np.float32
            np.savez(search_path,
                     verts=utils.Tensor2Array(verts).astype(dtype_np),
                     faces=utils.Tensor2Array(faces),
                     k_eig=max_k_eig,
                     verts_normals=utils.Tensor2Array(verts_normals).astype(dtype_np),
                     evals=utils.Tensor2Array(evals).astype(dtype_np),
                     evecs=utils.Tensor2Array(evecs).astype(dtype_np),
                     verts_dihedralAngles=utils.Tensor2Array(verts_dihedralAngles).astype(dtype_np),
                     hks=utils.Tensor2Array(hks).astype(dtype_np),
                     mass=utils.Tensor2Array(mass).astype(dtype_np)
                     )

    return verts, faces, verts_normals, evals, evecs, verts_dihedralAngles, hks, mass


def normalize_verts(verts_np, args, faces=None, mesh=None):
    verts = torch.tensor(np.ascontiguousarray(verts_np)).float()

    if args.norm_selection == 'center_unit':
        verts = utils.normalize_positions(verts, faces, method=args.norm_method, scale_method=args.norm_scale_method)
    elif args.norm_selection == 'norm_unit':
        verts = utils.vertices_normalize(verts)
    elif mesh != None and args.norm_selection == 'norm_mesh':
        verts = utils.normalize_mesh(mesh)
    else:
        raise ValueError("unrecognized normalization method")
    return verts


def compute_L_eigenvalue_decomposition_for_large_mesh_from_PP3d(verts, faces, max_k_eig, device, is_pointCloud=False):
    eps = 1e-8
    verts_np = utils.Tensor2Array(verts).astype(np.float64)
    faces_np = utils.Tensor2Array(faces)

    if not is_pointCloud:
        L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(verts_np, faces_np)
        massvec_np += eps * np.mean(massvec_np)
    else:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
        massvec_np = M.diagonal()

    if (np.isnan(L.data).any()):
        raise RuntimeError("NaN Laplace matrix")
    if (np.isnan(massvec_np).any()):
        raise RuntimeError("NaN mass matrix")

    # === Compute the eigenbasis
    if max_k_eig > 0:
        # Prepare matrices
        cot_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
        Mmat = scipy.sparse.diags(massvec_np)

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise, but for some reason those parameters seem to have no effect
                evals_np, evecs_np = sla.eigsh(cot_eigsh, k=max_k_eig, M=Mmat, sigma=eps)
                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))
                break
            except Exception as e:
                print(e)
                if (failcount > 3):
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                cot_eigsh = cot_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10 ** failcount)
    else:  # k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0], 0))

    evals = torch.from_numpy(evals_np).to(device=device, dtype=torch.float32)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=torch.float32)
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=torch.float32)
    return evals, evecs, massvec



def compute_cot_eigenvalue_decomposition_for_large_mesh_from_PP3d(verts, faces, max_k_eig, device, is_pointCloud=False):
    eps = 1e-8
    verts_np = utils.Tensor2Array(verts).astype(np.float64)
    faces_np = utils.Tensor2Array(faces)

    if not is_pointCloud:
        L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(verts_np, faces_np)
        massvec_np += eps * np.mean(massvec_np)
    else:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
        massvec_np = M.diagonal()

    if (np.isnan(L.data).any()):
        raise RuntimeError("NaN Laplace matrix")
    if (np.isnan(massvec_np).any()):
        raise RuntimeError("NaN mass matrix")

    # === Compute the eigenbasis
    if max_k_eig > 0:
        # Prepare matrices
        cot_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
        Mmat = scipy.sparse.diags(massvec_np)

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise, but for some reason those parameters seem to have no effect
                evals_np, evecs_np = sla.eigsh(cot_eigsh, k=max_k_eig, sigma=eps)
                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))
                break
            except Exception as e:
                print(e)
                if (failcount > 3):
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                cot_eigsh = cot_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10 ** failcount)
    else:  # k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0], 0))

    evals = torch.from_numpy(evals_np).to(device=device, dtype=torch.float32)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=torch.float32)
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=torch.float32)
    return evals, evecs, massvec


def compute_cot_eigenvalue_decomposition(verts, faces, device):
    cot = -igl.cotmatrix(verts, faces).toarray()
    cot = torch.from_numpy(cot).float().to(device)
    evals, evecs = torch.linalg.eigh(cot)
    return evals.float(), evecs.float()


def compute_cot_eigenvalue_decomposition_for_large_mesh(verts, faces, device, num_inputs):
    max_k = np.max(num_inputs)
    cot = -igl.cotmatrix(verts, faces).toarray()
    evals_np, evecs_np = sla.eigsh(cot, k=max_k, sigma=1e-8)

    evals = torch.from_numpy(evals_np).to(device=device, dtype=torch.float32)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=torch.float32)
    return evals, evecs


def compute_dihedral_angles_of_vertex(vertex_faces, faces_adjacency, faces_normals):
    # vertex faces adjacency matrix
    vertex_faces_adjacency_matrix = np.zeros((vertex_faces.shape[0], faces_normals.shape[0]), dtype=np.int32)
    valid_vf_rows = np.argwhere(vertex_faces[:] != -1)[:, 0].reshape(-1, 1)
    valid_vf_values = (vertex_faces[vertex_faces[:] != -1]).reshape(-1, 1)
    vertex_faces_adjacency_matrix[valid_vf_rows, valid_vf_values] = 1

    dihedral_angle = list()
    for i in range(faces_normals.shape[0]):
        dihedral_angle.append(list())

    for adj_faces in faces_adjacency:
        dihedral_angle[adj_faces[0]].append(
            np.abs(np.dot(faces_normals[adj_faces[0]], faces_normals[adj_faces[1]])))
        dihedral_angle[adj_faces[1]].append(
            np.abs(np.dot(faces_normals[adj_faces[0]], faces_normals[adj_faces[1]])))

    # process the non-watertight mesh which include some faces which dont have three neighbors.
    for i, l in enumerate(dihedral_angle):
        if (len(l)) == 3:
            continue
        l.append(1)
        if (len(l)) == 3:
            continue
        l.append(1)
        if (len(l)) == 3:
            continue
        l.append(1)
        if (len(l)) != 3:
            print(i, 'Padding Failed')
    face_dihedral_angle = np.array(dihedral_angle).reshape(-1, 3)

    exp_vf_adj = np.exp(1 - face_dihedral_angle)
    vertex_dihedral_angle = np.divide(np.dot(vertex_faces_adjacency_matrix, exp_vf_adj),
                                      np.sum(vertex_faces_adjacency_matrix, axis=1).reshape(-1, 1))

    return torch.from_numpy(vertex_dihedral_angle)


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


def compute_verts_gt_from_faces(face_labels, vertex_faces):
    labels = np.zeros((vertex_faces.shape[0], face_labels.max() + 1), dtype=int)
    v_labels = []

    for vertex, faces in enumerate(vertex_faces):
        for f in faces:
            if f == -1:
                break
            labels[vertex, face_labels[f]] += 1

    v_labels.append(np.argmax(labels, axis=1))
    v_labels = np.array(v_labels)
    v_labels = torch.from_numpy(v_labels).to(dtype=torch.int32)

    return v_labels.permute(1, 0)


def compute_verts_adjacency_matrix(vertex_neighbors, device):
    verts_num = len(vertex_neighbors)
    verts_adjacency_matrix = np.zeros((verts_num, verts_num), dtype=np.int32)

    for i in range(len(vertex_neighbors)):
        for _, j in enumerate(vertex_neighbors[i]):
            verts_adjacency_matrix[i][j] = 1
    verts_adjacency_matrix = verts_adjacency_matrix + np.eye(verts_num)

    mat_ij = np.argwhere(verts_adjacency_matrix[:] != 0)
    mat_data = verts_adjacency_matrix[verts_adjacency_matrix[:] != 0]

    verts_adjacency_matrix_coo = scipy.sparse.coo_matrix((mat_data, (mat_ij[:, 0], mat_ij[:, 1])),
                                                         shape=(verts_num, verts_num))
    adjacency_sparse_matrix = verts_adjacency_matrix_coo.tocsr()

    adjacency_matrix = utils.sparse_np_to_torch(adjacency_sparse_matrix).to(device=device)

    return adjacency_matrix


def compute_gaussian_curvature(verts, faces, device):
    verts_np = np.array(verts)
    faces_np = np.array(faces, dtype=np.int64)
    gaussian_curvature = igl.gaussian_curvature(verts_np, faces_np)
    gaussian_curvature = torch.from_numpy(gaussian_curvature).to(device=device, dtype=torch.float32)

    # process the gaussian curvature
    gaussian_curvature = torch.exp(-gaussian_curvature)
    gaussian_curvature = (gaussian_curvature - gaussian_curvature.min()) / (
            gaussian_curvature.max() - gaussian_curvature.min())

    return gaussian_curvature.unsqueeze(1)


def compute_face_adjacency(face_adjacency, faces_num):
    faces_adjacency_matrix = np.zeros((faces_num, faces_num), dtype=np.int32)
    faces_adjacency_matrix[face_adjacency[:, 0], face_adjacency[:, 1]] = 1
    faces_adjacency_matrix[face_adjacency[:, 1], face_adjacency[:, 0]] = 1
    faces_adjacency_matrix = faces_adjacency_matrix + np.eye(faces_num)

    mat_ij = np.argwhere(faces_adjacency_matrix[:] != 0)
    mat_data = faces_adjacency_matrix[faces_adjacency_matrix[:] != 0]

    faces_adjacency_matrix_coo = scipy.sparse.coo_matrix((mat_data, (mat_ij[:, 0], mat_ij[:, 1])),
                                                         shape=(faces_num, faces_num))

    return faces_adjacency_matrix_coo.tocsr()


def pointCloud_normal(verts, n_neighbors_cloud=30):
    verts_np = utils.Tensor2Array(verts)
    _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
    neigh_points = verts_np[neigh_inds, :]
    neigh_points = neigh_points - verts_np[:, np.newaxis, :]
    normals = neighborhood_normal(neigh_points)

    return normals


# Finds the k nearest neighbors of source on target.
# Return is two tensors (distances, indices). Returned points will be sorted in increasing order of distance.
def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):
    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = torch.norm(diff_mat, dim=-1)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result

    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = utils.Tensor2Array(points_source)
        points_target_np = utils.Tensor2Array(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k + 1 if omit_diagonal else k
        _, neighbors = kd_tree.query(points_source_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1] - 1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = torch.norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds], dim=-1)

        return dists, inds

    else:
        raise ValueError("unrecognized method")


def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:, 2, :]
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)
