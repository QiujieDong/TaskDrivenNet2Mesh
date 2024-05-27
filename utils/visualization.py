import os.path

import torch
import numpy as np

import trimesh
from trimesh import creation

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# import pymeshlab


def vertices_label(mesh_path, pred_label, save_path_name, max_label):
    print(mesh_path)

    mesh = trimesh.load(mesh_path, process=False)
    mesh: trimesh.Trimesh

    # visual test mesh
    vertex_label = np.argmax(pred_label[:len(mesh.vertices), :].cpu().detach().numpy(), axis=1)

    colors = plt.get_cmap("Accent")(vertex_label / (max_label - 1))

    mesh.visual.vertex_colors = colors[:, :3]

    if not os.path.exists(os.path.join('visualization_result', save_path_name, 'vertices')):
        os.makedirs(os.path.join('visualization_result', save_path_name, 'vertices'))

    mesh.export(os.path.join('visualization_result', save_path_name, 'vertices', 'vertex_' + os.path.splitext(mesh_path.split('/')[-1])[0] + '.ply'))


def vertices_label_for_point_cloud(mesh_path, pred_label, save_path_name, max_label):
    print(mesh_path)

    mesh = trimesh.load(mesh_path, process=False)
    mesh: trimesh.Trimesh

    # visual test mesh
    vertex_label = np.argmax(pred_label[:len(mesh.vertices), :].cpu().detach().numpy(), axis=1)

    colors = plt.get_cmap("Accent")(vertex_label / (max_label - 1))

    mesh.visual.vertex_colors = colors[:, :3]

    res_pts_mesh_v = []
    res_pts_mesh_f = []
    res_pts_mesh_face_colors = []
    prev_v_num = 0
    prev_f_num = 0
    for i, p in enumerate(mesh.vertices):
        sphere = creation.icosphere(radius=0.01, subdivisions=1)
        res_pts_mesh_v.append(p + sphere.vertices)
        res_pts_mesh_f.append(sphere.faces + prev_v_num)
        prev_v_num += sphere.vertices.shape[0]

        prev_f_num_new = prev_f_num + sphere.faces.shape[0]
        for j in range(prev_f_num, prev_f_num_new):
            res_pts_mesh_face_colors.append(mesh.visual.vertex_colors[i])
        prev_f_num = prev_f_num_new

    res_pts_mesh = trimesh.Trimesh(vertices=np.concatenate(res_pts_mesh_v), faces=np.concatenate(res_pts_mesh_f))
    res_pts_mesh.visual.face_colors = res_pts_mesh_face_colors

    if not os.path.exists(os.path.join('visualization_result', save_path_name, 'vertices_only')):
        os.makedirs(os.path.join('visualization_result', save_path_name, 'vertices_only'))

    res_pts_mesh.export(os.path.join('visualization_result', save_path_name, 'vertices_only', 'vertex_only_' + os.path.splitext(mesh_path.split('/')[-1])[0] + '.ply'))




def faces_label(mesh_name, pred_labels, save_path_name, max_label):
    mesh = trimesh.load(mesh_name, process=False)
    mesh: trimesh.Trimesh

    pred_labels = pred_labels.detach().cpu().numpy()

    colors = plt.get_cmap("tab20")(
        pred_labels[:] / (max_label - 1))  # the color of 'Accent' from HodgeNet, coseg-chair is 'tab20'

    mesh.visual.face_colors = colors[:, :3]

    mesh.export(os.path.join(save_path_name, 'face_' + os.path.splitext(mesh_name.split('/')[-1])[0] + '.ply'))



def faces_label_from_vertex(mesh_name, pred_labels, save_path_name, max_label):
    mesh = trimesh.load(mesh_name, process=False)
    mesh: trimesh.Trimesh

    pred_labels = pred_labels.detach().cpu().numpy()

    # Remap to faces
    x_gather = pred_labels.unsqueeze(-1).expand(-1, -1, -1, 3)
    faces_gather = mesh.faces.unsqueeze(2).expand(-1, -1, pred_labels.shape[-1], -1)
    xf = torch.gather(x_gather, 1, faces_gather)
    x_out = torch.mean(xf, dim=-1)

    colors = plt.get_cmap("tab20")(
        pred_labels[:] / (max_label - 1))  # the color of 'Accent' from HodgeNet, coseg-chair is 'tab20'

    mesh.visual.face_colors = colors[:, :3]

    mesh.export(os.path.join(save_path_name, 'face_' + os.path.splitext(mesh_name.split('/')[-1])[0] + '.ply'))


def visual_face_label(mesh_path, pred_vertex_label, save_path_name, max_label):
    mesh_path = os.path.join('data/noise_data/test_ply', os.path.splitext(mesh_path.split('/')[-1])[0] + '.ply')
    if not os.path.exists(mesh_path):
        return
    print(mesh_path)
    mesh = trimesh.load(mesh_path, process=False)
    mesh: trimesh.Trimesh

    x_gather = pred_vertex_label.unsqueeze(-1).expand(-1, -1, 3)
    faces_gather = torch.from_numpy(mesh.faces).to(pred_vertex_label.device)
    faces_gather = faces_gather.unsqueeze(1).expand(-1, pred_vertex_label.shape[-1], -1)
    xf = torch.gather(x_gather, 0, faces_gather)
    pred_face = torch.mean(xf, dim=-1)
    preds = torch.log_softmax(pred_face, dim=-1)
    pred_face_label = (torch.max(preds, dim=1).indices).detach().cpu().numpy()

    colors = plt.get_cmap("Accent")(pred_face_label / (max_label - 1))  # the color of 'Accent' from HodgeNet

    mesh.visual.face_colors = colors[:, :3]

    if not os.path.exists(os.path.join('visualization_result', save_path_name, 'faces')):
        os.makedirs(os.path.join('visualization_result', save_path_name, 'faces'))

    mesh.export(
        os.path.join('visualization_result', save_path_name, 'faces', 'gt_ply_face_' + os.path.splitext(mesh_path.split('/')[-1])[0] + '.ply'))


def save_confusion_matrix_figue(
        cmtx,
        num_classes,
        subset_ids=None,
        class_names=None,
        tag="Confusion Matrix",
        figsize=None,
        save_path_name=None
):
    """
    Calculate and plot confusion matrix to a SummaryWriter.
    Args:
        writer (SummaryWriter): the SummaryWriter to write the matrix to.
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        global_step (Optional[int]): current step.
        subset_ids (list of ints): a list of label indices to keep.
        class_names (list of strs, optional): a list of all class names.
        tag (str or list of strs): name(s) of the confusion matrix image.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    """
    if subset_ids is None or len(subset_ids) != 0:
        # If class names are not provided, use class indices as class names.
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        # If subset is not provided, take every classes.
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            num_classes=len(subset_ids),
            class_names=sub_names,
            figsize=figsize,
        )

        sub_cmtx.savefig(os.path.join('visualization_result', save_path_name, 'confusion_matrix'))


def visual_model(writer, net, level_0, level_1, level_2, c_1, c_2, c_3, final_mat):
    input_to_model = list()
    input_to_model.append(level_0)
    input_to_model.append(level_1)
    input_to_model.append(level_2)
    input_to_model.append(c_1)
    input_to_model.append(c_2)
    input_to_model.append(c_3)
    if final_mat is not None:
        input_to_model.append(final_mat)
    writer.add_graph(net, input_to_model)


# get confusion_matrix and show in the tensorboard
def get_confusion_matrix(preds, labels, num_classes):
    """
    Calculate confusion matrix on the provided preds and labels.
    Args:
        preds (tensor or lists of tensors): predictions. Each tensor is in
            in the shape of (n_batch, num_classes). Tensor(s) must be on CPU.
        labels (tensor or lists of tensors): corresponding labels. Each tensor is
            in the shape of either (n_batch,) or (n_batch, num_classes).
        num_classes (int): number of classes. Tensor(s) must be on CPU.
        normalize (Optional[str]) : {‘true’, ‘pred’, ‘all’}, default="true"
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix
            will not be normalized.
    Returns:
        cmtx (ndarray): confusion matrix of size (num_classes x num_classes)
    """
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))
    return cmtx


def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


def add_confusion_matrix(
        cmtx,
        num_classes,
        writer=None,
        args=None,
        global_step=None,
        subset_ids=None,
        class_names=None,
        tag="Confusion Matrix",
        figsize=None,
):
    """
    Calculate and plot confusion matrix to a SummaryWriter.
    Args:
        writer (SummaryWriter): the SummaryWriter to write the matrix to.
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        global_step (Optional[int]): current step.
        subset_ids (list of ints): a list of label indices to keep.
        class_names (list of strs, optional): a list of all class names.
        tag (str or list of strs): name(s) of the confusion matrix image.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    """
    if subset_ids is None or len(subset_ids) != 0:
        # If class names are not provided, use class indices as class names.
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        # If subset is not provided, take every classes.
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            num_classes=len(subset_ids),
            class_names=sub_names,
            figsize=figsize,
        )
        # Add the confusion matrix image to writer.
        if args.mode == 'train':
            writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)
        else:
            sub_cmtx.savefig(os.path.join('visualization_result', args.name, 'confusion_matrix'))


def visual_confusion_matrix(args, preds, labels, epoch=None, writer=None, is_train=True):
    class_names = None
    if 'vases' in args.data_path.split('/'):
        class_names = ['neck', 'hand', 'body', 'bottom']
    elif 'chairs' in args.data_path.split('/'):
        class_names = ['back', 'surface', 'leg']
    elif 'aliens' in args.data_path.split('/'):
        class_names = ['eyes', 'leg', 'body', 'ear']
    elif 'shrec' in args.data_path.split('/'):
        # class_names = ['0', '1', '2']
        class_names = os.listdir(os.path.join(args.data_path, 'train_norm'))

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    cmtx = get_confusion_matrix(preds, labels, args.num_classes)

    if args.mode == 'train' and is_train == True:
        add_confusion_matrix(cmtx, num_classes=args.num_classes, writer=writer, args=args, class_names=class_names,
                             tag="Train Confusion Matrix", figsize=[10, 8], global_step=epoch)
    elif args.mode == 'train' and is_train == False:
        add_confusion_matrix(cmtx, num_classes=args.num_classes, writer=writer, args=args, class_names=class_names,
                             tag="Test Confusion Matrix", figsize=[10, 8], global_step=epoch)
    else:
        add_confusion_matrix(cmtx, num_classes=args.num_classes, args=args, class_names=class_names,
                             tag="Test Confusion Matrix", figsize=[10, 8])


def visual_result(epoch, writer, mesh_path, pred_label):
    # visualize train_mesh vertices' color in the tensorboard
    train_mesh = trimesh.load(mesh_path, process=False)
    train_mesh: trimesh.Trimesh
    max_label = np.argmax(pred_label.cpu().detach().numpy(), axis=1).max()
    colors = plt.get_cmap("tab20")(
        np.argmax(pred_label[:len(train_mesh.vertices)].cpu().detach().numpy(), axis=1) / (
            max_label if max_label > 0 else 1)) * 255
    colors = colors.astype(np.int32)[:, :3]
    vertices_tensor = torch.from_numpy(train_mesh.vertices).unsqueeze(0)
    faces_tensor = torch.from_numpy(train_mesh.faces).unsqueeze(0)
    colors_tensor = torch.from_numpy(colors).unsqueeze(0)
    writer.add_mesh('train_mesh', vertices=vertices_tensor, faces=faces_tensor, colors=colors_tensor,
                    global_step=epoch)
    writer.add_mesh('train_mesh_vertices', vertices=vertices_tensor, colors=colors_tensor,
                    global_step=epoch)


def visual_vertex_label_single_model_IntrA():
    save_path_name = 'single_model_IntrA'
    mesh_path = os.path.join('data/Aneurysm/train', 'AN2_full.obj')
    pred_vertex_label = np.loadtxt(os.path.join('data/Aneurysm/ad', 'AN2-_norm.ad'))[:, -1]
    print(mesh_path)
    mesh = trimesh.load_mesh(mesh_path, process=False)
    mesh: trimesh.Trimesh

    max_label = 3
    colors = plt.get_cmap("Accent")(pred_vertex_label / (max_label - 1))  # the color of 'Accent' from HodgeNet

    mesh.visual.vertex_colors = colors[:, :3]

    if not os.path.exists(os.path.join('visualization_result', save_path_name, 'vertex')):
        os.makedirs(os.path.join('visualization_result', save_path_name, 'vertex'))

    mesh.export(
        os.path.join('visualization_result', save_path_name, 'vertex', mesh_path.split('/')[-1]))



# visualize the ground truth of faces
if __name__ == '__main__':

    save_path_name = 'vis/'
    meshes_path = 'data/humanBody_simplify/test'
    ground_truth_face_path = 'data/humanBody_simplify/seg/'

    for file in sorted(os.listdir(meshes_path)):
        file.strip()
        if os.path.splitext(file)[1] not in ['.ply']:
            continue
        print(os.path.join(meshes_path, file))

        gt_face = torch.from_numpy(
            np.loadtxt(os.path.join(ground_truth_face_path, os.path.splitext(file)[0] + '.eseg')).astype(int)).squeeze()
        mesh_path = os.path.join(meshes_path, file)

        faces_label(mesh_path, gt_face, save_path_name, max_label=8)


