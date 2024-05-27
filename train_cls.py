import os
import argparse
import sys

from tqdm import tqdm
import numpy as np

import torch

import torch.optim as optim

from network.net import Net
from dataset.base_dataset import BaseDataset

import utils.util as utils

# import wandb


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def train(args, net, train_dl, criterion, optimizer):
    net.train()
    optimizer.zero_grad()

    train_loss_avg = utils.RunningAverage()
    train_acc_avg = utils.RunningAverage()

    with tqdm(total=len(train_dl), desc="Training") as tq:

        for batch_idx, data in enumerate(train_dl):
            # load train data
            verts, faces, verts_normals, evals, evecs, verts_dihedralAngles, hks, mass, labels, mesh_path = data

            # Move to args.device
            verts = verts.to(args.device)
            faces = faces.to(args.device)
            verts_normals = verts_normals.to(args.device)
            evals = evals.to(args.device)
            evecs = evecs.to(args.device)
            verts_dihedralAngles = verts_dihedralAngles.to(args.device)
            hks = hks.to(args.device)
            mass = mass.to(args.device)

            labels = labels.to(args.device)
            if not args.is_pointCloud:
                labels = labels.unsqueeze(0)

            #  the verts_normals of Cubes has nan values
            if args.dataset_name == 'cubes':
                verts_normals[torch.argwhere(torch.isnan(verts_normals))] = 0.

            # normalization
            if args.normal_rest_data:
                if not args.is_pointCloud:
                    verts_dihedralAngles = utils.normalization(verts_dihedralAngles)
                    hks = utils.normalization(hks)
                else:
                    hks = utils.normalization(hks)

            # Randomly rotate positions
            if args.augment_data:
                verts, verts_normals = utils.random_rotate_points_axis_x(verts, verts_normals)

            if not args.is_pointCloud:
                features = torch.cat((verts, verts_normals), dim=-1)
            else:
                features = torch.cat((verts, verts_normals, hks), dim=-1)

            preds = net(x_in=features, mass=mass, faces=faces)
            preds = preds.unsqueeze(0)



            loss = criterion(preds, labels)
            loss /= args.batch_size
            loss.backward()
            train_loss_avg.update(loss.detach().cpu().item())

            # track accuracy
            preds = torch.log_softmax(preds, dim=-1)
            pred_labels = torch.max(preds, dim=1).indices

            correct_label = pred_labels.eq(labels).sum().item()
            train_acc_avg.update(correct_label)

            # Step the optimizer
            if (batch_idx + 1) % args.batch_size == 0 or (batch_idx + 1 == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()
            tq.set_postfix(loss='{:05.4f}'.format(train_loss_avg()))
            tq.update()

        train_acc = train_acc_avg()
        loss = train_loss_avg()

    return train_acc, loss


def test(args, net, test_dl, criterion):
    net.eval()

    test_loss_avg = utils.RunningAverage()
    test_acc_avg = utils.RunningAverage()

    with torch.no_grad():

        for data in tqdm(test_dl):
            # load test data
            verts, faces, verts_normals, evals, evecs, verts_dihedralAngles, hks, mass, labels, mesh_path = data

            # Move to args.device
            verts = verts.to(args.device)
            faces = faces.to(args.device)
            verts_normals = verts_normals.to(args.device)
            evals = evals.to(args.device)
            evecs = evecs.to(args.device)
            verts_dihedralAngles = verts_dihedralAngles.to(args.device)
            hks = hks.to(args.device)
            mass = mass.to(args.device)

            labels = labels.to(args.device)
            if not args.is_pointCloud:
                labels = labels.unsqueeze(0)

            #  the verts_normals of Cubes has nan values
            if args.dataset_name == 'cubes':
                verts_normals[torch.argwhere(torch.isnan(verts_normals))] = 0.

            # normalization
            if args.normal_rest_data:
                if not args.is_pointCloud:
                    verts_dihedralAngles = utils.normalization(verts_dihedralAngles)
                    hks = utils.normalization(hks)
                else:
                    hks = utils.normalization(hks)

            if not args.is_pointCloud:
                features = torch.cat((verts, verts_normals), dim=-1)
            else:
                features = torch.cat((verts, verts_normals, hks), dim=-1)

            preds = net(x_in=features, mass=mass, faces=faces)
            preds = preds.unsqueeze(0)

            loss = criterion(preds, labels)
            test_loss_avg.update(loss.detach().cpu().item())

            # track accuracy
            preds = torch.log_softmax(preds, dim=-1)
            pred_labels = torch.max(preds, dim=1).indices

            correct_faces = pred_labels.eq(labels).sum().item()
            test_acc_avg.update(correct_faces)

        test_acc = test_acc_avg()
        loss = test_loss_avg()

    return test_acc, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--dataset_name', type=str, default='modelnet40_ply_hdf5_2048')
    parser.add_argument('--experiment_name', type=str, default='base')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_cache', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=40938661)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--scheduler_mode', choices=['CosWarm', 'MultiStep'], default='CosWarm')
    parser.add_argument('--scheduler_T0', type=int, default=30)
    parser.add_argument('--scheduler_eta_min', type=float, default=3e-7)
    parser.add_argument('--norm_selection', choices=['center_unit', 'norm_unit', 'norm_mesh'], default='center_unit')
    parser.add_argument('--norm_method', choices=['mean', 'bbox'], default='mean')
    parser.add_argument('--norm_scale_method', choices=['max_rad', 'area'], default='max_rad')
    parser.add_argument('--normal_rest_data', type=bool, default=True)
    parser.add_argument('--hks_count', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--loss_rate', type=float, default=5e-3)
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--drop_prob', type=float, default=0.1, help='the drop probability')
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--feature_dim', type=int, default=6)
    parser.add_argument('--split_size', type=int, default=16)
    parser.add_argument('--modelnet40_num_points', type=int, default=2048)
    parser.add_argument('--n_neighbors_cloud', type=int, default=30)
    parser.add_argument('--augment_data', action='store_true')
    parser.add_argument('--use_adj_loss', action='store_true')
    parser.add_argument('--is_pointCloud', action='store_true')
    parser.add_argument('--k_eig_list', nargs='+', default=[2047, 128, 32], type=int, help='Multi-resolution input')

    args = parser.parse_args()

    utils.same_seed(args.seed)

    print('Load Dataset...')
    base_dataset = BaseDataset(args)
    if args.mode == 'train':
        train_dl, test_dl = base_dataset.load_dataset()
    else:
        test_dl = base_dataset.load_dataset()

    # define the Net
    net = Net(C_in=args.feature_dim, C_out=args.num_classes, drop_path_rate=args.drop_prob,
              k_eig_list=args.k_eig_list, outputs_at='global_mean').to(args.device)

    # define the segmentation_loss
    criterion = utils.segmentation_loss(smoothing=args.smoothing)

    # save the checkpoints
    checkpoints_save_path = os.path.join('data', args.dataset_name, 'checkpoints', args.experiment_name)
    utils.ensure_folder_exists(checkpoints_save_path)

    max_val_acc = -np.inf

    if args.mode == 'train':

        # Use wandb to visualize the training process
        # wandb.init(project='lapNN_' + args.dataset_name, entity='lap2mesh', config=args, name=args.experiment_name)
        # wandb.watch(net, log="all", log_graph=False)

        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

        # select scheduler mode
        # if args.scheduler_mode == 'CosWarm':
        #     scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=2,
        #                                                          eta_min=args.scheduler_eta_min, verbose=True)
        # else:
        #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 200, 500], gamma=0.5, verbose=True)

        for epoch in range(args.epochs):
            train_acc, train_loss = train(args, net, train_dl, criterion, optimizer)
            test_acc, test_loss = test(args, net, test_dl, criterion)

            # scheduler.step()
            # this_scheduler = scheduler.get_last_lr()[0]
            this_scheduler = args.lr

            # save best acc
            if max_val_acc < test_acc:
                max_val_acc = test_acc
                utils.save_logging(args.mode, test_acc, test_loss, checkpoints_save_path, model=net,
                                   train_acc=train_acc,
                                   train_loss=train_loss, epoch=epoch)

            print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%  BestEval: {:06.3f}%".format(epoch,
                                                                                                             100 * train_acc,
                                                                                                             100 * test_acc,
                                                                                                             100 * max_val_acc))

            # wandb.log(
            #     {'Accuracy/test_best_acc': max_val_acc, 'Accuracy/test_acc': test_acc, 'Accuracy/train_acc': train_acc,
            #      'Loss/test_loss': test_loss, 'Loss/train_loss': train_loss, 'Utils/lr_scheduler': this_scheduler})
            print(
                {'Accuracy/test_best_acc': max_val_acc, 'Accuracy/test_acc': test_acc, 'Accuracy/train_acc': train_acc,
                 'Loss/test_loss': test_loss, 'Loss/train_loss': train_loss, 'Utils/lr_scheduler': this_scheduler})
    else:
        # save the visualization result of faces
        vis_face_save_path = os.path.join('data', args.dataset_name, 'checkpoints', args.experiment_name, 'vis_face')
        utils.ensure_folder_exists(vis_face_save_path)
        net.load_state_dict(torch.load(os.path.join(checkpoints_save_path, 'best.pth')))
        test_acc, test_loss, test_smoothCE_loss, test_adjacency_loss = test(args, net, test_dl, criterion,
                                                                            vis_face_save_path)
        utils.save_logging(args.mode, test_acc, test_loss, checkpoints_save_path)
        print("Test overall: {:06.3f}%".format(100 * test_acc))
