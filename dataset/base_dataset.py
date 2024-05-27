from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset.humanBody import segHumanBody_simplify, segHumanBody_origin
from dataset.coseg import CosegDataset
from dataset.IntrA import AneurysmSegDataset
from dataset.shrec11 import Shrec11MeshDataset_Simplify, Shrec11MeshDataset_Original
from dataset.manifold40 import Manifold40ClsDataset


class BaseDataset(Dataset):
    def __init__(self, args):
        super(BaseDataset, self).__init__()

        self.args = args
        self.train_ds = None
        self.test_ds = None
        self.train_dl = None
        self.test_dl = None
        self.weight_train = None
        self.weight_test = None

    def load_dataset(self):
        if self.args.dataset_name == 'humanBody_simplify' or self.args.dataset_name == 'humanbody_sinplify_vertices':
            if self.args.mode == 'train':
                print('---loading training dataset---')
                self.train_ds = segHumanBody_simplify(self.args, is_train=True)
            print('---loading test dataset---')
            self.test_ds = segHumanBody_simplify(self.args, is_train=False)

        elif self.args.dataset_name == 'human_seg_origin':
            if self.args.mode == 'train':
                print('---loading training dataset---')
                self.train_ds = segHumanBody_origin(self.args, is_train=True)
            print('---loading test dataset---')
            self.test_ds = segHumanBody_origin(self.args, is_train=False)

        elif self.args.dataset_name in ['coseg_aliens', 'coseg_chairs', 'coseg_vases']:
            if self.args.mode == 'train':
                print('---loading training dataset---')
                self.train_ds = CosegDataset(self.args, is_train=True)
            print('---loading test dataset---')
            self.test_ds = CosegDataset(self.args, is_train=False)

        elif self.args.dataset_name == 'IntrA':
            if self.args.mode == 'train':
                print('---loading training dataset---')
                self.train_ds = AneurysmSegDataset(self.args, is_train=True)
            print('---loading test dataset---')
            self.test_ds = AneurysmSegDataset(self.args, is_train=False)

        elif self.args.dataset_name == 'shrec11_split16':
            if self.args.mode == 'train':
                print('---loading training dataset---')
                self.train_ds = Shrec11MeshDataset_Simplify(self.args, is_train=True)
            print('---loading test dataset---')
            self.test_ds = Shrec11MeshDataset_Simplify(self.args, is_train=False, exclude_dict=self.train_ds.entries)

        elif self.args.dataset_name == 'SHREC11_origin':
            if self.args.mode == 'train':
                print('---loading training dataset---')
                self.train_ds = Shrec11MeshDataset_Original(self.args, is_train=True)
            print('---loading test dataset---')
            self.test_ds = Shrec11MeshDataset_Original(self.args, is_train=False, exclude_dict=self.train_ds.entries)

        elif self.args.dataset_name == 'Manifold40':
            if self.args.mode == 'train':
                print('---loading training dataset---')
                self.train_ds = Manifold40ClsDataset(self.args, is_train=True)
            print('---loading test dataset---')
            self.test_ds = Manifold40ClsDataset(self.args, is_train=False)

        if self.args.mode == 'train':
            self.train_dl = DataLoader(self.train_ds, batch_size=None, shuffle=True, pin_memory=False)

        self.test_dl = DataLoader(self.test_ds, batch_size=None, shuffle=False, pin_memory=False)

        if self.args.mode == 'train':
            return self.train_dl, self.test_dl
        else:
            return self.test_dl