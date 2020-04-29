import torch
import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetDataset(Dataset):
    
    def __init__(self, root, npoints = 1024, split='train', normalize=True, normal_channel=False, cache_size=15000, shuffle=None):
        self.root = root
        # self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.normal_channel = normal_channel
        self.split = split
        self.num_class = 10

        assert self.num_class == 10 or self.num_class == 40
        assert(split=='train' or split=='test')

        class_file = os.path.join(self.root, f'modelnet{self.num_class}_shape_names.txt')
        self.classes = [c.rstrip() for c in open(class_file)]
        self.classes = dict(zip(self.classes, range(len(self.classes))))

        self.datapath = []
        
        name_file = os.path.join(self.root, f'modelnet{self.num_class}_{split}.txt')
        for tmp in open(name_file):
            tmp = tmp.rstrip()
            c = '_'.join(tmp.split('_')[0:-1]) # get class
            self.datapath.append((c,os.path.join(self.root, c, f'{tmp}.txt')))

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        if shuffle is None:
            if split == 'train': self.shuffle = True
            else: self.shuffle = False
        else:
            self.shuffle = shuffle

        # self.reset()

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',', dtype=np.float32)
            # Take the first npoints
            # print(point_set.shape)
            point_set = point_set[0:self.npoints,:]
            if self.normalize:
                point_set[:,0:3] = pc_normalize(point_set[:,0:3])
            if not self.normal_channel:
                point_set = point_set[:,0:3]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
        return point_set, cls

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3


def dataset_stats(dataset):
    classes = dataset.classes
    # classes = dict.fromkeys(classes.keys())
    max_name_len = max([len(s) for s in classes.keys()])

    print('Class\t\tTotal \tTrain \tTest')

    train_file = os.path.join(dataset.root, f'modelnet{dataset.num_class}_train.txt')
    train_lines = [l for l in open(train_file)]
    test_file = os.path.join(dataset.root, f'modelnet{dataset.num_class}_test.txt')
    test_lines = [l for l in open(test_file)]

    tot_train = len(train_lines)
    tot_test = len(test_lines)

    for cl in classes.keys():
        tr_sz = np.sum([cl in l for l in train_lines])
        te_sz = np.sum([cl in l for l in test_lines])
        tot_sz = tr_sz + te_sz
        print(f'{cl.ljust(max_name_len)} \t{tot_sz} \t{tr_sz} \t{te_sz}')

    print(f'Totals: \t{tot_train+tot_test} \t{tot_train} \t{tot_test}')


def voxelize_dataset(dataset, voxel_sizes, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for c in dataset.classes:
        if not os.path.exists(os.path.join(out_path, c)):
            os.mkdir(os.path.join(out_path, c))

    # flist = open(os.path.join(out_path, f'modelnet{dataset.num_class}_{dataset.split}.txt'),'w')

    for i, (xyz, c) in enumerate(dataset):

        if dataset.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        voxes = [torch.zeros((v_size, v_size, v_size), dtype=torch.int32) 
            for v_size in voxel_sizes]

        xyz = ((xyz + 1) / 2)

        for v, v_size in enumerate(voxel_sizes):
            xyz_tmp = xyz * v_size
            xyz_tmp[xyz_tmp >= v_size] = v_size - 1
            xyz_tmp = xyz_tmp.astype(int)

            # print(xyz_tmp)
            
            for x,y,z in xyz_tmp:
                voxes[v][x,y,z] += 1

        # c = torch.tensor(c)

        fn = os.path.basename(dataset.datapath[i][1])
        fn = os.path.splitext(fn)[0]

        print(fn)
        class_name = '_'.join(fn.split('_')[0:-1]) # get class
        torch.save(voxes, os.path.join(out_path, class_name , f'{fn}.pth'))
    #     flist.write(f'{fn}\n')
    # flist.close()

class VoxelNetDataset(Dataset):
    def __init__(self, root, split='train', normal_channel=False, cache_size=15000):
        self.root = root
        # self.batch_size = batch_size
        # self.npoints = npoints
        # self.normalize = normalize
        self.normal_channel = normal_channel
        self.split = split
        self.num_class = 10

        assert self.num_class == 10 or self.num_class == 40
        assert(split=='train' or split=='test')

        class_file = os.path.join(self.root, f'modelnet{self.num_class}_shape_names.txt')
        self.classes = [c.rstrip() for c in open(class_file)]
        self.classes = dict(zip(self.classes, range(len(self.classes))))

        self.datapath = []
        
        name_file = os.path.join(self.root, f'modelnet{self.num_class}_{split}.txt')
        for tmp in open(name_file):
            tmp = tmp.rstrip()
            c = '_'.join(tmp.split('_')[0:-1]) # get class
            self.datapath.append((c,os.path.join(self.root, c, f'{tmp}.pth')))

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        # if shuffle is None:
        #     if split == 'train': self.shuffle = True
        #     else: self.shuffle = False
        # else:
        #     self.shuffle = shuffle

        # self.reset()

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = torch.load(fn[1])#np.loadtxt(fn[1], delimiter=',', dtype=np.float32)
            # Take the first npoints
            # print(point_set.shape)
            # point_set = point_set[0:self.npoints,:]
            # if self.normalize:
            #     point_set[:,0:3] = pc_normalize(point_set[:,0:3])
            # if not self.normal_channel:
            #     point_set = point_set[:,0:3]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
        return point_set, cls

    def __len__(self):
        return len(self.datapath)

    # def num_channel(self):
    #     if self.normal_channel:
    #         return 6
    #     else:
    #         return 3

if __name__ == '__main__':
    modelnet_dir = 'modelnet40_normal_resampled'
    voxelnet_dir = 'modelnet_voxelized'
    md = ModelNetDataset(modelnet_dir, npoints=20000)

    import time
    # tic = time.time()
    # for i in range(10):
    #     ps, cls = md[i]
    # print(time.time() - tic)
    # print(ps.shape, type(ps), cls)

    dataset_stats(md)

    voxelize_dataset(md, [8, 16, 32], voxelnet_dir)
    md = ModelNetDataset(modelnet_dir, split='test', npoints=20000)
    voxelize_dataset(md, [8, 16, 32], voxelnet_dir)

    vd = VoxelNetDataset(voxelnet_dir)

    tic = time.time()
    for i in range(10):
        voxes, cls = vd[i]
    print(time.time() - tic)
    # print(voxes)
    # print(len(ps), type(ps), cls)
