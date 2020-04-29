import os
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# import provider

from dataset import VoxelNetDataset
from model import MyModel

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Test a neural network classifying point data')
parser.add_argument('--root_dir', metavar='R', dest='root_dir', type=str, default='modelnet_voxelized',
                    help='Root directory for the dataset')
parser.add_argument('--model_dir', metavar='M', dest='model_dir', type=str, default='saved_models',
                    help='Root directory for the saved models to test')
parser.add_argument('--seed', metavar='S', dest='seed', default=2020,
                    help='Pseudorandom seed')
# parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate')
# # parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay')
# parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay')
# parser.add_argument('--momentum', metavar='M', type=float, default=0.9, help='Initial learning rate')
# parser.add_argument('--optimizer', metavar='O', type=str, default='adam', help='Optimizer (default: adam)')
# parser.add_argument('--batch_size', metavar='B', type=int, default=16, help='Batch Size during training')
# parser.add_argument('--num_point', type=int, default=1024, help='Max number of points')
parser.add_argument('-all', action='store_true', help='Test all epochs')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# NUM_POINT = args.num_point
VOXEL_SIZE = [8,16,32]

# TRAIN_DATASET = ModelNetDataset(args.root_dir, npoints=NUM_POINT)
TEST_DATASET = VoxelNetDataset(args.root_dir, split='test')
# TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
TEST_DATALOADER = DataLoader(TEST_DATASET, shuffle=False, num_workers=4)

MODEL = MyModel(len(TEST_DATASET.classes),voxel_size=VOXEL_SIZE, normal_channel=TEST_DATASET.normal_channel)
MODEL_SAVE_DIR = args.model_dir

def test(model, dataloader, save_dir):
    num_class = dataloader.dataset.num_class
    correct = 0
    tot = 0
    class_acc = np.zeros((num_class,3))
    confusion = np.zeros((num_class, num_class))
    for j, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        points, target = data
        target = target[:, 0]
        # points = points.transpose(2, 1)
        model = model.eval()
        pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]
        # print(target)

        for choice, truth in zip(pred_choice, target):
            if choice.eq(truth).data:
                correct += 1
                class_acc[truth.data,0] += 1
            class_acc[truth.data,1] += 1
            confusion[choice.data, truth.data] += 1
        tot += len(target)

    class_acc[:,2] = class_acc[:,0] / class_acc[:,1]
    
    acc = correct / tot
    print(f'Accuracy: {acc * 100}%')
    print(f'Class Accuracy:')
    print(class_acc)
    print('Confusion Matrix:')
    print(confusion)

ALL = args.all

if __name__ == '__main__':
    print(f'Begin testing of {MODEL.__class__.__name__}')

    if ALL:
        i = 0
        model_path = os.path.join(MODEL_SAVE_DIR, f'model_epoch_{i}.dict')
        while os.path.exists(model_path):
            MODEL.load_state_dict(torch.load(model_path))

            test(MODEL, TEST_DATALOADER, MODEL_SAVE_DIR)
            i += 1
            model_path = os.path.join(MODEL_SAVE_DIR, f'model_epoch_{i}.dict')

    else:
        i = 0
        model_path = os.path.join(MODEL_SAVE_DIR, f'model_epoch_{i}.dict')
        while os.path.exists(model_path):
            i += 1
            model_path = os.path.join(MODEL_SAVE_DIR, f'model_epoch_{i}.dict')
        model_path = os.path.join(MODEL_SAVE_DIR, f'model_epoch_{i-1}.dict')
        test(MODEL, TEST_DATALOADER, MODEL_SAVE_DIR)