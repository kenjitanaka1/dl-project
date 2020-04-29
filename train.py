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

parser = argparse.ArgumentParser(description='Train a neural network to classify point cloud data')
parser.add_argument('--root_dir', metavar='R', dest='root_dir', type=str, default='modelnet_voxelized',
                    help='Root directory for the dataset')
parser.add_argument('--seed', metavar='S', dest='seed', default=2020,
                    help='Pseudorandom seed')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate')
# parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay')
parser.add_argument('--momentum', metavar='M', type=float, default=0.9, help='Initial learning rate')
parser.add_argument('--optimizer', metavar='O', type=str, default='adam', help='Optimizer (default: adam)')
parser.add_argument('--batch_size', metavar='B', type=int, default=16, help='Batch Size during training')
parser.add_argument('--num_point', type=int, default=1024, help='Max number of points')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

EPOCHS = args.epochs
MOMENTUM = args.momentum
BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
LEARNING_RATE = args.lr
LR_DECAY_RATE = args.decay_rate
# LR_DECAY_STEP = args.decay_step
LOSS_CRITERION = nn.CrossEntropyLoss()
VOXEL_SIZE = [8,16,32]

TRAIN_DATASET = VoxelNetDataset(args.root_dir, cache_size=4000)
# TEST_DATASET = VoxelNetDataset(args.root_dir, split='test')
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# TEST_DATALOADER = DataLoader(TEST_DATASET, shuffle=False, num_workers=4)

MODEL = MyModel(len(TRAIN_DATASET.classes),voxel_size=VOXEL_SIZE, normal_channel=TRAIN_DATASET.normal_channel)
MODEL_SAVE_DIR = 'saved_models'

if args.optimizer == 'adam':
    OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
elif args.optimizer == 'adadelta':
    OPTIMIZER = optim.Adadelta(MODEL.parameters())
elif args.optimizer == 'sgd':
    OPTIMIZER = optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)
else:
    raise NotImplementedError('Unknown optimizer given')

LR_SCHEDULER = torch.optim.lr_scheduler.ExponentialLR(optimizer=OPTIMIZER, gamma=LR_DECAY_RATE)

def train(model, optimizer, lr_scheduler, criterion, dataloader, save_dir, e0, epochs):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mean_correct = []
    mean_loss = []

    for epoch in range(e0, e0+epochs):
        print(f'**** EPOCH {epoch+1}/{e0+epochs} ****')

        for batch_id, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), smoothing=0.9):
            points, target = data

            # print(type(points[0]))
            # print(len(points))
            # print(len(points[0]))
            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            # points = torch.Tensor(points)
            target = target[:, 0]

            # points = points.transpose(2, 1)
            optimizer.zero_grad()

            model = model.train()
            pred, trans_feat = model(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            # print(pred_choice.shape)
            # print(pred_choice.eq(target.long().data).shape)
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(len(target)))
            mean_loss.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        epoch_acc = np.mean(mean_correct[-len(dataloader):])
        epoch_loss = np.mean(mean_loss[-len(dataloader):])
        print(f'Epoch Training Accuracy: {epoch_acc} \t Loss {epoch_loss}')

        np.save(os.path.join(save_dir, f'loss_epoch_{epoch}.npy'), (epoch_acc, epoch_loss))
        torch.save(model.state_dict(), os.path.join(save_dir,f'model_epoch_{epoch}.dict'))

if __name__=='__main__':
    E0 = 0

    if os.path.exists(MODEL_SAVE_DIR):
        while os.path.exists(os.path.join(MODEL_SAVE_DIR, f'model_epoch_{E0}.dict')):
            E0 += 1
        print(E0)
        if E0 > 0:
            MODEL.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, f'model_epoch_{E0-1}.dict')))
    

    print(f'Begin training of {MODEL.__class__.__name__}')
    train(MODEL, OPTIMIZER, LR_SCHEDULER, LOSS_CRITERION, TRAIN_DATALOADER, MODEL_SAVE_DIR, E0, EPOCHS)
    # loss criterion is cross entropy 