import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self,num_class, voxel_size = [4, 16, 32], normal_channel=False):
        super(MyModel, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.voxel_size = voxel_size

        self.drop = nn.Dropout(p=0.8)
        self.drop2 = nn.Dropout(p=0.5)

        # self.fc1 = nn.Linear(1024 * 3, num_class)

        self.conv1 = [nn.Conv3d(1, 8, 2) for v in voxel_size]
        self.conv2 = [nn.Conv3d(8, 16, 2) for v in voxel_size]
        self.pool1 = nn.MaxPool3d(2,2)
        # self.conv3 = [nn.Conv3d(6, 6, 4) for v in voxel_size]
        # self.conv4 = nn.Conv3d(8, 8, 4)
        # self.pool2 = nn.MaxPool3d(2,2)
        # todo (improvement) figure out how to calculate these sizes autmatically
        self.fc1 = [nn.Linear(432, 1024), nn.Linear(5488, 1024), nn.Linear(54000, 1024)] 
        self.concat = lambda x : torch.cat(x, 1)
        self.fc2 = nn.Linear(len(voxel_size) * 1024, 128)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, x):
        B = x[0].shape[0]

        for i in range(len(x)):
            v = self.voxel_size[i]
            x[i] = F.relu(self.conv1[i](x[i].unsqueeze(1).float()))
            x[i] = self.drop(x[i])
            x[i] = F.relu(self.conv2[i](x[i]))
            x[i] = self.pool1(x[i])
            x[i] = self.drop2(x[i])
            # x[i] = F.relu(self.conv3[i](x[i]))
        #     x[i] = F.relu(self.conv4(x[i]))
            # x[i] = self.pool2(x[i])
            # print(type(x))
            # x[i] = x[i].view(*(B,-1))
            # print(x[i].shape)
            x[i] = F.relu(self.fc1[i](x[i].view(B,-1)))

        x = self.concat(x)
        # print(x.shape)
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)


            
        # B, _, _ = xyz.shape
        # if self.normal_channel:
        #     norm = xyz[:, 3:, :]
        #     xyz = xyz[:, :3, :]
        # else:
        #     norm = None

        # x = self.fc1(xyz.reshape(-1, 1024 * 3))
        # l1_xyz, l1_points = self.sa1(xyz, norm)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x, None#, l3_points
