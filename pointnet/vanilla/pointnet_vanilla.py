import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

# Spatial Transform Net 3d
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0] # 只返回最大值的value
        x = x.view(-1, 1024) # n*1024

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(3).flatten().astype(np.float32))).view(1,9).repeat(batchsize,1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


# feature extraction net 64->128->1024
class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        trans_matrix3d = self.stn(x)
        x = x.transpose(2,1)
        # Input transform, bmm performs a batch matrix-matrix product of matrices
        x = torch.bmm(x, trans_matrix3d)
        x = x.transpose(2,1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x,2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x, trans_matrix3d


class PointNetCls(nn.Module):
    def __init__(self, k=40):
        super(PointNetCls, self).__init__()
        self.feature = FeatNet()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k) # k = num_classes
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans_matrix3d = self.feature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), trans_matrix3d

if __name__ == '__main__':
    sim_data = torch.rand(32, 3, 2500)
    trans_matrix3d = STN3d()
    out = trans_matrix3d(sim_data)
    print("input transform matrix size ", out.size())

    pt_feat = FeatNet()
    out, _ = pt_feat(sim_data)
    print('global feature size: ', out.size())

    cls = PointNetCls()
    out, _ = cls(sim_data)
    print('class: ', out.size())