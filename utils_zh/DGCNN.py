import torch
import torch.nn.functional as F
from torch import nn
from EdgeConvfeature import graph_feature


class DGCNN(nn.Module):
    def __init__(self, ndf, k, num_classes):
        """
        参数：
            ndf: 基础的卷积通道数 方便以后调整网络的通道数
            k: 每个点的k近邻数
            num_classes: 数据集的类别数量

        说明：
            因为在EdgeConv操作中 我们将点和点之间的差异特征与原始特征进行相加
            使得模型在学习局部几何信息的同时能保留每个点的原始特征
            并且因为要每一层都进行一次动态邻居图的构建 所以传入的通道数会一直比前一层的通道数高一倍
            正常传入的初始特征通道数是x,y,z 但在经过构建之后 传入的初始特征就会变成6维
            并使用了一个简单的瓶颈结构在保持官方代码的结构的同时 尽可能地提高网络的性能
            并且最后通过全连接层将网络的输出映射到目标类别的数量上
        """
        super(DGCNN, self).__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, ndf, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ELU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*2, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.ELU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*4, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.ELU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf*8, ndf*16, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.ELU(),
        )

        self.linear1 = nn.Sequential(
            nn.Linear(ndf*32, ndf*8, bias=False),
            nn.BatchNorm1d(ndf*8),
            nn.ELU(),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(ndf*8, ndf*4, bias=False),
            nn.BatchNorm1d(ndf*4),
            nn.ELU(),
        )

        self.linear3 = nn.Linear(ndf*4, num_classes)

    def forward(self, x):
        """
        参数：
            x: 点云特征矩阵 形状为(batch_size, num_points, num_dims)

        说明：
            将特征矩阵x传入graph_feature函数来计算每个点与其k个邻居之间的特征差异并构建邻接图
            随后传入每一层的卷积进行运算
            并在每次卷积之后 使用最大池化选取每个点的邻居中最具代表性的特征
            在卷积的最后一层同时使用最大池化和平均池化来进一步整合 使其能同时保留最具代表性的局部特征和平滑的全局特征
        """
        batch_size = x.size(0)

        x = graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1)[0]
        x1 = x1.permute(0, 2, 1)

        x = graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1)[0]
        x2 = x2.permute(0, 2, 1)

        x = graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1)[0]
        x3 = x3.permute(0, 2, 1)

        x = graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1)[0]
        x4 = x4.permute(0, 2, 1)

        x = torch.cat((x1, x2, x3, x4), dim=2)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv5(x)
        x = x.squeeze(-1)

        x5 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x6 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x5, x6), dim=1)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x




