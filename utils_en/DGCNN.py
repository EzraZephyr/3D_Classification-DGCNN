import torch
import torch.nn.functional as F
from torch import nn
from EdgeConvfeature import graph_feature


class DGCNN(nn.Module):
    def __init__(self, ndf, k, num_classes):
        """
        Parameters:
            ndf: The base number of convolution channels, making it easier to adjust the network's channel size in the future.
            k: The number of k-nearest neighbors for each point.
            num_classes: The number of categories in the dataset.

        Explanation:
            In the EdgeConv operation, we add the difference feature between points and their neighbors to the original feature.
            This allows the model to preserve the original features of each point while learning local geometric information.
            Also, because we need to dynamically build the neighbor graph at each layer, the number of channels passed to each layer will always be twice the number of channels of the previous layer.
            Normally, the initial feature channel size is x, y, z, but after construction, the initial feature becomes 6-dimensional.
            A simple bottleneck structure is used to improve the performance of the network as much as possible while maintaining the original structure of the official code.
            Finally, the network output is mapped to the number of target categories through fully connected layers.
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
        Parameters:
            x: Point cloud feature matrix with shape (batch_size, num_points, num_dims)

        Explanation:
            Pass the feature matrix x into the graph_feature function to calculate the feature differences between each point and its k-nearest neighbors and construct the adjacency graph.
            Then, pass the graph through each convolution layer.
            After each convolution, use max pooling to select the most representative feature from each point's neighbors.
            In the final convolution layer, both max pooling and average pooling are used to further integrate the features, so that it can preserve both the most representative local features and smooth global features.
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
