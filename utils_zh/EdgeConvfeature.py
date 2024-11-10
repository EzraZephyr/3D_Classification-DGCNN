import torch
from torch_geometric.nn import knn_graph


def graph_feature(x, k):
    """
    参数：
        x: 点云特征矩阵 形状为(batch_size, num_points, num_dims)
        k: 每个点的k近邻数

    说明：
        使用batch创建一个与x中每个点一一对应的批次信息张量 使得knn_graph知道每个点所属的批次是什么
        随后计算原点和目标点之间的差异 同时在拼接上原始特征 再调整成可以被卷积层接受的维度
    """
    batch_size, num_points, num_dims = x.size()

    x = x.contiguous()
    x = x.view(batch_size * num_points, num_dims)

    batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_points)
    edge_index = knn_graph(x, k=k, batch=batch, loop=False)

    src = edge_index[0]
    idx = edge_index[1]

    x_i = x[src, :]
    x_j = x[idx, :]

    feature = torch.cat((x_i - x_j, x_j), dim=1)
    feature = feature.view(batch_size, num_points, k, -1)
    feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature
