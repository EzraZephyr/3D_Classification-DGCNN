import torch
from torch_geometric.nn import knn_graph


def graph_feature(x, k):
    """
    Parameters:
        x: Point cloud feature matrix, shape (batch_size, num_points, num_dims)
        k: The number of k-nearest neighbors for each point

    Explanation:
        Create a batch-wise tensor corresponding to each point in x,
        so that the knn_graph knows which batch each point belongs to.
        Then, calculate the difference between the source and target points,
        concatenate with the original features, and reshape the result
        into a format that can be accepted by the convolutional layers.
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
