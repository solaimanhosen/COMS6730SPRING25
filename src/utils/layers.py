import torch
import torch.nn as nn


class GConv(nn.Module):

    def __init__(self, in_dim, out_dim):
        # in_dim is the number of features of each node
        # out_dim is the number of output features
        super(GConv, self).__init__()
        # #### YOUR CODE HERE

        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W)

        # #### YOUR CODE HERE

    def forward(self, A, X):
        # A is the normalized adjacency matrix
        # X is the feature matrix
        # #### YOUR CODE HERE

        X = torch.mm(A, X)    # Aggregate neighbor features
        X = torch.mm(X, self.W)  # Apply linear transformation
        X = torch.relu(X)      # Apply ReLU activation

        return X

        # #### YOUR CODE HERE


class GAttn(nn.Module):

    def __init__(self, in_dim, out_dim):
        # in_dim is the number of features of each node
        # out_dim is the number of output features
        super(GAttn, self).__init__()
        # #### YOUR CODE HERE

        # #### YOUR CODE HERE

    def forward(self, A, X):
        # A is the normalized adjacency matrix
        # X is the feature matrix
        # #### YOUR CODE HERE

        return X
        # #### YOUR CODE HERE


class GPool(nn.Module):

    def __init__(self, k, in_dim):
        # k is the percentage of output nodes with respect to the number of input nodes
        # in_dim is the number of features of each node
        super(GPool, self).__init__()
        # #### YOUR CODE HERE

        # #### YOUR CODE HERE

    def forward(self, A, X):
        # A is the normalized adjacency matrix
        # X is the feature matrix
        # #### YOUR CODE HERE
        return A, X
        # #### YOUR CODE HERE


class GDiffPool(nn.Module):

    def __init__(self, k, in_dim):
        # k is the number of output nodes
        # in_dim is the number of features of each node
        super(GDiffPool, self).__init__()
        # #### YOUR CODE HERE

        # #### YOUR CODE HERE

    def forward(self, A, X):
        # A is the normalized adjacency matrix
        # X is the feature matrix
        # #### YOUR CODE HERE
        return A, X
        # #### YOUR CODE HERE


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g
