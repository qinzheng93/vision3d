import torch
import torch.nn as nn

from vision3d.loss import SmoothCrossEntropyLoss


class TNet(nn.Module):
    def __init__(self, input_dim, local_dims, global_dims):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        for i, output_dim in enumerate(local_dims):
            layers.append(nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.local_mlp = nn.Sequential(*layers)

        layers = []
        for i, output_dim in enumerate(global_dims):
            layers.append(nn.Linear(input_dim, output_dim, bias=False))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.global_mlp = nn.Sequential(*layers)

        self.weight = nn.Parameter(torch.zeros(global_dims[-1], input_dim * input_dim))
        self.bias = nn.Parameter(torch.eye(input_dim).view(-1))

    def forward(self, points):
        batch_size = points.shape[0]
        points = self.local_mlp(points)
        points, _ = points.max(dim=2)
        points = self.global_mlp(points)
        points = torch.matmul(points, self.weight) + self.bias
        points = points.view(batch_size, self.input_dim, self.input_dim)
        return points


class TNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(transforms):
        assert (
            transforms.dim() == 3 and transforms.shape[1] == transforms.shape[2]
        ), f"The transform matrix must be 3x3 matrices ({transforms.shape} found)."
        identity = torch.eye(transforms.shape[1]).to(transforms.device)
        transforms = identity - torch.matmul(transforms, transforms.transpose(1, 2))
        loss = torch.sum(transforms ** 2) / 2
        return loss


class PointNetLoss(nn.Module):
    def __init__(self, alpha=0.001, eps=None):
        super().__init__()
        self.tnet_loss = TNetLoss()
        if eps is None:
            self.cls_loss = nn.CrossEntropyLoss()
        else:
            self.cls_loss = SmoothCrossEntropyLoss(eps=eps)
        self.alpha = alpha

    def forward(self, outputs, labels, transforms):
        cls_loss = self.cls_loss(outputs, labels)
        tnet_loss = self.alpha * self.tnet_loss(transforms)
        loss = cls_loss + tnet_loss
        return loss, cls_loss, tnet_loss
