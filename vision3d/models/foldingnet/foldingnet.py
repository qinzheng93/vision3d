import torch
import torch.nn as nn


class FoldingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, steps):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.grid_size = self.steps * self.steps

        self.grid_to_points = nn.Sequential(
            nn.Conv1d(input_dim + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

        self.refine_points = nn.Sequential(
            nn.Conv1d(input_dim + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

        self.seeds = None

    def generate_seeds(self):
        steps = torch.linspace(-1.0, 1.0, steps=self.steps, dtype=torch.float32)
        rows = steps.unsqueeze(1).repeat(1, self.steps).view(1, -1)  # (S,) -> (S, 1) -> (S, S) -> (1, F)
        cols = steps.unsqueeze(0).repeat(self.steps, 1).view(1, -1)  # (S,) -> (1, S) -> (S, S) -> (1, F)
        folding_seeds = torch.cat([rows, cols], dim=0).cuda()  # (1, F) + (1, F) -> (2, F)
        return folding_seeds

    def forward(self, global_feats):
        """FoldingNet forward.

        Args:
            global_feats (Tensor): (B, C)

        Returns:
            points (Tensor): (B, 3, F)
        """
        if self.seeds is None:
            self.seeds = self.generate_seeds()  # (2, F)

        batch_size, input_dim = global_feats.shape
        feats = global_feats.unsqueeze(2).expand(-1, -1, self.grid_size)  # (B, C, F)
        seeds = self.seeds.unsqueeze(0).expand(batch_size, -1, -1)  # (2, F) -> (B, 2, F)

        seed_feats = torch.cat([seeds, feats], dim=1)  # (B, 2, F) + (B, C, F) -> (B, C+2, F)
        points = self.grid_to_points(seed_feats)  # (B, C+2, F) -> (B, 3, F)

        point_feats = torch.cat([points, feats], dim=1)  # (B, 3, F) + (B, C, F) -> (B, C+3, F)
        points = self.refine_points(point_feats)  # (B, C+3, F) -> (B, 3, F)

        return points
