import torch.nn as nn
from torch import Tensor


# モデルの定義
class FullConnectedModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        dim1 = int(input_dim / 2)
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        dim2 = int(dim1 / 2)
        self.fc2 = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        dim3 = int(dim2 / 2)
        self.fc3 = nn.Sequential(
            nn.Linear(dim2, dim3),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.output = nn.Sequential(
            nn.Linear(dim3, output_dim),
            nn.Softmax(dim=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.output(x)
