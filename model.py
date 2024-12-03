import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(8, 16)  # Первый слой GCN с 8 входными и 16 выходными признаками
        self.conv2 = GCNConv(16, 32)  # Второй слой GCN с 16 входными и 32 выходными признаками
        self.fc1 = torch.nn.Linear(32, 64)  # Полносвязный слой после второго GCN, исправлена размерность
        self.fc2 = torch.nn.Linear(64, 1)  # Выходной слой

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Применение первого GCN слоя
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Функция активации ReLU

        # Применение второго GCN слоя
        x = self.conv2(x, edge_index)
        x = F.relu(x)  # Функция активации ReLU

        # Для прогнозирования атрибутов рёбер, используем сумму признаков для каждой пары рёбер
        row, col = edge_index
        edge_attr_pred = self.fc1(x[row] + x[col])  # Суммируем признаки узлов, соединённых рёбром
        edge_attr_pred = F.relu(edge_attr_pred)  # Активация
        edge_attr_pred = self.fc2(edge_attr_pred)  # Выходной слой
        return edge_attr_pred
