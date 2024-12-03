from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import numpy as np
import folium
import os
import requests
import json
import threading
import time

# Установка устройства: используем GPU, если он доступен, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Определение модели
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(8, 16, heads=2, dropout=0.2)
        self.conv2 = GATConv(16 * 2, 32, heads=2, dropout=0.2)
        self.fc1 = torch.nn.Linear(32 * 2, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        row, col = edge_index
        edge_attr_pred = self.fc1(x[row] + x[col])
        edge_attr_pred = F.relu(edge_attr_pred)
        edge_attr_pred = self.fc2(edge_attr_pred)
        return edge_attr_pred

# Инициализация модели и оптимизатора
model = GAT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Создание Flask приложения
app = Flask(__name__)
CORS(app)  # Добавляем поддержку CORS для всего приложения

# Маршрут для корневой страницы
@app.route('/')
def index():
    return "Flask server is up and running!"

# Определение маршрута для обслуживания статических файлов карты
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Определение маршрута API для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из запроса
        data = request.get_json()

        # Проверка на наличие начальной и конечной точки
        if 'start_point' not in data or 'end_point' not in data:
            return jsonify({'error': 'Missing required fields: start_point or end_point'}), 400

        start_point = data['start_point']
        end_point = data['end_point']

        # Логика создания графа и расчета маршрута на основе координат
        num_nodes = 20  # Временное значение для демонстрации
        num_edges = 40  # Временное значение для демонстрации

        # Создание случайных рёбер и весов
        edges = torch.tensor(np.random.choice(num_nodes, (num_edges, 2)), dtype=torch.long).t().contiguous().to(device)
        weights = torch.tensor(np.random.rand(num_edges) * 10, dtype=torch.float).to(device)  # Положительные веса
        x = torch.randn(num_nodes, 8).to(device)

        # Нормализация весов рёбер
        weights = (weights - weights.mean()) / weights.std()
        weights = torch.abs(weights)  # Убедимся, что все веса положительные

        # Создание графа
        graph_data = Data(x=x, edge_index=edges, edge_attr=weights).to(device)

        # Оценка модели
        model.eval()
        with torch.no_grad():
            predicted_weights = model(graph_data).view(-1).cpu().numpy()

        # Создание карты на основе полученных координат
        map_center = [(float(start_point['lat']) + float(end_point['lat'])) / 2,
                      (float(start_point['lon']) + float(end_point['lon'])) / 2]
        map = folium.Map(location=map_center, zoom_start=12)

        # Добавление начальной и конечной точки на карту
        folium.Marker(
            location=(start_point['lat'], start_point['lon']),
            popup='Начальная точка',
            icon=folium.Icon(color='green')
        ).add_to(map)

        folium.Marker(
            location=(end_point['lat'], end_point['lon']),
            popup='Конечная точка',
            icon=folium.Icon(color='red')
        ).add_to(map)

        # Сохранение карты в HTML файл в папке static
        if not os.path.exists('static'):
            os.makedirs('static')
        map_path = os.path.join('static', 'map_taldykorgan.html')
        map.save(map_path)

        # Возвращение пути к карте
        return jsonify({'map_path': 'static/map_taldykorgan.html'})
    except Exception as e:
        print(f"Ошибка: {e}")  # Печать ошибки для отладки
        return jsonify({'error': str(e)}), 400

# Функция для запуска сервера и отправки тестового запроса после проверки готовности сервера
def run_server():
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)).start()
    wait_for_server()
    send_test_request()

# Функция для ожидания готовности сервера
def wait_for_server():
    url = "http://127.0.0.1:5000/"
    for _ in range(20):  # Пытаемся подключиться 20 раз
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is up and running.")
                return
        except requests.exceptions.ConnectionError:
            time.sleep(1)  # Ожидание перед новой попыткой

    print("Server did not start in time.")
    return

# Функция для отправки тестового запроса после запуска сервера
def send_test_request():
    url = "http://127.0.0.1:5000//predict"
    data = {
        "start_point": {"lat": 45.0172, "lon": 78.4040},
        "end_point": {"lat": 45.0272, "lon": 78.4140}
    }
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
        if response.status_code == 200:
            response_data = response.json()
            print("Map Path:", response_data["map_path"])
        else:
            print("Error:", response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

if __name__ == '__main__':
    run_server()
