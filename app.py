from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
import numpy as np
import folium
import os
import requests
import json
import threading
import time
import uuid
import networkx as nx
import osmnx as ox

# Импортируем класс GCN
from model import GCN  # Убедитесь, что файл model.py существует и класс GCN в нем определен

# Установка устройства: используем GPU, если он доступен, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка модели
torch.serialization.add_safe_globals([GCN])  # Разрешить использование класса GCN
model = torch.load("full_gcn_model.pth", map_location=device)  # Загружаем модель

model.eval()  # Переводим модель в режим инференса

# Создание Flask приложения
app = Flask(__name__)
CORS(app)  # Добавляем поддержку CORS для всего приложения

@app.route('/')
def index():
    return "Flask server is up and running!"

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

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

        # Загрузка графа через osmnx
        G = ox.load_graphml("almaty_graph.gpickle")
        print(nx.info(G))  # Проверка информации о графе

        # Преобразуем координаты в индексы узлов графа (например, с помощью nearest_nodes из osmnx)
        start_node = ox.distance.nearest_nodes(G, X=start_point['lon'], Y=start_point['lat'])
        end_node = ox.distance.nearest_nodes(G, X=end_point['lon'], Y=end_point['lat'])

        print(f"Start node: {start_node}, End node: {end_node}")

        # Вычисление кратчайшего пути
        route = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
        print("Кратчайший путь:", route)

        # Создание карты
        map_center = [(start_point['lat'] + end_point['lat']) / 2,
                      (start_point['lon'] + end_point['lon']) / 2]
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

        # Сохранение карты в HTML файл
        map_filename = f"map_{uuid.uuid4().hex}.html"
        map_path = os.path.join('static', map_filename)
        map.save(map_path)

        # Возвращаем путь к сохраненной карте
        return jsonify({'map_path': f'static/{map_filename}'}), 200

    except Exception as e:
        print(f"Ошибка: {e}")  # Печать ошибки для отладки
        return jsonify({'error': str(e)}), 400

def run_server():
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)).start()
    wait_for_server()
    send_test_request()

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

def send_test_request():
    url = "http://127.0.0.1:5000/predict"
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
