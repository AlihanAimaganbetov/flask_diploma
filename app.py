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
G = ox.load_graphml("almaty_graph.gpickle")

model = GCN()
model.load_state_dict(torch.load('model_name.pth',map_location=torch.device('cpu'),weights_only=True))
# Загрузка модели
# torch.serialization.add_safe_globals([GCN])  # Разрешить использование класса GCN
# model = torch.load("full_gcn_model.pth")  # Загружаем модель

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

        # Преобразуем строки в числа (float)
        start_lat = float(start_point['lat'])
        start_lon = float(start_point['lon'])
        end_lat = float(end_point['lat'])
        end_lon = float(end_point['lon'])

        # Логирование входных данных
        print(f"Start point: {start_point}, End point: {end_point}")

        # Загрузка графа через osmnx


        # Преобразуем координаты в индексы узлов графа (например, с помощью nearest_nodes из osmnx)
        start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
        end_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

        # Логирование узлов
        print(f"Start node: {start_node}, End node: {end_node}")

        # Проверка на корректность узлов
        if start_node is None or end_node is None:
            return jsonify({'error': 'Invalid nodes: could not find nearest nodes for the given coordinates'}), 400

        # Вычисление кратчайшего пути
        route = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
        print("Кратчайший путь:", route)

        # Получаем координаты для маршрута
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]  # (lat, lon)

        # Создание карты
        map_center = [(start_lat + end_lat) / 2, (start_lon + end_lon) / 2]
        map = folium.Map(location=map_center, zoom_start=12)

        # Добавление начальной и конечной точки на карту
        folium.Marker(
            location=(start_lat, start_lon),
            popup='Начальная точка',
            icon=folium.Icon(color='green')
        ).add_to(map)

        folium.Marker(
            location=(end_lat, end_lon),
            popup='Конечная точка',
            icon=folium.Icon(color='red')
        ).add_to(map)

        # Добавление пути на карту
        folium.PolyLine(route_coords, color="blue", weight=4, opacity=0.7).add_to(map)

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
