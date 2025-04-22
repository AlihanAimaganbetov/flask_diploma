from flask import Flask, request, jsonify
from flask_cors import CORS
import osmnx as ox
import networkx as nx
import torch
from model import GCN

app = Flask(__name__)
CORS(app)

# Загружаем граф Алматы
G = ox.load_graphml("almaty_graph.gpickle")

# Загружаем модель (если она используется в других функциях)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = GCN()
model.load_state_dict(torch.load('model_name.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

@app.route('/')
def index():
    return "Flask GeoJSON Route API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'start_point' not in data or 'end_point' not in data:
            return jsonify({'error': 'Missing required fields: start_point or end_point'}), 400

        start_point = data['start_point']
        end_point = data['end_point']

        start_lat = float(start_point['lat'])
        start_lon = float(start_point['lon'])
        end_lat = float(end_point['lat'])
        end_lon = float(end_point['lon'])

        start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
        end_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

        if start_node is None or end_node is None:
            return jsonify({'error': 'Invalid nodes: could not find nearest nodes for the given coordinates'}), 400

        route = nx.shortest_path(G, source=start_node, target=end_node, weight='length')
        route_coords = [(G.nodes[node]['x'], G.nodes[node]['y']) for node in route]  # (lon, lat)

        geojson_route = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": route_coords
            },
            "properties": {}
        }

        return jsonify({"route": geojson_route}), 200

    except Exception as e:
        print(f"Ошибка: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
