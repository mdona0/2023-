import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import floyd_warshall

def compute_persistence(data, max_dimension=2):
    # 距離行列を計算
    distance_matrix = squareform(pdist(data))

    # フロイド・ウォーシャルアルゴリズムを使用して全点間距離を計算
    all_pairs_shortest_paths = floyd_warshall(distance_matrix)

    # 無限大を表す定数
    inf = float('inf')

    # 各点の距離を無限大に設定
    distance_to_next_vertex = [inf] * len(data)

    # 各点のコホモロジー生成子を格納するリスト
    coboundary_generators = [set() for _ in range(len(data))]

    # 距離行列からエッジを見つけ、対応する距離を更新
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if all_pairs_shortest_paths[i, j] < distance_to_next_vertex[i]:
                distance_to_next_vertex[i] = all_pairs_shortest_paths[i, j]
                coboundary_generators[i] = {j}
            elif all_pairs_shortest_paths[i, j] == distance_to_next_vertex[i]:
                coboundary_generators[i].add(j)

    # パーシステントホモロジーの結果を格納するリスト
    persistence = []

    # 各次元のコホモロジー・パーシステンスを計算
    for dimension in range(max_dimension):
        if dimension == 0:
            # 0次元のホモロジー (連結成分) を計算
            for i in range(len(data)):
                if distance_to_next_vertex[i] != inf:
                    persistence.append((0, distance_to_next_vertex[i]))
        else:
            # 1次元以上のホモロジーを計算
            for i in range(len(data)):
                if coboundary_generators[i]:
                    min_coboundary_distance = min(distance_to_next_vertex[j] for j in coboundary_generators[i])
                    if distance_to_next_vertex[i] != inf and min_coboundary_distance != inf:
                        persistence.append((dimension, min_coboundary_distance - distance_to_next_vertex[i]))

    return persistence

# データの生成 (ここでは10個の2次元のランダムな点を生成)
data = np.random.random((10, 2))

# パーシステントホモロジーの計算
persistence = compute_persistence(data)

# 結果の表示
print(persistence)
