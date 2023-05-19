import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import floyd_warshall

def compute_persistence(data, max_dimension=2):
    distance_matrix = squareform(pdist(data))
    all_pairs_shortest_paths = floyd_warshall(distance_matrix)
    inf = float('inf')
    distance_to_next_vertex = [inf] * len(data)
    coboundary_generators = [set() for _ in range(len(data))]

    # 距離行列からエッジを見つけ、対応する距離を更新
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if all_pairs_shortest_paths[i, j] < distance_to_next_vertex[i]:
                distance_to_next_vertex[i] = all_pairs_shortest_paths[i, j]
                coboundary_generators[i] = {j}
            elif all_pairs_shortest_paths[i, j] == distance_to_next_vertex[i]:
                coboundary_generators[i].add(j)

    persistence = []

    # 各次元のコホモロジー・パーシステンスを計算
    for dimension in range(max_dimension):
        if dimension == 0:
            for i in range(len(data)):
                if distance_to_next_vertex[i] != inf:
                    persistence.append((0, distance_to_next_vertex[i]))
        else:
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

import matplotlib.pyplot as plt

def plot_persistence_diagram(persistence):
    # x軸とy軸の値を格納するリストを初期化
    x_values = []
    y_values = []

    # パーシステントホモロジーの結果から、x軸とy軸の値を抽出
    for dim, persistence_value in persistence:
        x_values.append(dim)
        y_values.append(dim + persistence_value)

    # プロットの準備
    plt.scatter(x_values, y_values, marker='o', c='b')
    plt.xlabel('Dimension')
    plt.ylabel('Persistence')

    # 対角線をプロット
    max_range = max(y_values)
    plt.plot([0, max_range], [0, max_range], linestyle='--', color='gray')

    # プロットの表示
    plt.show()

# データの生成 (ここでは10個の2次元のランダムな点を生成)
data = np.random.random((10, 2))

# パーシステントホモロジーの計算
persistence = compute_persistence(data)

# パーシステンスダイアグラムのプロット
plot_persistence_diagram(persistence)


import numpy as np
from ripser import ripser
from persim import plot_diagrams

# データの生成 (ここでは10個の2次元のランダムな点を生成)
data = np.random.random((100, 2))

# パーシステントホモロジーの計算
result = ripser(data)

# 結果の表示
print(result)

# パーシステントダイアグラムのプロット
diagrams = result['dgms']
plot_diagrams(diagrams, show=True)
