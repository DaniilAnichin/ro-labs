"""Based on python 2.7 lib https://github.com/PyRadar/pyradar"""
from copy import deepcopy
from pprint import pprint

import numpy as np


def find_nearest_pairs(clusters, theta_c):
    pair_dists = []
    for i, cluster_1 in enumerate(clusters):
        for cluster_2 in clusters[i + 1:]:
            distance = distance_to(cluster_1[0], cluster_2[0])
            if distance < theta_c:
                pair_dists.append((distance, (cluster_1, cluster_2)))

    return [pair for distance, pair in sorted(pair_dists, key=lambda k: k[0])]


def distance_to(a, b):
    return np.linalg.norm(a - b)


def isodata_iteration(
        clusters: list,
        K: int = 5,
        L: int = 4,
        theta_n: int = 10,
        theta_s: float = 1,
        theta_c: int = 20,
        split_clusters: bool = False,
) -> list:
    # Step 4
    to_discard = [cluster for cluster in clusters if len(cluster[1]) <= theta_n]
    if to_discard:
        print(f'Clusters discarded: {to_discard}')
    clusters = [cluster for cluster in clusters if len(cluster[1]) > theta_n]

    # Step 5
    clusters = [
        [sum(shapes) / len(shapes), shapes] for _, shapes in clusters
    ]

    # Step 6
    cluster_average_distances = [
        np.mean([distance_to(shape, center) for shape in shapes])
        for center, shapes in clusters
    ]

    # Step 7
    total_avg = np.mean([
        distance_to(shape, center) for center, shapes in clusters for shape in shapes
    ])

    # Step 8
    broken_cluster = False
    if not (split_clusters or len(clusters) >= K * 2):
        # Step 9
        deviations_by_cluster = []
        for center, shapes in clusters:
            deviations = np.sqrt(sum((shape - center) ** 2 for shape in shapes) / len(shapes))
            deviations_by_cluster.append([np.max(deviations), np.argmax(deviations)])

        # Step 10
        for (center, shapes), (max_deviation, deviation_index), avg_distance in zip(
                clusters.copy(), deviations_by_cluster,
                cluster_average_distances,
        ):
            if max_deviation <= theta_s:
                continue

            if (avg_distance > total_avg and len(shapes) > 2 * (
                    theta_n + 1)) or len(clusters) <= K / 2:
                broken_cluster = True
                right_center, left_center = deepcopy(center), deepcopy(center)
                right_center[deviation_index] += max_deviation / 2
                left_center[deviation_index] -= max_deviation / 2

                right_shapes, left_shapes = [], []
                for shape in shapes:
                    if distance_to(shape, right_center) < distance_to(shape, left_center):
                        right_shapes.append(shape)
                    else:
                        left_shapes.append(shape)

                clusters.remove([center, shapes])
                clusters.extend([[right_center, right_shapes], [left_center, left_shapes]])

    if broken_cluster:
        return clusters

    # Step 12 & 13
    mergeable_pairs = find_nearest_pairs(clusters, theta_c)[:L]

    # Step 14
    merged_clusters = []
    for cluster_1, cluster_2 in mergeable_pairs:
        if cluster_1 in merged_clusters or cluster_2 in merged_clusters:
            continue
        merged_clusters += [cluster_1, cluster_2]

        new_center = np.average(
            [cluster_1[0], cluster_2[0]],
            weights=[len(cluster_1[1]), len(cluster_2[1])],
            axis=0
        )
        clusters.append([new_center, cluster_1[1] + cluster_2[1]])
    return [
        cluster for cluster in clusters
        if not any(np.allclose(cluster[0], to_remove[0]) for to_remove in
                   merged_clusters)
    ]


def isodata_clusterize(
        init_shapes: list,
        initial_cluster_centers: list,
        K: int = 5,
        L: int = 4,
        I: int = 100,
        theta_n: int = 10,
        theta_s: float = 1,
        theta_c: int = 20,
):
    init_shapes = [np.array(shape) for shape in init_shapes]
    clusters = [[np.array(center), []] for center in initial_cluster_centers]

    # Step 3
    for shape in init_shapes:
        distances_to_cluster_centers = [
            distance_to(shape, center) for center, _ in clusters
        ]
        clusters[int(np.argmin(distances_to_cluster_centers))][1].append(shape)

    for i in range(1, I + 1):
        if i == I:
            theta_c = 0
        split_clusters = i == I or i % 2 == 0
        clusters = isodata_iteration(
            clusters, K, L, theta_n, theta_s, theta_c, split_clusters
        )

    return clusters


def print_clusters(clusters: list):
    for center, shapes in clusters:
        print(f"Center: {center}")
        pprint(shapes)
    print()


def main():
    print('Task 1.1; 8 shapes, 1 cluster, 1 initial center')
    shapes = [
        [0, 0], [1, 1],
        [2, 2], [4, 3],
        [5, 3], [4, 4],
        [5, 4], [6, 5]
    ]
    print_clusters(isodata_clusterize(
        shapes,
        initial_cluster_centers=[[0, 0]],
        K=2,
        theta_n=1,
        theta_s=1,
        theta_c=4,
        L=0,
        I=4,
    ))
    print('Task 1.2; 8 shapes, 2 clusters, 2 initial centers')
    print_clusters(isodata_clusterize(
        shapes,
        initial_cluster_centers=[[0, 0], [1, 1]],
        K=3,
        theta_n=2,
        theta_s=0.8,
        theta_c=3,
        L=2,
        I=5,
    ))
    print('Task 1.3; 8 shapes, 3 clusters, 3 initial centers')
    print_clusters(isodata_clusterize(
        shapes,
        initial_cluster_centers=[[0, 0], [4, 4], [6, 5]],
        K=4,
        theta_n=2,
        theta_s=0.5,
        theta_c=2,
        L=3,
        I=6,
    ))

    print('Task 2.1; 20 shapes, 1 cluster, 1 initial center')
    shapes = [
        [0, 0], [1, 1],
        [0, 1], [1, 1],
        [2, 1], [1, 2],
        [3, 2], [1, 7],
        [0, 7], [0, 8],
        [1, 8], [0, 9],
        [2, 8], [2, 9],
        [6, 6], [7, 6],
        [8, 6], [7, 7],
        [8, 8], [9, 9]
    ]
    print_clusters(isodata_clusterize(
        shapes,
        initial_cluster_centers=[[0, 0]],
        K=3,
        theta_n=2,
        theta_s=1,
        theta_c=4,
        L=2,
        I=4,
    ))
    print('Task 2.2; 20 shapes, 3 clusters, 3 initial centers')
    print_clusters(isodata_clusterize(
        shapes,
        initial_cluster_centers=[[0, 0], [1, 0], [0, 1]],
        K=4,
        theta_n=1,
        theta_s=0.5,
        theta_c=3,
        L=1,
        I=5,
    ))
    print('Task 2.3; 20 shapes, 4 cluster, 2 initial centers')
    print_clusters(isodata_clusterize(
        shapes,
        initial_cluster_centers=[[0, 0], [9, 1]],
        K=2,
        theta_n=4,
        theta_s=1.2,
        theta_c=5,
        L=3,
        I=6,
    ))


if __name__ == '__main__':
    main()
