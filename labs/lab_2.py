import random
from typing import Dict, Tuple, List

import click
import numpy as np

from .base import Shape


def optional(function):
    def wrapper(cluster, *args, **kwargs):
        if not cluster:
            return float('nan')
        return function(cluster, *args, **kwargs)
    return wrapper


class Cluster(dict):
    neighbours: Tuple['Cluster'] = ()

    def __init__(self, center: Shape, name: str):
        super().__init__()
        self.center = center
        self.name = name

    def __str__(self):
        name = f'{self.name:>3}({self.center})' if self.name else self.center
        cluster_str = f'''{name} with
    max(d) = {self.max_distance:7.3f},
    min(d) = {self.min_distance:7.3f},
    avg(d) = {self.avg_distance:7.3f},
    var(d) = {self.variance:7.3f},
    std(d) = {self.std_deviation:7.3f}'''
        if self:
            elements = '\n        '.join([str(element) for element in self.keys()])
            cluster_str += f''',
    with elements:
        {elements}'''
        if self.neighbours:
            distances = '\n        '.join([
                f'{cluster.name:>3}: {self.center.distance(cluster.center):7.3f}'
                for cluster in self.neighbours
            ])
            cluster_str += f'''
    with distances to others:
        {distances}
        '''
        return cluster_str

    @property
    @optional
    def max_distance(self):
        return max(self.values())

    @property
    @optional
    def min_distance(self):
        return min(self.values())

    @property
    @optional
    def avg_distance(self):
        return np.average(list(self.values()))

    @property
    @optional
    def variance(self):
        return np.var(list(self.values()))

    @property
    @optional
    def std_deviation(self):
        return np.sqrt(self.variance)


def clusterize(shapes: List[Shape], center: Shape, max_distance: int) -> List[Cluster]:
    shapes = [shape for shape in shapes if shape != center]
    clusters = [Cluster(center, 'c0')]

    for shape in shapes:
        distances = [shape.distance(c2.center) for c2 in clusters]
        if min(distances) > max_distance:
            clusters.append(Cluster(shape, f'c{len(clusters)}'))
        else:
            cluster = clusters[int(np.argmin(distances))]
            cluster[shape] = min(distances)

    for c2 in clusters:
        c2.neighbours = tuple(cluster for cluster in clusters if cluster != c2)
    return clusters


def clusterize_by_distances(
        shapes: List[Shape], center: Shape, max_distances: List[int],
) -> Dict[int, Cluster]:
    return {
        max_distance: clusterize(shapes, center, max_distance)
        for max_distance in max_distances
    }


def first_getter_from_option(center_option: str):
    if center_option == 'FIRST':
        print('Using first shape')
        return lambda x: x[0]
    if center_option == 'LAST':
        print('Using last shape')
        return lambda x: x[-1]
    print('Using random shape')
    return lambda x: random.choice(x)


def print_results(results_by_t):
    for distance, clusters in results_by_t.items():
        print(f'T = {distance}')
        print('Clusters count:', len(clusters))
        for cluster in clusters:
            print(cluster)
        print()


@click.command()
@click.option(
    '--center', type=click.Choice(['FIRST', 'LAST', 'RANDOM']), default='FIRST'
)
def main(center):
    first_getter = first_getter_from_option(center)

    print('Task 1; 21 shape, T: {2, 5, 7, 9, 12}')
    max_distances = [2, 5, 7, 9, 12]
    shapes = Shape.make_shapes([
        [7.1, 7.3], [2.79, 7.54], [3.34, 5.17], [7.2, 3.06], [5.92, 7.28],
        [4.44, 3.37], [-4.47, 6.41], [-6.42, 7.56], [-7.25, 6.16], [-8.45, 4.28],
        [-2.89, 2.69], [-5.8, 5.06], [-5.92, 2.69], [0.74, -2.66], [-1.29, 2.94],
        [-2.89, -5.71], [2.37, -7.23], [-2.02, -8.42], [-1.62, 5.12], [2.29, -5.12],
        [0.54, -6.4]
    ])
    print_results(clusterize_by_distances(shapes, first_getter(shapes), max_distances))

    print('Task 2; 35 shapes, T: {3, 4, 6, 7, 8, 10}')
    max_distances = [3, 4, 6, 7, 8, 10]
    shapes = Shape.make_shapes([
        [-7.02, 2.94], [-5.77, 2.16], [-8.17, -1.19], [-2.42, -2.87], [-4.49, 0.76],
        [-4.54, -1.51], [0.26, -5.71], [-0.29, -7.21], [1.74, -7.8], [-1.84, -8.49],
        [1.01, -9.44], [-1.19, -5.76], [7, 0.33], [3.64, 0.05], [5.97, 0.42],
        [5.24, -1.99], [5.39, -3.97], [7.95, -3.17], [7.05, -1.7], [3.94, -2.25],
        [-0.64, 7.88], [-1.34, 9.17], [-2.57, 9.17], [-3.94, 7.74], [-1.04, 5.06],
        [-3.02, 5.52], [-1.87, 7.7], [7.7, 7.26], [5.52, 8.36], [4.52, 7.22],
        [4.87, 5.72], [7.15, 4.67], [6.82, 7.1], [6.15, 6.94], [6.65, 9.4],
    ])
    print_results(clusterize_by_distances(shapes, first_getter(shapes), max_distances))

    print('Task 3 (variant 1); 24 shapes, T: {2, 5, 7, 9, 12}')
    max_distances = [2, 5, 7, 9, 12]
    shapes = Shape.make_shapes([
        [6.98, 7.7], [4.7, 8.09], [5.57, 4.94], [8.48, 4.76], [5.86, 6.94],
        [-5.5, 8.36], [-6.31, 7.46], [-7.43, 5.45], [-2.71, 6.15], [-4.79, 6.79],
        [-5.64, 4.42], [-5.12, - 2.97], [-7.32, -3.09], [-4.36, -4.94], [-5.86, -4.55],
        [-7.05, - 5.76], [3.31, -2.18], [4.63, -2.37], [6.04, -3.73], [3.38, - 4.52],
        [5.3, -6.19], [4.59, -4.15], [-6.35, -1.94], [-4.14, 4.85],
    ])
    print_results(clusterize_by_distances(shapes, first_getter(shapes), max_distances))


if __name__ == '__main__':
    main()
