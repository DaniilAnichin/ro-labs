from typing import List

import numpy as np
import matplotlib.pyplot as plt

from .base import Shape


class BayesianClass:
    def __init__(self, name, shapes, pr):
        self.name = name
        self.probability = pr
        self.shapes = [
            shape if isinstance(shape, Shape) else Shape(shape)
            for shape in shapes
        ]
        self.m = np.mean([shape.coords for shape in self.shapes], axis=0)
        self.coverats = self.calc_coverats()
        self.reverse_coverats = np.linalg.inv(self.coverats)
        self.linear_coefs = [round(x, 5) for x in self.division_coefs()]
        self.square_coefs = [round(x, 5) for x in self.decisive_coefs()]

    def calc_coverats(self):
        feature_normal = np.stack([shape.coords for shape in self.shapes], axis=0)

        coverats = [[0, 0], [0, 0]]
        for feature in feature_normal:
            row = np.array([feature])
            coverats += row.transpose().dot(row) / len(feature_normal)

        row = np.array(self.m)
        coverats += row.transpose().dot(row) / len(feature_normal)
        return coverats

    def decisive_coefs(self):
        return [
            -0.5 * self.reverse_coverats[0][0],
            -0.5 * self.reverse_coverats[1][1],
            0.5 * (2 * self.reverse_coverats[0][0] * self.m[0] + (self.reverse_coverats[0][1] + self.reverse_coverats[1][0]) * self.m[1]),
            0.5 * (2 * self.reverse_coverats[1][1] * self.m[1] + (self.reverse_coverats[0][1] + self.reverse_coverats[1][0]) * self.m[0]),
            -0.5 * (self.reverse_coverats[0][1] + self.reverse_coverats[1][0]),
            -0.5 * (
                    (self.reverse_coverats[0][1] + self.reverse_coverats[1][0]) * self.m[0] *
                    self.m[1]
                    + self.m[0] * self.m[0] * self.reverse_coverats[0][0]
                    + self.m[1] * self.m[1] * self.reverse_coverats[1][1]
                    + np.log(np.linalg.det(self.coverats))
            ) + np.log(self.probability),
        ]

    def division_coefs(self):
        m_transposed = np.transpose(self.m).dot(self.reverse_coverats).dot(self.m)
        return [
            0,
            0,
            0,
            self.m[0] * self.reverse_coverats[0][0] + self.m[1] * self.reverse_coverats[0][1],
            self.m[1] * self.reverse_coverats[1][1] + self.m[0] * self.reverse_coverats[1][0],
            -0.5 * m_transposed + np.log(self.probability),
        ]

    def __eq__(self, other):
        return self.name == other.name and self.shapes == other.shapes

    def __repr__(self):
        return self.name

    def __str__(self):
        return f'''{self.name} -> середнє: {self.m}
    ковaріаційна матриця:
{self.coverats}
    обернена до неї матриця:
{self.reverse_coverats}
    вірогідність класу: {self.probability}
    лінійні коефіцієнти: {self.linear_coefs}
    квадратичні коефіцієнти: {self.square_coefs}
'''


def search_double_function(first_c, second_c):
    return np.array(first_c) - np.array(second_c)


def calc_func(coef, x, y):
    return coef[0] * x * x + coef[1] * y * y + coef[2] * x * y + coef[3] * x + coef[4] * y + coef[5]


def print_classes(classes: List[BayesianClass], shapes: List[Shape]):
    for etalon in classes:
        print(etalon)

    coef_in_between = search_double_function(classes[1].square_coefs, classes[0].square_coefs)
    print(f"Коефіцієнти функції d12: {coef_in_between}")

    i = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(i, i)
    Z = calc_func(coef_in_between, X, Y)

    # 2-dim intersection plot
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    plt.grid(True)
    ax.minorticks_on()
    ax.grid(which='minor', color='k', linestyle=':')
    ax.set_title('Відображення роздільної площини')
    ax.clabel(CS, inline=1, fontsize=10)
    ax.plot(*np.transpose([shape.coords for shape in classes[0].shapes]), 'ro')
    ax.plot(*np.transpose([shape.coords for shape in classes[1].shapes]), 'go')
    plt.show()


def main():
    print('Task 1; Two classes, 10 shapes')
    classes = [
        BayesianClass("w1", [[3, 2], [9, 4], [4, -1], [0, 5]], 0.5),
        BayesianClass("w2", [[-1, -2], [-4, 0], [0, -3], [-2, 2], [1, -3], [-3, 3]], 0.5)
    ]

    shapes = Shape.make_shapes([
        [3, 2], [-1, -2], [9, 4], [-4, 0], [4, -1],
        [0, -3], [0, 5], [-2, 2], [1, -3], [-3, 3],
    ])
    print_classes(classes, shapes)

    print('Task 2; Two classes, 20 shapes')
    classes = [
        BayesianClass("w1", [
            [0, 5], [1, 4], [-1, 3], [1, 1], [2, 1],
            [1, 2], [-2, 5], [6, -2], [3, 4], [0, 4],
            [4, 0], [0, 3],
        ], 0.5),
        BayesianClass("w2", [
            [-3, 2], [-2, -4], [2, -4], [-2, 2], [-3, -3],
            [1, -5], [0, -3], [-2, 0],
        ], 0.5)
    ]
    shapes = Shape.make_shapes([
        [0, 5], [1, 4], [-1, 3], [1, 1], [2, 1],
        [1, 2], [-3, 2], [-2, -4], [2, -4], [-2, 5],
        [6, -2], [3, 4], [-2, 2], [-3, -3], [1, -5],
        [0, 4], [0, -3], [-2, 0], [4, 0], [0, 3],
    ])
    print_classes(classes, shapes)


if __name__ == '__main__':
    main()
