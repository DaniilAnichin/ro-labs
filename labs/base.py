from typing import Iterable, Union, SupportsFloat, Tuple, Optional

import numpy as np


CoordsType = Union[Iterable[SupportsFloat], np.array]


class Shape:
    floating = False

    def __init__(self, coords: CoordsType, name: str = ''):
        self.name = name
        self.coords = coords
        if not isinstance(self.coords, np.ndarray):
            self.coords = np.array(self.coords)
        if self.coords.dtype == float:
            self.floating = True

    def __repr__(self) -> str:
        return f'<Shape "{self.name}": {self.coords.tolist()}>'

    def __str__(self) -> str:
        if self.floating:
            coords = ', '.join(f"{coord:>6.3f}" for coord in self.coords)
        else:
            coords = ', '.join(f"{coord:>2}" for coord in self.coords)
        return f'{self.name:>3} {{{coords}}}'

    def distance(self, standard_shape: 'Shape'):
        return np.linalg.norm(self.coords - standard_shape.coords)

    def decisive_value(self, standard_shape: 'Shape'):
        return (np.dot(self.coords.T, standard_shape.coords)
                - 0.5 * np.dot(standard_shape.coords.T, standard_shape.coords))

    def distance_classify(
            self, classes: Iterable['ShapeClass']
    ) -> Tuple[Optional['ShapeClass'], SupportsFloat]:
        cls, best_distance = None, 10 ** 10
        for cls in classes:
            distance = min([self.distance(shape) for shape in cls.shapes])
            if distance < best_distance:
                cls, best_distance = cls, distance
        return cls, best_distance

    def decisive_classify(
            self, classes: Iterable['ShapeClass']
    ) -> Tuple[Optional['ShapeClass'], SupportsFloat]:
        cls, best_value = None, -10 ** 10
        for cls in classes:
            value = max([
                self.decisive_value(standard_shape)
                for standard_shape in cls.shapes
            ])
            if value > best_value:
                cls, best_value = cls, value
        return cls, best_value

    def division_classify(
            self, classes: Iterable['ShapeClass']
    ) -> Tuple[Optional['ShapeClass'], SupportsFloat]:
        classes = set(classes)
        for cls in classes:
            cls_results = []
            for second_cls in classes - {cls}:
                di = max([
                    self.decisive_value(standard_shape)
                    for standard_shape in cls.shapes
                ])
                dj = max([
                    self.decisive_value(standard_shape)
                    for standard_shape in second_cls.shapes
                ])
                cls_results.append(di - dj)

            if all(value >= 0 for value in cls_results):
                return cls, np.max(cls_results)


class ShapeClass:
    def __init__(self, shapes: Iterable[Union[Shape, CoordsType]], name: str):
        self.name = name
        self.shapes = list(shapes)
        if self.shapes and not isinstance(self.shapes[0], Shape):
            self.shapes = list(map(Shape, self.shapes))

    def __eq__(self, other: 'ShapeClass') -> bool:
        return set(self.shapes) == set(other.shapes)

    def __repr__(self) -> str:
        return f'<StandardClass "{self.name}">'

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash(tuple(self.shapes))
