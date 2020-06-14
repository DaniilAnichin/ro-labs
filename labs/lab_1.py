from typing import Iterable

from .base import Shape, ShapeClass


def evaluate_shape(shape: Shape, classes: Iterable[ShapeClass]):
    c1, f1 = shape.distance_classify(classes)
    c2, f2 = shape.decisive_classify(classes)
    c3, f3 = shape.division_classify(classes)
    print(f'{shape}:  {c1}({f1:5.3f});  {c2}({f2:7.3f});  {c3}({f3:6.3f})')


def evaluate_shapes(shapes: Iterable[Shape], classes: Iterable[ShapeClass]):
    [evaluate_shape(shape, classes) for shape in shapes]
    print()


def make_classes(class_shapes, name: str = 'ω') -> Iterable[ShapeClass]:
    return [ShapeClass(shapes, f"{name}{i}") for i, shapes in enumerate(class_shapes, 1)]


def main():
    print('Task 1; Two classes, 10 shapes')
    classes = make_classes([[[5, 6]], [[-3, -4]]])
    shapes = Shape.make_shapes([
        [3, 2], [-1, -2], [9, 4], [-4, 0], [4, -1],
        [0, -3], [0, 5], [-2, 2], [1, -3], [-3, 3],
    ])
    evaluate_shapes(shapes, classes)

    print('Task 2; Two classes, 20 shapes')
    classes = make_classes([[[2, 4], [3, 2]], [[-4, -2], [-1, -3], [-5, 0]]])
    shapes = Shape.make_shapes([
        [0, 5], [1, 4], [-1, 3], [1, 1], [2, 1],
        [1, 2], [-3, 2], [-2, -4], [2, -4], [-2, 5],
        [6, -2], [3, 4], [-2, 2], [-3, -3], [1, -5],
        [0, 4], [0, -3], [-2, 0], [4, 0], [0, 3],
    ])
    evaluate_shapes(shapes, classes)

    averaged_classes = make_classes([
        [sum([e.coords for e in sc.shapes]) / len(sc.shapes)]
        for sc in classes], 'ωa',
    )
    print('Averaged class shapes: ')
    for cls in averaged_classes:
        print(f'{cls.name}: {cls.shapes[0]}')
    evaluate_shapes(shapes, averaged_classes)

    print('Task 3; Three classes, 20 shapes')
    classes = make_classes([[[-3, -4]], [[-3.5, -2.8]], [[2, 4]]])
    evaluate_shapes(shapes, classes)


if __name__ == '__main__':
    main()