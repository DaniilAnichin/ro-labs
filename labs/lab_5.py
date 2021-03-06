from .lab_3 import isodata_clusterize, print_clusters


def main():
    print('Task 1; 21 shape')
    shapes = [
        [7.1, 7.3], [2.79, 7.54], [3.34, 5.17], [7.2, 3.06], [5.92, 7.28],
        [4.44, 3.37], [-4.47, 6.41], [-6.42, 7.56], [-7.25, 6.16], [-8.45, 4.28],
        [-2.89, 2.69], [-5.8, 5.06], [-5.92, 2.69], [0.74, -2.66], [-1.29, 2.94],
        [-2.89, -5.71], [2.37, -7.23], [-2.02, -8.42], [-1.62, 5.12], [2.29, -5.12],
        [0.54, -6.4]
    ]
    print_clusters(isodata_clusterize(
        shapes,
        initial_cluster_centers=[[-5, -5], [5, 5]],
        K=4,
        theta_n=1,
        theta_s=4,
        theta_c=4,
        L=1,
        I=10,
    ))

    print('Task 2; 35 shapes')
    shapes = [
        [-7.02, 2.94], [-5.77, 2.16], [-8.17, -1.19], [-2.42, -2.87], [-4.49, 0.76],
        [-4.54, -1.51], [0.26, -5.71], [-0.29, -7.21], [1.74, -7.8], [-1.84, -8.49],
        [1.01, -9.44], [-1.19, -5.76], [7, 0.33], [3.64, 0.05], [5.97, 0.42],
        [5.24, -1.99], [5.39, -3.97], [7.95, -3.17], [7.05, -1.7], [3.94, -2.25],
        [-0.64, 7.88], [-1.34, 9.17], [-2.57, 9.17], [-3.94, 7.74], [-1.04, 5.06],
        [-3.02, 5.52], [-1.87, 7.7], [7.7, 7.26], [5.52, 8.36], [4.52, 7.22],
        [4.87, 5.72], [7.15, 4.67], [6.82, 7.1], [6.15, 6.94], [6.65, 9.4],
    ]
    print_clusters(isodata_clusterize(
        shapes,
        initial_cluster_centers=[[-5, -5], [5, 5]],
        K=6,
        theta_n=1,
        theta_s=4,
        theta_c=4,
        L=1,
        I=10,
    ))

    print('Task 3 (variant 1); 24 shapes')
    shapes = [
        [6.98, 7.7], [4.7, 8.09], [5.57, 4.94], [8.48, 4.76], [5.86, 6.94],
        [-5.5, 8.36], [-6.31, 7.46], [-7.43, 5.45], [-2.71, 6.15], [-4.79, 6.79],
        [-5.64, 4.42], [-5.12, - 2.97], [-7.32, -3.09], [-4.36, -4.94],
        [-5.86, -4.55],
        [-7.05, - 5.76], [3.31, -2.18], [4.63, -2.37], [6.04, -3.73],
        [3.38, - 4.52],
        [5.3, -6.19], [4.59, -4.15], [-6.35, -1.94], [-4.14, 4.85],
    ]
    print_clusters(isodata_clusterize(
        shapes,
        initial_cluster_centers=[[-5, -5], [5, 5]],
        K=4,
        theta_n=1,
        theta_s=3,
        theta_c=6,
        L=1,
        I=10,
    ))


if __name__ == '__main__':
    main()