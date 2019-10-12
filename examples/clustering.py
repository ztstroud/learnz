import numpy as np

import learnz.ml.clustering as clustering


# Create a data set
data = np.array([[0,   0],
                 [0,   1],
                 [1,   0],
                 [100, 100],
                 [100, 101],
                 [101, 100]])

# Run gonzalez algorithm to get centers
clusters, centers = clustering.gonzalez(data, 2, return_centers = True)

# Display clusters
for cluster_index, cluster in enumerate(clusters):
    print(f"Cluster {cluster_index}, center = {centers[cluster_index]}")

    for vector in cluster:
        print(f"  {vector}")

    print()
