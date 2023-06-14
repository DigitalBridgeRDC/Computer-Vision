import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Define the sparse face mesh points
points = np.array([
    [100, 100],    # Vertex 0
    [150, 200],    # Vertex 1
    [200, 150],    # Vertex 2
    [250, 200],    # Vertex 3
    [300, 100],    # Vertex 4
    [175, 125],    # Vertex 5
    [225, 125]     # Vertex 6
])

# Perform Delaunay triangulation
tri = Delaunay(points)

# Plot the sparse face mesh and the triangulation
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Delaunay Triangulation on Sparse Face Mesh')
plt.show()
