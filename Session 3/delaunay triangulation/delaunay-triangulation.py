import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Generate some random 2D points
np.random.seed(42)
points = np.random.rand(20, 2)

# Perform Delaunay triangulation
tri = Delaunay(points)

# Plot the points and the triangulation
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Delaunay Triangulation')
plt.show()
