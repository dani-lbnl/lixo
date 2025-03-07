#!/usr/bin/env python3

import pyvista as pv
import numpy as np

def main():
    size = 100
    binary_stack = np.zeros((size, size, size), dtype=np.uint8)
    center = size // 2
    radius = size // 4

    # Fill the array with a sphere (voxels inside the radius become 1)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2:
                    binary_stack[x, y, z] = 1

    # Wrap the NumPy array into a PyVista dataset.
    wrapped = pv.wrap(binary_stack)

    # Threshold to isolate values above 0.5 (i.e., the "inside" of the sphere)
    thresholded = wrapped.threshold(0.5)

    # Extract the outer surface of the shape.
    mesh = thresholded.extract_surface()

    # Optionally smooth the mesh.
    smoothed_mesh = mesh.smooth(n_iter=10)

    # Visualize the result.
    plotter = pv.Plotter()
    plotter.add_mesh(smoothed_mesh, color='white')
    plotter.show()

if __name__ == '__main__':
    main()
