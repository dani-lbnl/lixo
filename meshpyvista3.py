#!/usr/bin/env python3
import numpy as np
import pyvista as pv
from PIL import Image
import argparse

def read_multi_tiff_pillow(file_path):
    """
    Read a multi-page TIFF file and return a 3D NumPy array.
    Assumes all pages are the same size.
    """
    img = Image.open(file_path)
    pages = []
    try:
        while True:
            # Convert the current page to a NumPy array.
            page_array = np.array(img)
            # If the image is in color (3 channels), convert to grayscale.
            if page_array.ndim == 3:
                # Using luminosity method for grayscale conversion.
                page_array = np.dot(page_array[...,:3], [0.299, 0.587, 0.114])
            pages.append(page_array)
            img.seek(img.tell() + 1)
    except EOFError:
        # No more pages in the TIFF file.
        pass
    return np.stack(pages, axis=0)

def create_sphere_array(size=50):
    """
    Create a binary 3D NumPy array containing a sphere.
    Voxels inside the sphere are 1, outside are 0.
    """
    binary_stack = np.zeros((size, size, size), dtype=np.uint8)
    center = size // 2
    radius = size // 4
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2:
                    binary_stack[x, y, z] = 1
    return binary_stack

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D volume data using PyVista and optionally save mesh to STL")
    parser.add_argument(
        '--input', type=str,
        help="Path to a multi-page TIFF file. If not provided, a sphere volume will be generated."
    )
    parser.add_argument(
        '--output', type=str,
        help="Path for saving the output mesh as an STL file. (Optional)"
    )
    args = parser.parse_args()

    if args.input:
        print(f"Loading multi-page TIFF from {args.input}...")
        volume = read_multi_tiff_pillow(args.input)
    else:
        print("No input file provided. Generating a sphere volume.")
        volume = create_sphere_array(size=50)

    # Wrap the NumPy array into a PyVista dataset.
    wrapped = pv.wrap(volume)

    # Threshold the volume to isolate the region of interest.
    thresholded = wrapped.threshold(0.5)

    # Extract the outer surface of the volume.
    mesh = thresholded.extract_surface()

    # Optionally smooth the mesh.
    smoothed_mesh = mesh.smooth(n_iter=10)

    # If an output path is provided, save the mesh as an STL file.
    if args.output:
        print(f"Saving mesh to {args.output} ...")
        smoothed_mesh.save(args.output)

    # Visualize the mesh.
    plotter = pv.Plotter()
    plotter.add_mesh(smoothed_mesh, color='white')
    plotter.show()

if __name__ == '__main__':
    main()
