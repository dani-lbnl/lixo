#!/usr/bin/env python3
import numpy as np
import pyvista as pv
from PIL import Image
import argparse
import cv2

def read_multi_tiff_pillow(file_path):
    """
    Read a multi-page TIFF file and return a 3D NumPy array.
    """
    img = Image.open(file_path)
    pages = []
    try:
        while True:
            page_array = np.array(img)
            # Convert color images to grayscale if necessary
            if page_array.ndim == 3:
                page_array = np.dot(page_array[...,:3], [0.299, 0.587, 0.114])
            pages.append(page_array)
            img.seek(img.tell() + 1)
    except EOFError:
        pass  # No more pages
    return np.stack(pages, axis=0)

def create_sphere_array(size=50):
    """
    Create a binary 3D NumPy array containing a sphere.
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

def random_colors(n):
    """Generate `n` distinct colors for visualization."""
    return np.random.rand(n, 3)

def label_connected_components(volume):
    """
    Label connected components in a 3D volume using OpenCV.
    """
    labeled_volume = np.zeros_like(volume, dtype=np.int32)
    current_label = 1

    for i in range(volume.shape[0]):
        num_labels, labels = cv2.connectedComponents(volume[i].astype(np.uint8))
        labels[labels > 0] += current_label  # Offset labels to be unique across slices
        labeled_volume[i] = labels
        current_label += num_labels - 1  # Update label index

    return labeled_volume, current_label - 1  # Return labeled volume and number of components

def main():
    parser = argparse.ArgumentParser(description="Extract and visualize 3D connected components with distinct colors.")
    parser.add_argument('--input', type=str, help="Path to a multi-page TIFF file.")
    parser.add_argument('--output', type=str, help="Path for saving the output mesh as an OBJ file. (Optional)")
    parser.add_argument('--reduction', type=float, default=0.9, help="Mesh decimation factor (0.0 to 1.0).")
    args = parser.parse_args()

    if args.input:
        print(f"Loading multi-page TIFF from {args.input}...")
        volume = read_multi_tiff_pillow(args.input)
    else:
        print("No input file provided. Generating a sphere volume.")
        volume = create_sphere_array(size=50)

    # Label connected components using OpenCV
    labeled_volume, num_labels = label_connected_components(volume)
    print(f"Detected {num_labels} connected components.")

    # Convert labeled volume to PyVista dataset
    wrapped = pv.wrap(labeled_volume)

    # Extract surface from labeled regions
    mesh = wrapped.threshold(0.5).extract_surface()

    # Separate connected components as distinct meshes
    components = mesh.connectivity()
    unique_regions = np.unique(components.point_data["RegionId"])
    colors = random_colors(len(unique_regions))  # Assign colors

    # Create PyVista plotter
    plotter = pv.Plotter()

    # Process each connected component separately
    for idx, region_id in enumerate(unique_regions):
        submesh = components.threshold([region_id - 0.1, region_id + 0.1], scalars="RegionId")

        # Smooth and decimate
        submesh = submesh.smooth(n_iter=5).triangulate()
        if args.reduction > 0:
            submesh = submesh.decimate(args.reduction)
        submesh = submesh.compute_normals()

        # Add to visualization
        plotter.add_mesh(submesh, color=colors[idx], show_edges=False)

        # Save as OBJ if needed
        if args.output:
            obj_path = args.output.replace(".obj", f"_{idx}.obj")
            print(f"Saving component {idx} to {obj_path} ...")
            submesh.save(obj_path)

    # Show final visualization
    plotter.show()

if __name__ == '__main__':
    main()
