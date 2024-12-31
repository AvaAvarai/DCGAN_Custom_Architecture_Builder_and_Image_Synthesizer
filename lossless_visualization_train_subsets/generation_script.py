import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Function to load and preprocess an image
def load_image(filepath, target_size=None):
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    if target_size:
        img = img.resize(target_size, Image.NEAREST)  # Use faster resizing
    return np.array(img, dtype=np.int32)  # Convert to int32 to prevent overflow

# Normalization function using global min and max
def normalize_global(data, global_min, global_max):
    range_ = global_max - global_min
    if range_ == 0:
        range_ = 1  # Prevent division by zero
    return (data - global_min) / range_

# Prepare data for plotting
def prepare_data_with_offsets(images, image_shape, global_min, global_max):
    rows, cols = image_shape
    all_lines = []
    colors = []
    for idx, img_data in enumerate(images):
        # Normalize the image data globally
        img_data = normalize_global(img_data, global_min, global_max)

        # Reverse the row order to match the correct orientation
        img_data = img_data[::-1, :]

        # Apply offsets to separate each row visually
        for row in range(rows):
            offset = row  # Offset for visual separation
            y_values = img_data[row, :] + offset
            x_values = np.arange(cols)
            all_lines.append(np.column_stack((x_values, y_values)))
            colors.append(row)  # Assign a unique color per row
    return all_lines, colors

# Plotting function with optimization
def plot_and_save_parallel_coordinates(lines, colors, num_rows, num_cols, output_path):
    fig, ax = plt.subplots(figsize=(25.5, 25.5))

    # Create LineCollection for efficiency
    lc = LineCollection(lines, linewidths=0.5, alpha=0.01, cmap='viridis', array=np.array(colors))
    ax.add_collection(lc)

    # Set layout
    ax.set_xlim(0, num_cols - 1)
    ax.set_ylim(0, num_rows)
    ax.axis('off')  # Remove axes for clean plot
    ax.set_title("Parallel Coordinates Plot of Image Pixel Data with Row Offsets", fontsize=16)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")

# Main program
def main():
    # Input: Directory containing subfolders of images and target size
    root_dir = input("Enter the root directory containing subfolders of images: ")
    target_size = tuple(map(int, input("Enter the target image size (width, height): ").split(',')))

    global_min, global_max = float("inf"), float("-inf")  # Initialize global min and max

    # First pass: Calculate global min and max across all images
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for filename in os.listdir(subfolder_path):
            filepath = os.path.join(subfolder_path, filename)
            if filepath.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                img_array = load_image(filepath, target_size=target_size)
                global_min = min(global_min, img_array.min())
                global_max = max(global_max, img_array.max())

    # Print global min and max for debugging
    print(f"Global min: {global_min}, Global max: {global_max}")

    # Second pass: Process each folder and plot
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        print(f"Processing folder: {subfolder}")
        images = []
        image_shape = None

        for filename in os.listdir(subfolder_path):
            filepath = os.path.join(subfolder_path, filename)
            if filepath.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                img_array = load_image(filepath, target_size=target_size)
                if image_shape is None:
                    image_shape = img_array.shape
                elif img_array.shape != image_shape:
                    print(f"Image {filename} skipped due to size mismatch.")
                    continue
                images.append(img_array)

        if not images:
            print(f"No valid images to process in folder {subfolder}.")
            continue

        # Prepare data and plot
        lines, colors = prepare_data_with_offsets(images, image_shape, global_min, global_max)
        output_path = os.path.join(root_dir, f"{subfolder}_plot.png")
        plot_and_save_parallel_coordinates(lines, colors, image_shape[0], image_shape[1], output_path)

if __name__ == "__main__":
    main()
