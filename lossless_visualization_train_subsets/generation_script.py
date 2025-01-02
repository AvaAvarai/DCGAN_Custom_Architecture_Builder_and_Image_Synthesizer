# pip install numpy pillow matplotlib
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
def prepare_data_with_offsets(images, image_shape, global_min, global_max, color):
    """Now takes a color parameter for all lines from the same folder"""
    rows, cols = image_shape
    all_lines = []
    colors = []
    
    for idx, img_data in enumerate(images):
        # Make a copy of the data
        processed_data = img_data.copy()
        
        # Normalize the image data globally
        processed_data = normalize_global(processed_data, global_min, global_max)

        # Reverse the row order to match the correct orientation
        processed_data = processed_data[::-1, :]

        # Apply offsets to separate each row visually
        for row in range(rows):
            offset = row  # Offset for visual separation
            y_values = processed_data[row, :] + offset
            x_values = np.arange(cols)
            all_lines.append(np.column_stack((x_values, y_values)))
            colors.append(color)  # Use the same color for all rows of this image
    return all_lines, colors

# Plotting function with optimization
def plot_and_save_parallel_coordinates(lines, colors, num_rows, num_cols, output_path):
    fig, ax = plt.subplots(figsize=(25.5, 25.5))

    # Calculate opacity based on number of samples
    n_samples = len(lines) // num_rows  # Total number of images
    opacity = max(0.05, min(0.2, 5.0 / n_samples))  # Ensure opacity is between 0.05 and 0.2
    print(f"Using opacity: {opacity:.3f}")

    # Create LineCollection for efficiency
    lc = LineCollection(lines, linewidths=0.5, alpha=opacity, colors=colors)
    ax.add_collection(lc)

    # Set layout
    ax.set_xlim(0, num_cols - 1)
    ax.set_ylim(0, num_rows)
    
    # Add labels and grid
    ax.set_xlabel('Pixel Position (x)', fontsize=12)
    ax.set_ylabel('Row of Pixel Data (y)', fontsize=12)
    ax.grid(True, alpha=0.2)
    
    ax.set_title("Multi-Row Parallel Coordinates Plot of Image Pixel Data", fontsize=16)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")

def find_all_images(directory):
    """Recursively find all image files in directory and subdirectories"""
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                image_files.append(os.path.join(root, file))
    return image_files

def process_images(image_files, target_size):
    """Process a list of image files and return list of images and their shape"""
    images = []
    image_shape = None
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for filepath in image_files:
        try:
            img_array = load_image(filepath, target_size=target_size)
            if image_shape is None:
                image_shape = img_array.shape
            elif img_array.shape != image_shape:
                skipped_count += 1
                continue
            images.append(img_array)
            processed_count += 1
        except Exception as e:
            error_count += 1
    
    print(f"Processing summary: {processed_count} processed, {skipped_count} skipped, {error_count} failed")
    return images, image_shape

# Main program
def main():
    # Input: Directory containing images and target size
    root_dir = input("Enter the directory containing images: ")
    print(f"Looking in directory: {root_dir}")
    target_size = tuple(map(int, input("Enter the target image size (width, height): ").split(',')))

    # Get script directory for saving plots
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Group images by folder
    folder_images = {}
    print("Grouping images by folder...")
    for filepath in find_all_images(root_dir):
        folder = os.path.basename(os.path.dirname(filepath))
        if folder not in folder_images:
            folder_images[folder] = []
        folder_images[folder].append(filepath)

    # Calculate global min and max across ALL images
    global_min, global_max = float("inf"), float("-inf")
    all_processed_images = []
    
    print("\nFirst pass: calculating global min and max...")
    for folder, filepaths in folder_images.items():
        print(f"\nProcessing folder: {folder}")
        images, _ = process_images(filepaths, target_size)
        if images:
            all_processed_images.extend(images)
            for img in images:
                global_min = min(global_min, img.min())
                global_max = max(global_max, img.max())

    if not all_processed_images:
        print("No valid images to process.")
        return

    print(f"\nGlobal statistics:")
    print(f"Global min: {global_min}, Global max: {global_max}")
    print(f"Total images: {len(all_processed_images)}")

    # Generate individual folder plots
    print("\nGenerating individual folder plots...")
    all_lines = []
    all_colors = []
    
    for folder_idx, (folder, filepaths) in enumerate(folder_images.items()):
        print(f"\nProcessing folder: {folder}")
        images, image_shape = process_images(filepaths, target_size)
        
        if not images:
            print(f"No valid images in folder {folder}")
            continue

        # Generate color for this folder
        color = plt.cm.tab10(folder_idx % 10)  # Use tab10 colormap for distinct colors
        
        # Generate individual folder plot
        lines, colors = prepare_data_with_offsets(images, image_shape, global_min, global_max, color)
        output_path = os.path.join(script_dir, f"plot_{folder}.png")
        plot_and_save_parallel_coordinates(
            lines, colors, image_shape[0], image_shape[1], output_path
        )
        
        # Store for combined plot
        all_lines.extend(lines)
        all_colors.extend(colors)

    # Generate combined plot
    print("\nGenerating combined plot...")
    output_path = os.path.join(script_dir, "plot_combined.png")
    plot_and_save_parallel_coordinates(
        all_lines, all_colors, image_shape[0], image_shape[1], output_path
    )

if __name__ == "__main__":
    main()
