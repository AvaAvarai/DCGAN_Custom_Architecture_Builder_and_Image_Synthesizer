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
def prepare_data_with_offsets(images, image_shape, global_min, global_max):
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
                print(f"Image {filepath} skipped due to size mismatch.")
                skipped_count += 1
                continue
            images.append(img_array)
            processed_count += 1
            print(f"Processed {filepath}")
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            error_count += 1
    
    print(f"\nProcessing summary:")
    print(f"Successfully processed: {processed_count} images")
    print(f"Skipped due to size mismatch: {skipped_count} images")
    print(f"Failed to process: {error_count} images")
    print(f"Total attempted: {processed_count + skipped_count + error_count} images\n")
    
    return images, image_shape

# Main program
def main():
    # Input: Directory containing images and target size
    root_dir = input("Enter the directory containing images: ")
    print(f"Looking in directory: {root_dir}")
    target_size = tuple(map(int, input("Enter the target image size (width, height): ").split(',')))

    # Find all images recursively
    print("Searching for images recursively...")
    image_files = find_all_images(root_dir)
    
    if not image_files:
        print("No images found in the specified directory or its subdirectories!")
        return
    
    print(f"Found {len(image_files)} image files")

    # First pass: Calculate global min and max
    global_min, global_max = float("inf"), float("-inf")
    print("\nFirst pass: calculating global min and max...")
    images, _ = process_images(image_files, target_size)
    
    if not images:
        print("No valid images to process.")
        return

    print(f"Using {len(images)} images for visualization")
    
    for img in images:
        global_min = min(global_min, img.min())
        global_max = max(global_max, img.max())

    print(f"Global min: {global_min}, Global max: {global_max}")

    # Second pass: Generate plots
    print("\nSecond pass: generating plots...")
    images, image_shape = process_images(image_files, target_size)
    
    if not images:
        print("No valid images to process.")
        return

    print(f"\nGenerating plots for {len(images)} images...")

    # Generate plot
    print("Generating visualization plot...")
    lines, colors = prepare_data_with_offsets(images, image_shape, global_min, global_max)
    output_path = os.path.join(root_dir, "all_images_plot.png")
    plot_and_save_parallel_coordinates(
        lines, colors, image_shape[0], image_shape[1], output_path
    )

if __name__ == "__main__":
    main()
