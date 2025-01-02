import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tkinter import filedialog, Tk, Label, Entry, Button

# For diagram visualizations
import visualkeras
from PIL import ImageFont
from tensorflow.keras.utils import plot_model


# ------------------------ Parsing and Building Parametric Models ------------------------ #

def parse_layer_specs(spec_str):
    """
    Parse a semicolon-separated list of layer specs into a list of tuples.
    Example input: "512,5,2; 256,5,2; 128,5,2; 3,5,2"
    Returns: [(512, 5, 2), (256, 5, 2), (128, 5, 2), (3,5,2)]
    """
    spec_str = spec_str.strip()
    if not spec_str:
        return []
    
    layer_specs = []
    for part in spec_str.split(';'):
        part = part.strip()
        if not part:
            continue
        vals = part.split(',')
        if len(vals) != 3:
            raise ValueError(f"Invalid layer spec: '{part}'. Must have 3 comma-separated values.")
        out_channels = int(vals[0].strip())
        kernel_size = int(vals[1].strip())
        stride = int(vals[2].strip())
        layer_specs.append((out_channels, kernel_size, stride))
    
    return layer_specs


def build_param_generator(latent_dim, layer_specs, start_spatial=4, grayscale=False):
    """
    Build a DCGAN-style generator, dynamically adjusting the final layer for grayscale or RGB output.
    """
    if not layer_specs:
        raise ValueError("Generator layer specs cannot be empty.")

    output_channels = 1 if grayscale else 3  # Adjust output channels for grayscale or RGB
    model = keras.Sequential(name="generator")

    # Input layer for latent vector
    model.add(keras.Input(shape=(latent_dim,)))  
    
    # Projection layer
    out_channels_0, _, _ = layer_specs[0]
    model.add(layers.Dense(start_spatial * start_spatial * out_channels_0, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Reshape((start_spatial, start_spatial, out_channels_0)))

    # Intermediate upsampling layers
    for out_ch, k, s in layer_specs[1:-1]:
        model.add(layers.Conv2DTranspose(out_ch, k, strides=s, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(negative_slope=0.2))

    # Final layer
    out_channels_last, kernel_last, stride_last = layer_specs[-1]
    model.add(layers.Conv2DTranspose(
        filters=output_channels,  # 1 for grayscale, 3 for RGB
        kernel_size=kernel_last,
        strides=stride_last,
        padding='same',
        use_bias=False,
        activation='tanh'
    ))

    return model


def build_param_discriminator(layer_specs, grayscale=False):
    """
    Build a DCGAN-style discriminator, dynamically adjusting the input layer for grayscale or RGB input.
    """
    if not layer_specs:
        raise ValueError("Discriminator layer specs cannot be empty.")

    input_channels = 1 if grayscale else 3  # Adjust input channels for grayscale or RGB
    model = keras.Sequential(name="discriminator")

    # Input layer
    model.add(keras.Input(shape=(64, 64, input_channels)))  
    
    # First conv layer
    out_channels_0, kernel_0, stride_0 = layer_specs[0]
    model.add(layers.Conv2D(out_channels_0, kernel_0, strides=stride_0, padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    # Intermediate downsampling layers
    for out_ch, k, s in layer_specs[1:]:
        model.add(layers.Conv2D(out_ch, k, strides=s, padding='same'))
        model.add(layers.LeakyReLU(negative_slope=0.2))
        model.add(layers.Dropout(0.3))

    # Output layer
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


# ------------------------ Data Loading / Training / Generation ------------------------ #
def load_images_from_folder(folder, image_size=(64, 64), batch_size=32, grayscale=False):
    """
    Load images from a folder, processing them in grayscale or RGB mode based on the `grayscale` flag.
    """
    all_images = []
    file_extensions = ('*.png', '*.jpg', '*.jpeg')
    color_mode = "grayscale" if grayscale else "rgb"  # Set color mode

    for ext in file_extensions:
        for file in glob.glob(os.path.join(folder, ext)):
            img = keras.preprocessing.image.load_img(file, target_size=image_size, color_mode=color_mode)
            img = keras.preprocessing.image.img_to_array(img)
            all_images.append(img)

    if not all_images:
        raise ValueError("No images found in the specified folder.")

    all_images = np.array(all_images, dtype=np.float32)
    all_images = (all_images - 127.5) / 127.5  # Normalize to [-1, 1]

    # Ensure grayscale images have shape (H, W, 1)
    if grayscale and all_images.shape[-1] != 1:
        all_images = all_images[..., np.newaxis]  # Add channel only if not already present

    dataset = tf.data.Dataset.from_tensor_slices(all_images)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    
    # Expected shape: (num_images, 64, 64, 1) for grayscale, (num_images, 64, 64, 3) for RGB
    # If shapes mismatch, print the shape of the dataset to debug
    # print(f"Dataset shape: {all_images.shape}")  # Debugging

    return dataset


def gan_train_step(real_images, generator, discriminator,
                   g_optimizer, d_optimizer, latent_dim):
    """
    One step of training for both discriminator and generator using GradientTape.
    No @tf.function so we remain in pure eager mode.
    """
    batch_size = tf.shape(real_images)[0]

    # Labels
    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))

    # 1) Train Discriminator
    with tf.GradientTape() as tape_d:
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        fake_images = generator(random_latent_vectors, training=True)

        real_preds = discriminator(real_images, training=True)
        fake_preds = discriminator(fake_images, training=True)

        d_loss_real = keras.losses.binary_crossentropy(real_labels, real_preds)
        d_loss_fake = keras.losses.binary_crossentropy(fake_labels, fake_preds)
        d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

    # 2) Train Generator
    with tf.GradientTape() as tape_g:
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        generated_images = generator(random_latent_vectors, training=True)
        fake_preds_for_g = discriminator(generated_images, training=True)

        # We want to fool the discriminator -> label=1
        g_loss = keras.losses.binary_crossentropy(tf.ones((batch_size, 1)), fake_preds_for_g)
        g_loss = tf.reduce_mean(g_loss)

    grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

    return d_loss, g_loss


def train_gan(generator, discriminator, dataset,
              latent_dim=100, epochs=10, print_interval=100,
              g_learning_rate=0.0002, d_learning_rate=0.0002):
    """Train the GAN for a specified number of epochs using a custom loop."""
    g_optimizer = keras.optimizers.Adam(learning_rate=g_learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=d_learning_rate, beta_1=0.5)

    batch_count = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for real_images in dataset:
            d_loss, g_loss = gan_train_step(
                real_images, generator, discriminator,
                g_optimizer, d_optimizer, latent_dim
            )
            if batch_count % print_interval == 0:
                print(f"  batch={batch_count}, d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
            batch_count += 1


def generate_and_save_images(generator, latent_dim, num_images, output_folder):
    """Use the trained generator to create images and save them."""
    os.makedirs(output_folder, exist_ok=True)
    random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))
    generated_images = generator(random_latent_vectors, training=False)
    generated_images = (generated_images * 127.5) + 127.5  # [-1,1] -> [0,255]

    for i in range(num_images):
        img_array = generated_images[i].numpy().astype(np.uint8)
        if img_array.shape[-1] == 1:  # Grayscale
            img_array = img_array.squeeze(-1)  # Remove the channel dimension
        plt.imsave(os.path.join(output_folder, f"generated_{i}.png"), img_array, cmap="gray" if img_array.ndim == 2 else None)


# ------------------------ Visualization Helpers ------------------------ #

def visualize_models_keras_plot(generator, discriminator, output_folder):
    """
    Save standard Keras model flowchart diagrams. (flowchart style)
    Requires graphviz + pydot
    """
    from tensorflow.keras.utils import plot_model
    os.makedirs(output_folder, exist_ok=True)

    gen_path = os.path.join(output_folder, "dcgan_generator_flowchart.png")
    disc_path = os.path.join(output_folder, "dcgan_discriminator_flowchart.png")

    plot_model(generator, to_file=gen_path, show_shapes=True, show_layer_names=True)
    print(f"[flowchart] Generator diagram saved to: {gen_path}")

    plot_model(discriminator, to_file=disc_path, show_shapes=True, show_layer_names=True)
    print(f"[flowchart] Discriminator diagram saved to: {disc_path}")


def visualize_models_visualkeras(generator, discriminator, output_folder):
    """
    Save layered block diagrams using `visualkeras.layered_view`.
    This approach yields 'diagrams', not 'flowcharts'.
    """
    import visualkeras
    from PIL import ImageFont
    os.makedirs(output_folder, exist_ok=True)

    gen_diagram = visualkeras.layered_view(
        generator, 
        legend=True, 
        spacing=40,      
        draw_volume=False
    )
    gen_path = os.path.join(output_folder, "dcgan_generator_blockdiagram.png")
    gen_diagram.save(gen_path)
    print(f"[visualkeras] Generator layered diagram saved to: {gen_path}")

    disc_diagram = visualkeras.layered_view(
        discriminator, 
        legend=True, 
        spacing=40,
        draw_volume=False
    )
    disc_path = os.path.join(output_folder, "dcgan_discriminator_blockdiagram.png")
    disc_diagram.save(disc_path)
    print(f"[visualkeras] Discriminator layered diagram saved to: {disc_path}")


# ------------------------ GUI / Buttons ------------------------ #

def select_folder(entry_widget):
    """Open a dialog to select a folder and place its path into the given entry widget."""
    folder = filedialog.askdirectory()
    if folder:
        entry_widget.delete(0, "end")
        entry_widget.insert(0, folder)


def main():
    root = Tk()
    root.title("DCGAN Custom Architecture Builder and Synthetic Image Generator")

    # Row 0: Training folder
    Label(root, text="Training Folder:").grid(row=0, column=0, padx=5, pady=5)
    train_entry = Entry(root, width=50)
    train_entry.grid(row=0, column=1, padx=5, pady=5)
    Button(root, text="Browse", command=lambda: select_folder(train_entry)).grid(row=0, column=2, padx=5, pady=5)

    # Row 1: Output folder
    Label(root, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5)
    output_entry = Entry(root, width=50)
    output_entry.grid(row=1, column=1, padx=5, pady=5)
    Button(root, text="Browse", command=lambda: select_folder(output_entry)).grid(row=1, column=2, padx=5, pady=5)

    # Row 2: Epochs
    Label(root, text="Epochs:").grid(row=2, column=0, padx=5, pady=5)
    epochs_entry = Entry(root, width=10)
    epochs_entry.insert(0, "10")
    epochs_entry.grid(row=2, column=1, padx=5, pady=5)

    # Row 3: Number of Images to Generate
    Label(root, text="Number of Images:").grid(row=3, column=0, padx=5, pady=5)
    num_images_entry = Entry(root, width=10)
    num_images_entry.insert(0, "5")
    num_images_entry.grid(row=3, column=1, padx=5, pady=5)

    # Row 4: Latent Dim
    Label(root, text="Latent Dim:").grid(row=4, column=0, padx=5, pady=5)
    latent_entry = Entry(root, width=10)
    latent_entry.insert(0, "100")
    latent_entry.grid(row=4, column=1, padx=5, pady=5)

    # Row 5: Generator Specs
    Label(root, text="Generator Specs:\n(out_ch, k, stride ; ... )").grid(row=5, column=0, padx=5, pady=5)
    gen_entry = Entry(root, width=50)
    gen_entry.insert(0, "1024,4,1; 512,4,2; 256,4,2; 128,4,2; 3,4,2")
    gen_entry.grid(row=5, column=1, padx=5, pady=5, columnspan=2)

    # Row 6: Discriminator Specs
    Label(root, text="Discriminator Specs:\n(out_ch, k, stride ; ... )").grid(row=6, column=0, padx=5, pady=5)
    disc_entry = Entry(root, width=50)
    disc_entry.insert(0, "64,4,2; 128,4,2; 256,4,2; 512,4,2")
    disc_entry.grid(row=6, column=1, padx=5, pady=5, columnspan=2)

    # Row 7: Grayscale toggle
    Label(root, text="Grayscale? (yes/no):").grid(row=7, column=0, padx=5, pady=5)
    grayscale_entry = Entry(root, width=10)
    grayscale_entry.insert(0, "no")  # Default to RGB
    grayscale_entry.grid(row=7, column=1, padx=5, pady=5)

    # Row 8: Generator Learning Rate
    Label(root, text="Generator Learning Rate:").grid(row=8, column=0, padx=5, pady=5)
    g_lr_entry = Entry(root, width=10)
    g_lr_entry.insert(0, "0.0002")  # Default value
    g_lr_entry.grid(row=8, column=1, padx=5, pady=5)

    # Row 9: Discriminator Learning Rate
    Label(root, text="Discriminator Learning Rate:").grid(row=9, column=0, padx=5, pady=5)
    d_lr_entry = Entry(root, width=10)
    d_lr_entry.insert(0, "0.0002")  # Default value
    d_lr_entry.grid(row=9, column=1, padx=5, pady=5)

    def on_visualize_flowchart():
        try:
            output_folder = output_entry.get()
            latent_dim = int(latent_entry.get())
            gen_specs = parse_layer_specs(gen_entry.get())
            disc_specs = parse_layer_specs(disc_entry.get())
            grayscale = (grayscale_entry.get().strip().lower() == "yes")

            generator = build_param_generator(latent_dim, gen_specs, start_spatial=4, grayscale=grayscale)
            discriminator = build_param_discriminator(disc_specs, grayscale=grayscale)

            visualize_models_keras_plot(generator, discriminator, output_folder)
        except Exception as e:
            print(f"Flowchart visualization error: {e}")

    def on_visualize_layered():
        try:
            output_folder = output_entry.get()
            latent_dim = int(latent_entry.get())
            gen_specs = parse_layer_specs(gen_entry.get())
            disc_specs = parse_layer_specs(disc_entry.get())
            grayscale = (grayscale_entry.get().strip().lower() == "yes")

            generator = build_param_generator(latent_dim, gen_specs, start_spatial=4, grayscale=grayscale)
            discriminator = build_param_discriminator(disc_specs, grayscale=grayscale)

            visualize_models_visualkeras(generator, discriminator, output_folder)
        except Exception as e:
            print(f"Layered (visualkeras) visualization error: {e}")

    def on_run():
        try:
            train_folder = train_entry.get()
            output_folder = output_entry.get()
            epochs = int(epochs_entry.get())
            num_images = int(num_images_entry.get())
            latent_dim = int(latent_entry.get())
            gen_specs = parse_layer_specs(gen_entry.get())
            disc_specs = parse_layer_specs(disc_entry.get())
            grayscale = (grayscale_entry.get().strip().lower() == "yes")
            
            # Get learning rates from GUI
            g_learning_rate = float(g_lr_entry.get())
            d_learning_rate = float(d_lr_entry.get())

            dataset = load_images_from_folder(train_folder, image_size=(64, 64), grayscale=grayscale)

            generator = build_param_generator(latent_dim, gen_specs, start_spatial=4, grayscale=grayscale)
            discriminator = build_param_discriminator(disc_specs, grayscale=grayscale)

            train_gan(generator, discriminator, dataset, 
                     latent_dim=latent_dim, 
                     epochs=epochs,
                     g_learning_rate=g_learning_rate,
                     d_learning_rate=d_learning_rate)
            generate_and_save_images(generator, latent_dim, num_images, output_folder)
            print(f"Data generation complete! Check '{output_folder}' for generated images.")
        except Exception as e:
            print(f"Training/Generation error: {e}")

    # Row 10: Place Buttons (moved down one row)
    Button(root, text="Viz Flowchart", command=on_visualize_flowchart).grid(row=10, column=0, pady=10)
    Button(root, text="Viz Layered", command=on_visualize_layered).grid(row=10, column=1, pady=10)
    Button(root, text="Train & Generate", command=on_run).grid(row=10, column=2, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
