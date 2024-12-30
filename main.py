import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tkinter import filedialog, Tk, Label, Entry, Button


# ------------------------ Functional Components ------------------------ #

def build_generator(latent_dim=100):
    """Build a simple DCGAN generator model."""
    model = keras.Sequential(name="generator")
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    # Upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 32x32
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2),
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 64x64
    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2),
                                     padding='same', use_bias=False,
                                     activation='tanh'))
    return model


def build_discriminator():
    """Build a simple DCGAN discriminator model."""
    model = keras.Sequential(name="discriminator")
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                            input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def load_images_from_folder(folder, image_size=(64, 64), batch_size=32):
    """Load all images from a given folder into a tf.data.Dataset."""
    all_images = []
    file_extensions = ('*.png', '*.jpg', '*.jpeg')

    for ext in file_extensions:
        for file in glob.glob(os.path.join(folder, ext)):
            img = tf.keras.preprocessing.image.load_img(file,
                                                        target_size=image_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            all_images.append(img)

    if not all_images:
        raise ValueError("No images found in the specified folder.")

    # Convert to array and scale to [-1, 1] range
    all_images = np.array(all_images, dtype=np.float32)
    all_images = (all_images - 127.5) / 127.5

    # Build dataset
    dataset = tf.data.Dataset.from_tensor_slices(all_images)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    return dataset


@tf.function
def gan_train_step(real_images, generator, discriminator,
                   g_optimizer, d_optimizer, latent_dim):
    """One step of training for both discriminator and generator via GradientTape."""
    batch_size = tf.shape(real_images)[0]

    # Labels
    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))

    # 1) Train Discriminator
    with tf.GradientTape() as tape_d:
        # Generate fake images
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        fake_images = generator(random_latent_vectors, training=True)

        # Discriminator output
        real_preds = discriminator(real_images, training=True)
        fake_preds = discriminator(fake_images, training=True)

        # Discriminator loss: average cross-entropy
        d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_preds)
        d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_preds)
        d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))

    # 2) Train Generator
    with tf.GradientTape() as tape_g:
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        generated_images = generator(random_latent_vectors, training=True)
        fake_preds_for_g = discriminator(generated_images, training=True)

        # We want to fool the discriminator -> label = 1
        g_loss = tf.keras.losses.binary_crossentropy(tf.ones((batch_size, 1)), 
                                                     fake_preds_for_g)
        g_loss = tf.reduce_mean(g_loss)

    grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))

    return d_loss, g_loss


def train_gan(generator, discriminator, dataset,
              latent_dim=100, epochs=10, print_interval=100):
    """Train the GAN for a specified number of epochs using a custom training loop."""
    # Define separate optimizers for G and D
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    batch_count = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for real_images in dataset:
            d_loss, g_loss = gan_train_step(real_images, generator, discriminator,
                                            g_optimizer, d_optimizer, latent_dim)

            if batch_count % print_interval == 0:
                print(f"  batch={batch_count}, d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
            batch_count += 1


def generate_and_save_images(generator, latent_dim, num_images, output_folder):
    """Use the trained generator to create images and save them."""
    os.makedirs(output_folder, exist_ok=True)
    random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))
    generated_images = generator(random_latent_vectors, training=False)
    generated_images = (generated_images * 127.5) + 127.5  # Scale back to [0, 255]

    for i in range(num_images):
        img_array = generated_images[i].numpy().astype(np.uint8)
        plt.imsave(os.path.join(output_folder, f"generated_{i}.png"), img_array)


# ------------------------ GUI Components ------------------------ #

def run_gan_pipeline(train_folder, output_folder, epochs, num_images):
    """Encapsulate the end-to-end GAN flow given GUI inputs."""
    # Hyperparameters
    latent_dim = 100
    batch_size = 32
    
    # 1. Load dataset
    dataset = load_images_from_folder(train_folder, (64, 64), batch_size)
    
    # 2. Build models
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    
    # 3. Train using custom training loop
    train_gan(generator, discriminator, dataset,
              latent_dim=latent_dim, epochs=epochs, print_interval=100)
    
    # 4. Generate and save
    generate_and_save_images(generator, latent_dim, num_images, output_folder)
    print(f"Data generation complete! Images saved to: {output_folder}")


def select_folder(entry_widget):
    """Open a dialog to select a folder and place its path into the given entry widget."""
    folder = filedialog.askdirectory()
    if folder:
        entry_widget.delete(0, "end")
        entry_widget.insert(0, folder)


def main():
    root = Tk()
    root.title("DCGAN Data Synthesis (Custom Training Loop)")

    # Training folder
    Label(root, text="Training Folder:").grid(row=0, column=0, padx=5, pady=5)
    train_entry = Entry(root, width=50)
    train_entry.grid(row=0, column=1, padx=5, pady=5)
    Button(root, text="Browse", command=lambda: select_folder(train_entry)).grid(row=0, column=2, padx=5, pady=5)

    # Output folder
    Label(root, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5)
    output_entry = Entry(root, width=50)
    output_entry.grid(row=1, column=1, padx=5, pady=5)
    Button(root, text="Browse", command=lambda: select_folder(output_entry)).grid(row=1, column=2, padx=5, pady=5)

    # Number of epochs
    Label(root, text="Epochs:").grid(row=2, column=0, padx=5, pady=5)
    epochs_entry = Entry(root, width=10)
    epochs_entry.insert(0, "10")  # default
    epochs_entry.grid(row=2, column=1, padx=5, pady=5)

    # Number of images to generate
    Label(root, text="Number of Images:").grid(row=3, column=0, padx=5, pady=5)
    num_images_entry = Entry(root, width=10)
    num_images_entry.insert(0, "5")  # default
    num_images_entry.grid(row=3, column=1, padx=5, pady=5)

    # Run button
    def on_run():
        train_folder = train_entry.get()
        output_folder = output_entry.get()
        epochs = int(epochs_entry.get())
        num_images = int(num_images_entry.get())

        run_gan_pipeline(train_folder, output_folder, epochs, num_images)

    Button(root, text="Train & Generate", command=on_run).grid(row=4, column=0, columnspan=3, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
