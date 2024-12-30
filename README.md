# DCGAN Custom Architecture Builder and Synthetic Image Generator

This project is a DCGAN (Deep Convolutional Generative Adversarial Network) custom architecture builder and synthetic image generator.
It allows the user to specify the architecture of the generator and discriminator, visualize the models, train the GAN, and generate synthetic images.

The user interface is built using Tkinter, and the models are built using TensorFlow and Keras.

## First Experiment

We trained the custom DCGAN model on the MNIST 7's train data for 10 epochs with loss values of batch=1300, d_loss=1.2257, g_loss=0.9160, and generated 5 images.

![Generated Image 1](./synthetic_sevens_first_experiment/generated_0.png)
![Generated Image 2](./synthetic_sevens_first_experiment/generated_1.png)
![Generated Image 3](./synthetic_sevens_first_experiment/generated_2.png)
![Generated Image 4](./synthetic_sevens_first_experiment/generated_3.png)
![Generated Image 5](./synthetic_sevens_first_experiment/generated_4.png)

## Second Experiment

Epoch 30/30 batch=4100, d_loss=0.9101, g_loss=1.2164

![Generated Image 0](./synthetic_sevens_second_experiment/generated_0.png)
![Generated Image 1](./synthetic_sevens_second_experiment/generated_1.png)
![Generated Image 2](./synthetic_sevens_second_experiment/generated_2.png)
![Generated Image 3](./synthetic_sevens_second_experiment/generated_3.png)
![Generated Image 4](./synthetic_sevens_second_experiment/generated_4.png)
![Generated Image 5](./synthetic_sevens_second_experiment/generated_5.png)

Architecture block diagrams:

Generator:
![Generator Architecture](./synthetic_sevens_second_experiment/dcgan_generator_blockdiagram.png)

Discriminator:
![Discriminator Architecture](./synthetic_sevens_second_experiment/dcgan_discriminator_blockdiagram.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
