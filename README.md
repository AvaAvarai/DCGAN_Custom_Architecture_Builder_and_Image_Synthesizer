# DCGAN Custom Architecture Builder and Synthetic Image Generator

This project is a DCGAN (Deep Convolutional Generative Adversarial Network) custom architecture builder and image synthesizer.  
It allows the user to specify the architecture of the generator and discriminator, visualize the models, train the GAN, and synthesize images.  

The user interface is built in Python using Tkinter, and the models are built using TensorFlow and Keras the diagrams are visualized with visualkeras and tensorflow keras utils.

![ui screenshot](./image.png)

We will be using the MNIST dataset of handwritten digits, specifically the sevens for experimentation.  

## Ground Truth MNIST Sevens

Ten of the 4401 MNIST sevens train data images are shown below.

![mnist seven 0](./real_mnist_sevens_train_data/img_6.jpg)
![mnist seven 1](./real_mnist_sevens_train_data/img_18.jpg)
![mnist seven 2](./real_mnist_sevens_train_data/img_29.jpg)
![mnist seven 3](./real_mnist_sevens_train_data/img_47.jpg)
![mnist seven 4](./real_mnist_sevens_train_data/img_48.jpg)
![mnist seven 5](./real_mnist_sevens_train_data/img_50.jpg)
![mnist seven 6](./real_mnist_sevens_train_data/img_76.jpg)
![mnist seven 7](./real_mnist_sevens_train_data/img_102.jpg)
![mnist seven 8](./real_mnist_sevens_train_data/img_103.jpg)
![mnist seven 9](./real_mnist_sevens_train_data/img_116.jpg)

## First Experiment

We trained the custom DCGAN model on the MNIST 7's train data for 10 epochs with loss values of batch=1300, d_loss=1.2257, g_loss=0.9160, and generated 5 images.

![Generated Image 1](./synthetic_sevens_first_experiment/generated_0.png)
![Generated Image 2](./synthetic_sevens_first_experiment/generated_1.png)
![Generated Image 3](./synthetic_sevens_first_experiment/generated_2.png)
![Generated Image 4](./synthetic_sevens_first_experiment/generated_3.png)
![Generated Image 5](./synthetic_sevens_first_experiment/generated_4.png)

## Second Experiment

Training data: MNIST train set digit sevens.  
Epochs: 30  
Latent Dim: 100  
Generator: 1024,4,1; 512,5,2; 256,5,2; 128,5,2; 3,5,2  
Discriminator: 64,4,2; 128,4,2; 256,4,2; 512,4,2  
Resultant loss values: batch=4100, d_loss=0.9101, g_loss=1.2164  

![Generated Image 0](./synthetic_sevens_second_experiment/generated_0.png)
![Generated Image 1](./synthetic_sevens_second_experiment/generated_1.png)
![Generated Image 2](./synthetic_sevens_second_experiment/generated_2.png)
![Generated Image 3](./synthetic_sevens_second_experiment/generated_3.png)
![Generated Image 4](./synthetic_sevens_second_experiment/generated_4.png)
![Generated Image 5](./synthetic_sevens_second_experiment/generated_5.png)

### Architecture Block Diagrams

#### Generator

![Generator Architecture](./synthetic_sevens_second_experiment/dcgan_generator_blockdiagram.png)

#### Discriminator

![Discriminator Architecture](./synthetic_sevens_second_experiment/dcgan_discriminator_blockdiagram.png)

## Referenced Citations

- Radford, A. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.

- S. Vijaya Lakshmi, Vallik Sai Ganesh Raju Ganaraju, “Deep Convolutional Generative Adversial Network on
MNIST Dataset”, Journal of Science and Technology, Vol. 06, Issue 03, May-June 2021, pp169-177

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
