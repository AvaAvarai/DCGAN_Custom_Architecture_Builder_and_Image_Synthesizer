# DCGAN Custom Architecture Builder and Synthetic Image Generator

This project is a DCGAN (Deep Convolutional Generative Adversarial Network) custom architecture builder and image synthesizer. It allows the user to specify the architecture of the generator and discriminator, visualize the models, train the GAN, and synthesize images. This allows for dynamic experimentation with the architecture of the generator and discriminator because as detailed in the literature [1] and [2] the architecture of the generator and discriminator impacts the performance of the GAN.  

The user interface is built in Python using Tkinter, and the models are built using TensorFlow and Keras the diagrams are visualized with visualkeras and tensorflow keras utils.  

![ui screenshot](./image.png)

## Project Setup

Currently, the project is a single python file, and the dependencies are:

```sh
pip install numpy matplotlib tensorflow keras visualkeras pillow pydot
```

## Project Execution

```sh
python main.py
```

## Ground Truth MNIST Sevens

We will be using the MNIST dataset of handwritten digits, for training open character recognition models. We specifically use the sevens for initial experimentation.  

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

We trained the custom DCGAN model on the MNIST 7's train data for 30 epochs with loss values of batch=4100, d_loss=0.9101, g_loss=1.2164, and generated 6 images.  

This is the first experiment with custom architecture parameters.  

Experiment architecture parameters:  
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

### Architecture Block Diagrams for the Second Experiment

#### Second Experiment Generator Block Diagram

![Generator Architecture](./synthetic_sevens_second_experiment/dcgan_generator_blockdiagram.png)

#### Second Experiment Discriminator Block Diagram

![Discriminator Architecture](./synthetic_sevens_second_experiment/dcgan_discriminator_blockdiagram.png)

## Third Experiment

We trained the custom DCGAN model on the MNIST 7's train data for 10 epochs with loss values of batch=1300, d_loss=1.3237, g_loss=0.7212, and generated 10 images.  

Using the following architecture parameters:
Generator: 1024,4,1; 512,4,2; 256,4,2; 128,4,2; 3,4,2
Discriminator: 64,4,2; 128,4,2; 256,4,2; 512,4,2

This is the first experiment with dynamic starting spatial dimensions allowing for this architecture to be used for any image size.

![Generated Image 0](./synthetic_sevens_third_experiment/generated_0.png)
![Generated Image 1](./synthetic_sevens_third_experiment/generated_1.png)
![Generated Image 2](./synthetic_sevens_third_experiment/generated_2.png)
![Generated Image 3](./synthetic_sevens_third_experiment/generated_3.png)
![Generated Image 4](./synthetic_sevens_third_experiment/generated_4.png)
![Generated Image 5](./synthetic_sevens_third_experiment/generated_5.png)
![Generated Image 6](./synthetic_sevens_third_experiment/generated_6.png)
![Generated Image 7](./synthetic_sevens_third_experiment/generated_7.png)
![Generated Image 8](./synthetic_sevens_third_experiment/generated_8.png)
![Generated Image 9](./synthetic_sevens_third_experiment/generated_9.png)

### Architecture Block Diagrams for the Third Experiment

#### Third Experiment Generator Block Diagram

![Generator Architecture](./synthetic_sevens_third_experiment/dcgan_generator_blockdiagram.png)

#### Third Experiment Discriminator Block Diagram

![Discriminator Architecture](./synthetic_sevens_third_experiment/dcgan_discriminator_blockdiagram.png)

## Fourth Experiment

This is the first experiment with grayscale image input, previously the images were processed as RGB.  

2 epochs of training with the following loss values:
Not grayscale batch=200, d_loss=1.3812, g_loss=0.7681
Grayscale batch=200, d_loss=1.3975, g_loss=0.7304

Architecture parameters:
Generator: 1024,4,1; 512,4,2; 256,4,2; 128,4,2; 3,4,2
Discriminator: 64,4,2; 128,4,2; 256,4,2; 512,4,2

Latent dimension: 100
Generated images: 12

Processed as RGB:  
![Generated Image 0](./synthetic_sevens_fourth_experiment/not_grayscale/generated_0.png)
![Generated Image 1](./synthetic_sevens_fourth_experiment/not_grayscale/generated_1.png)
![Generated Image 2](./synthetic_sevens_fourth_experiment/not_grayscale/generated_2.png)
![Generated Image 3](./synthetic_sevens_fourth_experiment/not_grayscale/generated_3.png)
![Generated Image 4](./synthetic_sevens_fourth_experiment/not_grayscale/generated_4.png)
![Generated Image 5](./synthetic_sevens_fourth_experiment/not_grayscale/generated_5.png)
![Generated Image 6](./synthetic_sevens_fourth_experiment/not_grayscale/generated_6.png)
![Generated Image 7](./synthetic_sevens_fourth_experiment/not_grayscale/generated_7.png)
![Generated Image 8](./synthetic_sevens_fourth_experiment/not_grayscale/generated_8.png)
![Generated Image 9](./synthetic_sevens_fourth_experiment/not_grayscale/generated_9.png)
![Generated Image 10](./synthetic_sevens_fourth_experiment/not_grayscale/generated_10.png)
![Generated Image 11](./synthetic_sevens_fourth_experiment/not_grayscale/generated_11.png)

Processed as grayscale:  
![Generated Image 0](./synthetic_sevens_fourth_experiment/grayscale/generated_0.png)
![Generated Image 1](./synthetic_sevens_fourth_experiment/grayscale/generated_1.png)
![Generated Image 2](./synthetic_sevens_fourth_experiment/grayscale/generated_2.png)
![Generated Image 3](./synthetic_sevens_fourth_experiment/grayscale/generated_3.png)
![Generated Image 4](./synthetic_sevens_fourth_experiment/grayscale/generated_4.png)
![Generated Image 5](./synthetic_sevens_fourth_experiment/grayscale/generated_5.png)
![Generated Image 6](./synthetic_sevens_fourth_experiment/grayscale/generated_6.png)
![Generated Image 7](./synthetic_sevens_fourth_experiment/grayscale/generated_7.png)
![Generated Image 8](./synthetic_sevens_fourth_experiment/grayscale/generated_8.png)
![Generated Image 9](./synthetic_sevens_fourth_experiment/grayscale/generated_9.png)
![Generated Image 10](./synthetic_sevens_fourth_experiment/grayscale/generated_10.png)
![Generated Image 11](./synthetic_sevens_fourth_experiment/grayscale/generated_11.png)

## Todo

- [ ] Add graph visualization of the generator and discriminator loss values over training epochs.

## Referenced Citations

[1] Radford, A. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.

[2] S. Vijaya Lakshmi, Vallik Sai Ganesh Raju Ganaraju, “Deep Convolutional Generative Adversial Network on
MNIST Dataset”, Journal of Science and Technology, Vol. 06, Issue 03, May-June 2021, pp169-177

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
