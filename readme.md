# Exploring the Evolution of Generative Adversarial Networks
This repository contains the code and results of our project exploring the evolution of Generative Adversarial Networks (GANs). The project was conducted as part of a research effort at Duke University, Department of Electrical and Computer Engineering.

## Abstract
The paper explores various improvements and variants of the original GAN proposed by Ian Goodfellow. Specifically, it investigates the Wasserstein GAN (WGAN), the Auxiliary Classifier GAN (AC-GAN), and a modified Deep Convolutional GAN (DCGAN). The study includes qualitative analysis through random sampling and latent space interpolation, as well as quantitative metrics including Inception Score (IS), Fréchet Inception Distance (FID), Kernel Inception Distance (KID), and Perceptual Path Length (PPL) to evaluate the performance of each variant. Experimental results indicate that AC-GAN and DCGAN produce the most photo-realistic images, while WGAN achieves a more stable training process.

## Installation
To run the code in this repository, follow these steps:

1. Clone the repository: `git clone https://github.com/nrgbistro/GAN-Submission.git`
2. Install the required dependencies: `pip install -r requirements.txt`
2. Run the cells in the desired notebook file in the `/src` directory

## Results
The repository includes scripts for training and evaluating each GAN variant, as well as scripts for visualizing the generated images and analyzing the results. The results show the performance of each model in terms of image quality and training stability, providing insights into the evolution of GANs.

## License
This project is licensed under the MIT License. See the LICENSE file for details.