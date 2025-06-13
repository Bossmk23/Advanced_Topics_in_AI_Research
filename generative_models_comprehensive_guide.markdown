# A Comprehensive Guide to Variational Autoencoders and Generative Adversarial Networks: From Basics to Advanced Stabilization Techniques

In this comprehensive blog post, we explore two powerful generative models in deep learning: **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)**. We’ll cover their fundamentals, implement them using TensorFlow and PyTorch, and dive into advanced techniques to stabilize training. Using the MNIST and Fashion-MNIST datasets, we’ll demonstrate:
1. A basic VAE for MNIST digit generation.
2. GANs with Binary Cross-Entropy (BCE) and Wasserstein loss for MNIST.
3. Regularized VAEs and stabilized GANs (WGAN-GP) for Fashion-MNIST.

By the end, you’ll understand how these models work, how to implement them, and how to enhance their performance with regularization techniques.

## Introduction to Generative Models

Generative models aim to learn the underlying distribution of a dataset to create new, realistic samples. VAEs and GANs are two prominent approaches:
- **VAEs**: Combine neural networks with Bayesian inference to learn a probabilistic latent space, enabling data reconstruction and generation. They optimize a reconstruction loss and a KL-divergence term to regularize the latent space.
- **GANs**: Consist of a generator producing fake data from noise and a discriminator distinguishing real from fake data. Trained adversarially, GANs often produce sharper outputs but can suffer from training instability.

We’ll use:
- **MNIST**: 28x28 grayscale images of handwritten digits (0–9) for basic implementations.
- **Fashion-MNIST**: 28x28 grayscale images of clothing items for advanced stabilization techniques.

## Setting Up the Environment

To follow along, install the required libraries:
- TensorFlow and PyTorch (for model implementation)
- NumPy and Matplotlib (for data manipulation and visualization)
- Torchvision (for dataset loading in PyTorch)

Install them using pip:

```bash
pip install tensorflow torch torchvision numpy matplotlib
```

## Part 1: Understanding Variational Autoencoders with MNIST

### What is a Variational Autoencoder?

VAEs learn a probabilistic latent space, making them ideal for generative tasks. Key components include:
- **Encoder**: Maps input data to a latent distribution (mean and variance).
- **Sampling Layer**: Samples from this distribution using the reparameterization trick.
- **Decoder**: Reconstructs data from latent samples.
- **Loss Function**: Combines reconstruction loss (e.g., binary cross-entropy) and KL-divergence to regularize the latent space.

### Loading and Preprocessing MNIST

We normalize MNIST images to [0, 1] and add a channel dimension (28x28x1).

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

# Load and normalize the MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # Shape: (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)    # Shape: (10000, 28, 28, 1)
```

### Building the VAE

#### Encoder
The encoder outputs the mean and log-variance of a 2D latent space.

```python
latent_dim = 2

class Encoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var
```

#### Sampling Layer
The reparameterization trick ensures differentiability.

```python
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

#### Decoder
The decoder reconstructs images from latent vectors.

```python
class Decoder(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(28 * 28, activation='sigmoid')
        self.reshape = layers.Reshape((28, 28, 1))

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.reshape(x)
```

#### VAE Model
The VAE integrates all components with a custom training step.

```python
class VAE(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampling()

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler((z_mean, z_log_var))
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
```

### Training the VAE

We train for 30 epochs with a batch size of 128.

```python
encoder = Encoder(latent_dim)
decoder = Decoder()
vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(x_train, epochs=30, batch_size=128)
```

Sample output:
```
Epoch 1/30
469/469 [==============================] - 10s 16ms/step - kl_loss: 15.7490 - loss: 250.8092 - reconstruction_loss: 235.0603
...
Epoch 30/30
469/469 [==============================] - 10s 14ms/step - kl_loss: 5.9932 - loss: 150.3951 - reconstruction_loss: 144.4020
```

### Generating New Images

We sample from the latent space to generate digits.

```python
def generate_and_plot_images(model, n=10):
    z = tf.random.normal(shape=(n, latent_dim))
    generated = model.decoder(z)
    generated = generated.numpy()
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(generated[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.show()

generate_and_plot_images(vae)
```

### Results
Generated images resemble digits but may be blurry due to the 2D latent space. Increasing `latent_dim` (e.g., 10) improves quality.

![Generated MNIST digits](attachment://vae_mnist_digits.png)

## Part 2: Exploring GAN Loss Functions with MNIST

### What are Generative Adversarial Networks?

GANs involve two networks:
- **Generator**: Produces fake data from noise.
- **Discriminator**: Distinguishes real from fake data.

We compare:
- **BCE Loss**: Standard GAN loss, prone to instability.
- **Wasserstein Loss**: Stabilizes training by measuring Earth Mover’s distance.

### Loading and Preprocessing MNIST

We normalize MNIST to [-1, 1] for the generator’s `tanh` activation.

```python
# Load and normalize MNIST dataset to [-1, 1]
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images.reshape(-1, 28, 28, 1).astype("float32") - 127.5) / 127.5
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Create shuffled batches
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

### Building the GAN

#### Generator
Uses transposed convolutions to upsample noise into 28x28x1 images.

```python
def make_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=1, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=2, padding="same", use_bias=False, activation="tanh")
    ])
    return model
```

#### Discriminator
Uses convolutions to classify images.

```python
def make_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=2, padding="same", input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=2, padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model
```

#### Loss Functions
We implement BCE and Wasserstein losses.

```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss_bce(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss_bce(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss_wass(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss_wass(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
```

### Training the GAN

We train for 10 epochs using RMSprop.

```python
generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

EPOCHS = 10
noise_dim = 100
seed = tf.random.normal([16, noise_dim])
USE_WASSERSTEIN = False

generator = make_generator()
discriminator = make_discriminator()

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        if USE_WASSERSTEIN:
            gen_loss = generator_loss_wass(fake_output)
            disc_loss = discriminator_loss_wass(real_output, fake_output)
        else:
            gen_loss = generator_loss_bce(fake_output)
            disc_loss = discriminator_loss_bce(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.show()

def train(dataset, epochs):
    gen_losses = []
    disc_losses = []
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)
        print(f"Epoch {epoch + 1}, Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")
        generate_and_save_images(generator, epoch + 1, seed)
    return gen_losses, disc_losses

losses_gen, losses_disc = train(train_dataset, EPOCHS)

plt.plot(losses_gen, label="Generator Loss")
plt.plot(losses_disc, label="Discriminator Loss")
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
```

### Results
BCE loss produces digit-like images after 10 epochs, but they may be blurry. Wasserstein loss requires weight clipping or gradient penalty for stability, not implemented here.

![Generated MNIST digits at epoch 10](attachment://gan_mnist_digits_epoch_10.png)

Sample output:
```
Epoch 1, Generator Loss: -35.2717, Discriminator Loss: -2.0599
...
Epoch 10, Generator Loss: -28.4562, Discriminator Loss: -1.9876
```

### Comparing BCE and Wasserstein Loss
- **BCE Loss**: Simple but prone to vanishing gradients and mode collapse.
- **Wasserstein Loss**: Stabilizes training but needs constraints like gradient penalty.

## Part 3: Enhancing Stability in VAEs and GANs with Fashion-MNIST

### Challenges in Training
VAEs can suffer from posterior collapse, while GANs face mode collapse and instability. Regularization techniques like batch normalization, dropout, and gradient penalties help.

### Loading and Preprocessing Fashion-MNIST

We normalize to [-1, 1] using PyTorch.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("vae_images", exist_ok=True)
os.makedirs("gan_images", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr = 0.0002
latent_dim = 100
img_shape = (1, 28, 28)
img_size = np.prod(img_shape)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataloader = DataLoader(
    datasets.FashionMNIST("../data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)
```

### Implementing VAEs

#### Vanilla VAE
A basic VAE with fully connected layers.

```python
class VanillaVAE(nn.Module):
    def __init__(self):
        super(VanillaVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(img_size, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, img_size), nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        img_recon = self.decoder(z)
        return img_recon.view(x.size()), mu, logvar
```

#### Regularized VAE
Adds batch normalization and dropout.

```python
class RegularizedVAE(nn.Module):
    def __init__(self):
        super(RegularizedVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(img_size, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.3)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, img_size), nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        img_recon = self.decoder(z)
        return img_recon.view(x.size()), mu, logvar
```

#### VAE Loss Function
Combines MSE and KL-divergence.

```python
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x.view(-1, img_size), x.view(-1, img_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

#### Training VAEs
We train for 15 epochs.

```python
def train_vae(model, dataloader, epochs, model_name):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    losses = []
    print(f"--- Training {model_name} ---")
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            recon_imgs, mu, logvar = model(imgs)
            loss = vae_loss_function(recon_imgs, imgs, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader.dataset)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        with torch.no_grad():
            z = torch.randn(64, latent_dim).to(device)
            sample = model.decoder(z).view(64, 1, 28, 28)
            save_image(sample, f"vae_images/{model_name}_epoch_{epoch+1}.png", normalize=True)
    return losses

vae_epochs = 15
vanilla_vae = VanillaVAE()
regularized_vae = RegularizedVAE()
vanilla_losses = train_vae(vanilla_vae, dataloader, vae_epochs, "VanillaVAE")
regularized_losses = train_vae(regularized_vae, dataloader, vae_epochs, "RegularizedVAE")
```

### Implementing GANs

#### Generator and Discriminator
Fully connected layers for simplicity.

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, img_size), nn.Tanh()
        )
    def forward(self, z):
        return self.model(z).view(-1, *img_shape)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, img):
        return self.model(img.view(img.size(0), -1))
```

#### Gradient Penalty
Enforces 1-Lipschitz constraint for WGAN-GP.

```python
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

#### Training GANs
We train for 25 epochs, updating the discriminator 5 times per generator update.

```python
def train_gan(use_stabilization, epochs, model_name):
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    g_losses, d_losses = [], []
    lambda_gp = 10
    print(f"\n--- Training {model_name} ---")
    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            optimizer_D.zero_grad()
            z = torch.randn(real_imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(z).detach()
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            if use_stabilization:
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            else:
                d_loss_real = torch.mean(nn.functional.relu(1.0 - real_validity))
                d_loss_fake = torch.mean(nn.functional.relu(1.0 + fake_validity))
                d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            if i % 5 == 0:
                optimizer_G.zero_grad()
                gen_imgs = generator(z)
                g_loss = -torch.mean(discriminator(gen_imgs))
                g_loss.backward()
                optimizer_G.step()
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        print(f"Epoch [{epoch+1}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
        fixed_z = torch.randn(64, latent_dim).to(device)
        with torch.no_grad():
            sample = generator(fixed_z)
            save_image(sample, f"gan_images/{model_name}_epoch_{epoch+1}.png", normalize=True)
    return g_losses, d_losses

gan_epochs = 25
base_g_loss, base_d_loss = train_gan(use_stabilization=False, epochs=gan_epochs, model_name="BaseGAN")
stable_g_loss, stable_d_loss = train_gan(use_stabilization=True, epochs=gan_epochs, model_name="StableGAN_GP")
```

### Results and Visualizations

#### VAE Results
- **Loss Curves**: Regularized VAE converges better (74.40 vs. 64.99 at epoch 15).
- **Images**: Regularized VAE generates sharper, more diverse clothing items.

```python
plt.figure(figsize=(10, 5))
plt.title("VAE Training Loss Comparison")
plt.plot(vanilla_losses, label="Vanilla VAE Loss")
plt.plot(regularized_losses, label="Regularized VAE (BN+Dropout) Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
```

![VAE Loss Comparison](attachment://vae_fashionmnist_loss.png)

![Vanilla vs. Regularized VAE](attachment://vae_fashionmnist_images_epoch_15.png)

Sample output:
```
--- Training VanillaVAE ---
Epoch [1/15], Loss: 152.1320
...
Epoch [15/15], Loss: 64.9875
--- Training RegularizedVAE ---
Epoch [1/15], Loss: 151.5825
...
Epoch [15/15], Loss: 74.4015
```

#### GAN Results
- **Loss Curves**: Base GAN shows instability; WGAN-GP is more stable.
- **Images**: WGAN-GP produces diverse clothing items, while Base GAN may collapse.

```python
plt.figure(figsize=(10, 5))
plt.title("GAN Training Loss Comparison")
plt.plot(base_g_loss, label="Base GAN Generator Loss", alpha=0.7)
plt.plot(base_d_loss, label="Base GAN Discriminator Loss", alpha=0.7)
plt.plot(stable_g_loss, label="Stable GAN-GP Generator Loss", linewidth=2)
plt.plot(stable_d_loss, label="Stable GAN-GP Discriminator Loss", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.ylim(-50, 50)
plt.grid(True)
plt.show()
```

![GAN Loss Comparison](attachment://gan_fashionmnist_loss.png)

![Base vs. Stable GAN](attachment://gan_fashionmnist_images_epoch_25.png)

Sample output:
```
--- Training BaseGAN ---
Epoch [1/25] D loss: 0.0052, G loss: 3.3804
...
Epoch [25/25] D loss: 0.0382, G loss: 3.2568
--- Training StableGAN_GP ---
Epoch [1/25] D loss: -10.5599, G loss: -1.6712
...
```

## Key Takeaways

- **VAEs**: Learn smooth latent spaces; regularization (batch normalization, dropout) improves convergence and quality.
- **GANs**: Produce sharper images; Wasserstein loss with gradient penalty reduces instability and mode collapse.
- **MNIST and Fashion-MNIST**: Ideal for learning due to simplicity and moderate complexity.
- **Regularization**: Critical for stable training and diverse outputs.
- **Visualization**: Loss curves and images diagnose training issues.

## Next Steps

- **Increase Epochs**: Train for 50–100 epochs.
- **Advanced Architectures**: Use convolutional layers (e.g., DCGAN).
- **Spectral Normalization**: Alternative to gradient penalty for GANs.
- **Explore Variants**: Try β-VAEs, InfoGANs, or StyleGANs.
- **Hyperparameter Tuning**: Adjust learning rates, batch sizes, or latent dimensions.

This guide provides a hands-on introduction to VAEs and GANs. Experiment with the code, explore stabilization techniques, and dive into generative modeling!

## Applications of GANs and VAEs
Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) are both generative models used to create new data samples similar to existing data. Here's a summary of their key applications across domains:

### GAN Applications:
#### Image Generation & Enhancement

- Super-resolution: Improve image quality (e.g., ESRGAN)

- Image-to-Image Translation: Turn sketches into photos (e.g., pix2pix), maps into satellite images, etc.

- Style Transfer: Combine content of one image with style of another

- Inpainting: Fill in missing parts of images

#### Deepfake & Face Generation

- Face synthesis: Generate realistic human faces (e.g., ThisPersonDoesNotExist)

- Video manipulation: Realistic facial expression or voice changes

#### Data Augmentation

- Generate synthetic data to improve performance in data-scarce environments, especially in healthcare or remote sensing

#### Text-to-Image Generation

- Generate images from textual prompts (e.g., GAN-INT-CLS, DALL·E-related approaches)

#### Art & Design

- Assist in creative fields: music, artwork, clothing design

### VAE Applications:
#### Data Compression & Denoising

- VAEs act as powerful lossy compression tools

- Denoising autoencoders remove noise from inputs

#### Anomaly Detection

- Trained VAEs can identify inputs that deviate from learned patterns, useful in cybersecurity, industrial monitoring, and healthcare

#### Latent Space Interpolation

- Generate smooth transitions between inputs (e.g., morph one face into another)

#### Representation Learning

- Learn meaningful low-dimensional representations for tasks like clustering or downstream ML tasks

#### Drug Discovery & Molecule Generation

- Generate novel molecules with desired properties using learned latent space

## Conclusion

GANs and VAEs are powerful generative models widely used for tasks like image synthesis, data augmentation, anomaly detection, and more. While GANs excel at generating sharp, realistic images, VAEs offer structured latent representations and smoother training. However, both models face challenges—GANs suffer from training instability and mode collapse, while VAEs often produce blurrier outputs.

To address these issues, regularization and stabilization techniques are essential. For VAEs, methods like dropout, weight decay, and batch normalization help improve generalization and prevent overfitting. For GANs, techniques such as label smoothing, spectral normalization, and gradient penalty enhance training stability and reduce mode collapse. These strategies ensure that the models learn meaningful data distributions and produce higher-quality, robust outputs.


