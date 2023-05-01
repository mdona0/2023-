import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


#生成モデル（ジェネレータ）
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod((28, 28, 1)), activation='tanh'))
    model.add(tf.keras.layers.Reshape((28, 28, 1)))
    
    return model

#識別モデル(ディスクリミネーター）
def build_discriminator(img_shape):
    model = Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

#GAN構築、学習関数
def build_gan(generator, discriminator):
    z = Input(shape=(latent_dim,))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    
    return combined

def train_gan(epochs, batch_size, sample_interval):
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)
    
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)

        if epoch % sample_interval == 0:
            print(f"Epoch: {epoch}, D_loss: {d_loss[0]}, G_loss: {g_loss}")
            sample_images(epoch, generator)

def sample_images(epoch, generator, n=5):
    noise = np.random.normal(0, 1, (n, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        axs[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
        axs[i].axis('off')

    plt.show()

#学習
latent_dim = 100
img_shape = (28, 28, 1)

generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
combined = build_gan(generator, discriminator)

epochs = 30000
batch_size = 32
sample_interval = 1000

train_gan(epochs, batch_size, sample_interval)
