from matplotlib import pyplot as plt
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
import time

import numpy as np
import os

latent_dim = 64
noise_sigma = 0.35
train_AE = False
plot_AE_graph = False
train_classifier = False
train_generator = False
train_digit_generator = True
sml_train_size = 50

# load train and test images, and pad & reshape them to (-1,32,32,1)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)).astype('float32') / 255.0
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)).astype('float32') / 255.0
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))
print(x_train.shape)
print(x_test.shape)
# exit()
y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

encoder = Sequential()
encoder.add(layers.Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same', input_shape=(32, 32, 1)))
encoder.add(layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Reshape((2 * 2 * 96,)))
encoder.add(layers.Dense(latent_dim))

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
decoder = Sequential()
decoder.add(layers.Dense(2 * 2 * 96, activation='relu', input_shape=(latent_dim,)))
decoder.add(layers.Reshape((2, 2, 96)))
decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), activation='sigmoid', padding='same'))

autoencoder = keras.Model(encoder.inputs, decoder(encoder.outputs))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

checkpoint_path = "model_save/cp.ckpt"

if train_AE:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True)
    autoencoder.fit(x_train + noise_sigma * np.random.randn(*x_train.shape), x_train,
                    epochs=15,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[cp_callback])
else:
    autoencoder.load_weights(checkpoint_path)

# decoded_imgs = autoencoder.predict(x_test)
latent_codes = encoder.predict(x_test)
decoded_imgs = decoder.predict(latent_codes)

n = 10

if plot_AE_graph:
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# Question 2 - Classifier Network:
def model1():
    model = Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(latent_dim,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def model2():
    model = Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(latent_dim,)))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def model3():
    model = Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(latent_dim,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def model4():
    model = Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(latent_dim,)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def model5():
    model = Sequential()
    model.add(layers.Dense(128, activation='sigmoid', input_shape=(latent_dim,)))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def model6():
    model = Sequential()
    model.add(layers.Dense(64, activation='sigmoid', input_shape=(latent_dim,)))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def model7():
    model = Sequential()
    model.add(layers.Dense(128, activation='sigmoid', input_shape=(latent_dim,)))
    model.add(layers.Dense(256, activation='sigmoid'))
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def model8():
    model = Sequential()
    model.add(layers.Dense(128, activation='sigmoid', input_shape=(latent_dim,)))
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


if train_classifier:
    # model_funcs = [model1, model2, model3, model4, model5, model6, model7, model8]
    model_funcs = [model8]
    best_model = None
    best_model_full = None
    best_model_index = None

    for i, model_func in enumerate(model_funcs):
        classifier = model_func()

        train_codes = encoder.predict(x_train[:sml_train_size])
        test_codes = encoder.predict(x_test)

        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        results1 = classifier.fit(train_codes, y_train[:sml_train_size],
                                  epochs=200,
                                  batch_size=16,
                                  shuffle=True,
                                  validation_data=(test_codes, y_test))

        full_cls_enc = keras.models.clone_model(encoder)
        full_cls_cls = keras.models.clone_model(classifier)
        full_cls = keras.Model(full_cls_enc.inputs, full_cls_cls(full_cls_enc.outputs))

        full_cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        results2 = full_cls.fit(x_train[:sml_train_size], y_train[:sml_train_size],
                                epochs=200,
                                batch_size=16,
                                shuffle=True,
                                validation_data=(x_test, y_test))

        if best_model is None or results1.history['val_accuracy'][-1] > best_model.history['val_accuracy'][-1]:
            best_model = results1
            best_model_full = results2
            best_model_index = i + 1

    print("Best model is model " + str(best_model_index))
    print("Best model accuracy on last epoch: " + str(best_model.history['val_accuracy'][-1]))
    print("Best full model accuracy on last epoch: " + str(best_model_full.history['val_accuracy'][-1]))

    plt.figure()
    plt.title("Test accuracies of best classifier:")
    plt.plot(np.arange(1, 201), best_model.history['val_accuracy'], label="Only MLP")
    plt.plot(np.arange(1, 201), best_model_full.history['val_accuracy'], label="MLP + Encoder")
    plt.xticks(np.arange(0, 201, 20))
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend()

##############################################################################################
##############################################################################################
# Question 3:

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


if train_generator:
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(latent_dim))
        return model


    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1))
        return model


    generator = make_generator_model()
    discriminator = make_discriminator_model()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 200
    noise_dim = 100

    train_codes = encoder.predict(x_train)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_codes).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


    @tf.function
    def train_step(latent_codes):
        noise = tf.random.normal([len(latent_codes), noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_latent_codes = generator(noise, training=True)

            real_output = discriminator(latent_codes, training=True)
            fake_output = discriminator(generated_latent_codes, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for latent_code_batch in dataset:
                train_step(latent_code_batch)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


    train(train_dataset, EPOCHS)

    n = 10

    test_latent_codes = generator(tf.random.normal([n, noise_dim]))
    test_output = np.array(decoder(test_latent_codes))

    plt.figure(figsize=(20, 2))
    plt.title("Generator sample examples:")

    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(1, n, i)
        plt.imshow(test_output[i - 1].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    a_arr = np.linspace(0.0, 1.0, num=n)
    interpolation_latent_code = np.array(generator(tf.random.normal([2, noise_dim])))
    output_generator_interpolation = []
    output_real_interpolation = []
    real_code1 = np.random.randint(0, len(train_codes))
    real_code2 = np.random.randint(0, len(train_codes))
    for a in a_arr:
        output_generator_interpolation.append(np.array(decoder(
            np.reshape(a * interpolation_latent_code[0] + (1 - a) * interpolation_latent_code[1], (1, latent_dim)))))
        output_real_interpolation.append(np.array(decoder(
            np.reshape(a * np.array(train_codes[real_code1]) + (1 - a) * np.array(train_codes[real_code2]),
                       (1, latent_dim)))))

    plt.figure(figsize=(20, 4))

    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(output_generator_interpolation[i - 1].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + n)
        plt.imshow(output_real_interpolation[i - 1].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

#########################################################################################
#########################################################################################
# Question 4:

if train_digit_generator:
    def make_specific_digit_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(500, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(latent_dim))
        return model


    def make_specific_digit_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(200, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1))
        return model


    specific_digit_generator = make_specific_digit_generator_model()
    specific_digit_discriminator = make_specific_digit_discriminator_model()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 200
    noise_dim = 100

    train_codes = encoder.predict(x_train)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_codes, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


    @tf.function
    def train_step_q4(latent_codes, digits):
        noise = tf.random.normal([len(latent_codes), noise_dim])
        fake_digits = tf.one_hot(tf.random.uniform((len(latent_codes),), maxval=10, dtype=tf.int32), depth=10)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_latent_codes = specific_digit_generator(tf.concat([noise, fake_digits], axis=1), training=True)

            real_output = specific_digit_discriminator(tf.concat([latent_codes, digits], axis=1), training=True)
            fake_output = specific_digit_discriminator(tf.concat([generated_latent_codes, fake_digits], axis=1),
                                                       training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, specific_digit_generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, specific_digit_discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, specific_digit_generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, specific_digit_discriminator.trainable_variables))


    def train_q4(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for latent_code_batch, digits_batch in dataset:
                train_step_q4(latent_code_batch, digits_batch)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


    train_q4(train_dataset, EPOCHS)

    test_output = []
    for i in range(10):
        digit_vector = np.zeros((1, 10))
        digit_vector[0, i] = 1.
        test_output.append(
            np.array(specific_digit_generator(tf.concat([tf.random.normal([1, noise_dim]), digit_vector], axis=1))))

    plt.figure(figsize=(20, 2))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(1, n, i)
        plt.imshow(np.array(decoder(test_output[i - 1])).reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
