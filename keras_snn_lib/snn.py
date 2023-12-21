import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import *

from keras_snn_lib.distance import euclidean_distance
from keras_snn_lib.loss import triplet_loss
from keras_snn_lib.generator import TripletSNNDataGenerator

class TrainingBreaker():
    def __init__(self, avg_window_size: int = 5, limit: float = -0.001) -> None:
        """This class has a method that signals when the training of a Siamese neural network should be interrupted. 
        The interruption of training is authorized when the moving average of the numerical derivative of the validation 
        loss reaches a maximum limit close to 0.

        :param avg_window_size: Window size of the moving average. Defaults to 5
        :type avg_window_size: int, optional
        :param limit: maximum limit of the moving average of the numerical derivative of validation loss. Defaults to -0.001
        :type limit: float, optional
        """
        self.__avg_window_size = avg_window_size
        self.__limit = limit
        self.__queue = []

    def reset(self):
        """Clears the moving average queue
        """
        self.__queue = []

    def eval(self, val_loss: float) -> bool:
        """Computes the moving average of the derivative of the validation loss and returns True if it is time to stop training

        :param val_loss: Last validation loss computed
        :type val_loss: float
        :return: True if it's time to strop training
        :rtype: bool
        """
        self.__queue.append(val_loss)
        if len(self.__queue) >= self.__avg_window_size:
            del self.__queue[0]
            avg_diff_loss = np.mean(np.gradient(self.__queue))
            if avg_diff_loss > self.__limit:
                return True
        return False

class TripletSNN():
    def __init__(self, input_shape: tuple, encoder: keras.Model, optimizer: optimizers.Optimizer | str = 'adamax', distance_function: function = euclidean_distance):
        self.__input_shape = input_shape
        self.__encoder = encoder
        self.__optimizer = optimizer
        self.__distance_function = distance_function
        self.training_loss_history = []
        self.validation_loss_history = []

        input_anchor = layers.Input(shape=self.__input_shape)
        input_pos = layers.Input(shape=self.__input_shape)
        input_neg = layers.Input(shape=self.__input_shape)

        # One encoder for each input
        encoder_anchor = self.__encoder(input_anchor)
        encoder_pos = self.__encoder(input_pos)
        encoder_neg = self.__encoder(input_neg)

        # Join encoders to a distance layer
        pos_dist = layers.Lambda(self.__distance_function, name='PositiveDistance')(
            [encoder_anchor, encoder_pos])
        neg_dist = layers.Lambda(self.__distance_function, name='NegativeDistance')(
            [encoder_anchor, encoder_neg])

        out = K.concatenate([pos_dist, neg_dist], axis=0)

        # Turn tensors to a keras compiled model.
        self.keras_model = keras.Model(
            inputs=[input_anchor, input_pos, input_neg], outputs=out)
        self.keras_model.compile(loss=triplet_loss, optimizer=self.__optimizer)

    def fit(self, training_generator: TripletSNNDataGenerator, validation_generator: TripletSNNDataGenerator, epochs: int, start_epoch: int = 1, epoch_end_callback : function = None, training_breaker: TrainingBreaker = None) -> Tuple[List[float], List[float]]:
        assert epochs >= 1
        assert start_epoch >= 1
        assert epochs >= start_epoch

        tf.autograph.set_verbosity(0)

        self.training_loss_history = []
        self.validation_loss_history = []

        if training_breaker is not None:
            training_breaker.reset()

        for epoch in range(start_epoch-1, epochs):  # For each epoch
            print(f'EPOCH {epoch+1} OF {epochs}')

            # This variable will accumulate sums of the loss of each step of train_on_batch
            training_loss_sum = 0
            # This variable will accumulate sums of the loss of each step of eval-on-batch
            validation_loss_sum = 0
            training_loss = 0  # This variable will be updated to each iteration with the mean of the loss of each step of train_on_batch
            validation_loss = 0  # This variable will be updated to each iteration with the mean of the loss of each step of eval-on-batch

            # --- Epoch training loop ---
            for i in tqdm(range(len(training_generator))):  # For each batch index
                x, y = training_generator[i]  # Get the current training batch
                # Update models's weights for the current batch
                training_loss_sum += self.keras_model.train_on_batch(x, y, return_dict=True)['loss']
                training_loss = training_loss_sum / (i+1)  # Update loss mean

            print(f'training_loss={training_loss:.4f}')
            self.training_loss_history.append(training_loss)

            # --- Epoch validation loop ---
            for i in tqdm(range(len(validation_generator))):  # For each batch index
                # Get the current validation batch
                vx, vy = validation_generator[i]
                validation_loss_sum += self.keras_model.evaluate(vx, vy, batch_size=validation_generator.get_batch_size(), verbose=0, return_dict=True)['loss']  # evaluate model with the current batch
                validation_loss = validation_loss_sum / (i+1)  # Update loss mean

            print(f'validation_loss={validation_loss:.4f}')
            self.validation_loss_history.append(validation_loss)

            if epoch_end_callback is not None:
                epoch_end_callback(snn=self, epoch=epoch, training_loss=training_loss, validation_loss=validation_loss)

            if training_breaker is not None:
                if training_breaker.eval(validation_loss):
                    print('Interruption of the training process authorized by the training breaker')
                    break

    def get_encoder(self):
        return self.__encoder

    def save_encoder(self, path: str):
        self.__encoder.save_weights(path)

    def load_encoder(self, path: str):
        self.__encoder.load_weights(path)

    def plot_loss(self):
        plt.plot([i+1 for i in range(len(self.training_loss_history))], self.training_loss_history, label='Training')
        plt.plot([i+1 for i in range(len(self.validation_loss_history))], self.validation_loss_history, label='Validation')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def get_embeddings(self, x: np.array | tf.Tensor):
        return self.__encoder.predict(x, verbose=0)