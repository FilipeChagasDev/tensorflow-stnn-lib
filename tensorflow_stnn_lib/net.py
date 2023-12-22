import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm.auto import tqdm
from typing import *

from tensorflow_stnn_lib.distance import euclidean_distance, cosine_distance
from tensorflow_stnn_lib.loss import contrastive_loss, triplet_loss
from tensorflow_stnn_lib.generator import PairDataGenerator, TripletDataGenerator, PairDataset
from tensorflow_stnn_lib.metrics import get_roc_auc

class TrainingBreaker():
    def __init__(self, avg_window_size: int = 10, limit: float = -0.001) -> None:
        """This class has a method that signals when the training of an neural network should be interrupted. 
        The interruption of training is authorized when the moving average of the numerical derivative of the validation 
        loss reaches a maximum limit close to 0.

        :param avg_window_size: Window size of the moving average. Defaults to 10
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
    

class SiameseNet():
    """
    Siamese Neural Network
    """
    def __init__(self, input_shape: tuple, encoder: keras.Model, margin: float = 1.0, optimizer: optimizers.Optimizer | str = 'adam', distance: str = 'euclidean'):
        """ 
        :param input_shape: Shape of the encoder's input array/tensor 
        :type input_shape: tuple
        :param encoder: Encoder model
        :type encoder: keras.Model
        :param margin: Margin used in the contrastive loss function, defaults to 1.0
        :type margin: float, optional
        :param optimizer: Tensorflow optimization method, defaults to 'adamax'
        :type optimizer: optimizers.Optimizer | str, optional
        :param distance: Distance function used ('euclidean' or 'cosine'). Defaults to 'euclidean'
        :type distance: str, optional
        :raises Exception: The chosen distance is invalid
        """
        self.__input_shape = input_shape
        self.__encoder = encoder
        self.__optimizer = optimizer

        if distance == 'euclidean':
            self.__distance_function = euclidean_distance
        elif distance == 'cosine':
            self.__distance_function = cosine_distance
        else:
            raise Exception('The chosen distance is invalid')
        
        self.training_loss_history = []
        self.validation_loss_history = []
        self.validation_auc_history = []

        input_left = layers.Input(shape=self.__input_shape)
        input_right = layers.Input(shape=self.__input_shape)
        
        # One encoder for each input
        encoder_left = self.__encoder(input_left)
        encoder_right = self.__encoder(input_right)
        
        # Join encoders to a distance layer
        distance = layers.Lambda(self.__distance_function, name='DistanceLayer')([encoder_left, encoder_right])

        # Turn tensors to a keras compiled model.
        self.keras_model = keras.Model(inputs=[input_left, input_right], outputs=distance)
        self.keras_model.compile(loss=lambda yt, yp: contrastive_loss(yt, yp, margin), optimizer=self.__optimizer)

    def fit(self, training_generator: PairDataGenerator | PairDataset, validation_generator: PairDataGenerator | PairDataset, epochs: int, start_epoch: int = 1, epoch_end_callback : Callable = None, training_breaker: TrainingBreaker = None):
        """SNN training method. You must provide the training and validation data via PairDataGenerators. 
        The use of these generators is mandatory and they serve to reduce the use of RAM memory. 
        In addition, you can provide an end-of-epoch callback function and a TrainingBreaker object, 
        which is responsible for stopping the training when there is no more evolution in the validation loss. 

        :param training_generator: Training dataset or generator
        :type training_generator: PairDataGenerator | PairDataset
        :param validation_generator: Validation dataset or generator
        :type validation_generator: PairDataGenerator | PairDataset
        :param epochs: Number of training epochs
        :type epochs: int
        :param start_epoch: Initial epoch, defaults to 1
        :type start_epoch: int, optional
        :param epoch_end_callback: This function is called up at the end of each training season. This function receives a dictionary as an argument, containing the SiameseNet object, the training epoch, the training loss and the validation loss. Defaults to None
        :type epoch_end_callback: Callable, optional
        :param training_breaker: Object responsible for signaling that training should be interrupted when there is no progress in the validation loss. Defaults to None
        :type training_breaker: TrainingBreaker, optional
        """
        assert epochs >= 1
        assert start_epoch >= 1
        assert epochs >= start_epoch

        tf.autograph.set_verbosity(0)

        self.training_loss_history = []
        self.validation_loss_history = []
        self.validation_auc_history = []

        if training_breaker is not None:
            training_breaker.reset()

        for epoch in range(start_epoch-1, epochs):  # For each epoch
            print(f'EPOCH {epoch+1} OF {epochs}')
            training_loss_sum = 0 # This variable will accumulate sums of the loss of each step of train_on_batch
            validation_loss_sum = 0 # This variable will accumulate sums of the loss of each step of eval-on-batch
            validation_auc_sum = 0
            training_loss = 0  # This variable will be updated to each iteration with the mean of the loss of each step of train_on_batch
            validation_loss = 0  # This variable will be updated to each iteration with the mean of the loss of each step of eval-on-batch
            validation_auc = 0

            # --- Epoch training loop ---
            for i in tqdm(range(len(training_generator))):  # For each batch index
                x, y = training_generator[i]  # Get the current training batch
                # Update models's weights for the current batch
                training_loss_sum += self.keras_model.train_on_batch(x, y, return_dict=True)['loss']
                training_loss = training_loss_sum / (i+1)  # Update loss mean

            print(f'Training Loss: {training_loss:.4f}')
            
            # --- Epoch validation loop ---
            for i in tqdm(range(len(validation_generator))):  # For each batch index
                # Get the current validation batch
                vx, vy = validation_generator[i]
                vp = self.keras_model.predict(vx, verbose=0)
                validation_loss_sum += contrastive_loss(vy, vp)
                v_pos_dist = np.squeeze(vp[np.squeeze(vy.astype(bool))])
                v_neg_dist = np.squeeze(vp[np.squeeze(np.logical_not(vy.astype(bool)))])
                validation_auc_sum += get_roc_auc(v_pos_dist, v_neg_dist)
                #validation_loss_sum += self.keras_model.evaluate(vx, vy, batch_size=validation_generator.get_batch_size(), verbose=0, return_dict=True)['loss']  # evaluate model with the current batch
                validation_loss = validation_loss_sum / (i+1)  # Update loss mean
                validation_auc = validation_auc_sum / (i+1) #Update AUC mean

            print(f'Validation Loss: {validation_loss:.4f}')
            print(f'Validation AUC: {validation_auc:.4f}')

            self.training_loss_history.append(training_loss)
            self.validation_loss_history.append(validation_loss)
            self.validation_auc_history.append(validation_auc)

            if epoch_end_callback is not None:
                epoch_end_callback({
                    'SiameseNet': self, 
                    'epoch': epoch, 
                    'training_loss': training_loss, 
                    'validation_loss': validation_loss
                    })

            if training_breaker is not None:
                if training_breaker.eval(validation_loss):
                    print('Interruption of the training process authorized by the training breaker')
                    break

    def get_encoder(self) -> keras.Model:
        """
        :return: Encoder network
        :rtype: keras.Model
        """
        return self.__encoder

    def save_encoder(self, path: str):
        """Save encoder weights to a file

        :param path: target file path
        :type path: str
        """
        self.__encoder.save_weights(path)

    def load_encoder(self, path: str):
        """Load encoder weights from a file

        :param path: file path
        :type path: str
        """
        self.__encoder.load_weights(path)

    def plot_training_history(self):
        """Plot a line chart with the evolution of the training loss, validation loss and validation AUC over the course of the training.
        """
        epochs = np.arange(len(self.training_loss_history))+1
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, self.training_loss_history, color=color, label='Training Loss')
        ax1.plot(epochs, self.validation_loss_history, color=color, linestyle='--', label='Validation Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')
        ax1.legend()

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('AUC', color=color)  # we already handled the x-label with ax1
        ax2.plot(epochs, self.validation_auc_history, color=color, linestyle='-.',label='Validation AUC')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend()

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #plt.legend()
        plt.show()

        #plt.plot(, self.training_loss_history, label='Training')
        #plt.plot([i+1 for i in range(len(self.validation_loss_history))], self.validation_loss_history, label='Validation')
        #plt.yscale('log')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.legend()
        #plt.grid()
        #plt.show()

    def get_embeddings(self, x: np.ndarray | tf.Tensor) -> np.ndarray:
        """Gets the embeddings of an input array

        :param x: Input array or tensor
        :type x: np.ndarray | tf.Tensor
        :return: Embedding array
        :rtype: np.ndarray
        """
        return self.__encoder.predict(x, verbose=0)
    
    def get_test_distances(self, generator: PairDataGenerator | PairDataset, distance: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]:
        """Obtains positive and negative pair distances from embeddings to analyze encoder performance.

        :param generator: Test dataset or generator
        :type generator: PairDataGenerator | PairDataset
        :param distance: Distance function used ('euclidean' or 'cosine'). Defaults to 'euclidean'. Defaults to 'euclidean'
        :type distance: str, optional
        :raises Exception: The chosen distance is invalid
        :return: Tuple with two arrays: positive_distances and negative_distances. Both arrays are one-dimensional.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if distance == 'euclidean':
            distance_fn = lambda a, b: np.linalg.norm(a-b, axis=1)
        elif distance == 'cosine':
            distance_fn = lambda a, b: 1 - np.sum(a*b, axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
        else:
            raise Exception('The chosen distance is invalid')
        positive_distances = np.array([])
        negative_distances = np.array([])
        for i in tqdm(range(len(generator))):  # For each batch index
            # Get the current validation batch
            [left, right], labels = generator[i]
            left_emb = self.get_embeddings(left)
            right_emb = self.get_embeddings(right)
            labels = np.squeeze(np.array(labels)).astype(bool)
            positive_left_emb = left_emb[labels]
            positive_right_emb = right_emb[labels]
            negative_left_emb = left_emb[np.logical_not(labels)]
            negative_right_emb = right_emb[np.logical_not(labels)]
            positive_distances = np.append(positive_distances, distance_fn(positive_left_emb, positive_right_emb))
            negative_distances = np.append(negative_distances, distance_fn(negative_left_emb, negative_right_emb))
        return positive_distances, negative_distances


class TripletNet():
    """
    Triplet Neural Network
    """
    def __init__(self, input_shape: tuple, encoder: keras.Model, margin: float = 100.0, optimizer: optimizers.Optimizer | str = 'adamax', distance: str = 'euclidean'):
        """
        :param input_shape: Shape of the encoder's input array/tensor 
        :type input_shape: tuple
        :param encoder: Encoder model
        :type encoder: keras.Model
        :param margin: Margin used in the triplet loss function, defaults to 100.0
        :type margin: float, optional
        :param optimizer: Tensorflow optimization method, defaults to 'adamax'
        :type optimizer: optimizers.Optimizer | str, optional
        :param distance: Distance function used ('euclidean' or 'cosine'). Defaults to 'euclidean'
        :type distance: str, optional
        :raises Exception: The chosen distance is invalid
        """
        self.__input_shape = input_shape
        self.__encoder = encoder
        self.__optimizer = optimizer

        if distance == 'euclidean':
            self.__distance_function = euclidean_distance
        elif distance == 'cosine':
            self.__distance_function = cosine_distance
        else:
            raise Exception('The chosen distance is invalid')
        
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
        pos_dist = layers.Lambda(self.__distance_function, name='PositiveDistance')([encoder_anchor, encoder_pos])
        neg_dist = layers.Lambda(self.__distance_function, name='NegativeDistance')([encoder_anchor, encoder_neg])

        out = K.concatenate([pos_dist, neg_dist], axis=0)

        # Turn tensors to a keras compiled model.
        self.keras_model = keras.Model(inputs=[input_anchor, input_pos, input_neg], outputs=out)
        self.keras_model.compile(loss=lambda yt, yp: triplet_loss(yt, yp, margin), optimizer=self.__optimizer)

    def fit(self, training_generator: TripletDataGenerator, validation_generator: TripletDataGenerator, epochs: int, start_epoch: int = 1, epoch_end_callback : Callable = None, training_breaker: TrainingBreaker = None):
        """TNN training method. You must provide the training and validation data via TripletDataGenerator. 
        The use of these generators is mandatory and they serve to reduce the use of RAM memory. 
        In addition, you can provide an end-of-epoch callback function and a TrainingBreaker object, 
        which is responsible for stopping the training when there is no more evolution in the validation loss. 

        :param training_generator: Training data generator
        :type training_generator: TripletDataGenerator
        :param validation_generator: Validation data generator
        :type validation_generator: TripletDataGenerator
        :param epochs: Number of training epochs
        :type epochs: int
        :param start_epoch: Initial epoch, defaults to 1
        :type start_epoch: int, optional
        :param epoch_end_callback: This function is called up at the end of each training season. This function receives a dictionary as an argument, containing the SiameseNet object, the training epoch, the training loss and the validation loss. Defaults to None
        :type epoch_end_callback: Callable, optional
        :param training_breaker: Object responsible for signaling that training should be interrupted when there is no progress in the validation loss. Defaults to None
        :type training_breaker: TrainingBreaker, optional
        """
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
                epoch_end_callback({
                    'SiameseNet': self, 
                    'epoch': epoch, 
                    'training_loss': training_loss, 
                    'validation_loss': validation_loss
                    })

            if training_breaker is not None:
                if training_breaker.eval(validation_loss):
                    print('Interruption of the training process authorized by the training breaker')
                    break

    def get_encoder(self):
        """
        :return: Encoder network
        :rtype: keras.Model
        """
        return self.__encoder

    def save_encoder(self, path: str):
        """Save encoder weights to a file

        :param path: target file path
        :type path: str
        """
        self.__encoder.save_weights(path)

    def load_encoder(self, path: str):
        """Load encoder weights from a file

        :param path: file path
        :type path: str
        """
        self.__encoder.load_weights(path)

    def plot_loss(self):
        """Plot a line chart with the evolution of the training and validation losses over the course of the training
        """
        plt.plot([i+1 for i in range(len(self.training_loss_history))], self.training_loss_history, label='Training')
        plt.plot([i+1 for i in range(len(self.validation_loss_history))], self.validation_loss_history, label='Validation')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def get_embeddings(self, x: np.ndarray | tf.Tensor):
        """Gets the embeddings of an input array

        :param x: Input array or tensor
        :type x: np.ndarray | tf.Tensor
        :return: Embedding array
        :rtype: np.ndarray
        """
        return self.__encoder.predict(x, verbose=0)
    
    def get_test_distances(self, generator: TripletDataGenerator, distance: str = 'euclidean')  -> Tuple[np.ndarray, np.ndarray]:
        """Obtains positive and negative pair distances from embeddings to analyze encoder performance.

        :param generator: Test data generator
        :type generator: PairDataGenerator
        :param distance: Distance function used ('euclidean' or 'cosine'). Defaults to 'euclidean'. Defaults to 'euclidean'
        :type distance: str, optional
        :raises Exception: The chosen distance is invalid
        :return: Tuple with two arrays: positive_distances and negative_distances. Both arrays are one-dimensional.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if distance == 'euclidean':
            distance_fn = lambda a, b: np.linalg.norm(a-b, axis=1)
        elif distance == 'cosine':
            distance_fn = lambda a, b: 1 - np.sum(a*b, axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
        else:
            raise Exception('The chosen distance is invalid')
        positive_distances = np.array([])
        negative_distances = np.array([])
        for i in tqdm(range(len(generator))):  # For each batch index
            # Get the current validation batch
            [anchors, positives, negatives], vy = generator[i]
            anchors_emb = self.get_embeddings(anchors)
            positives_emb = self.get_embeddings(positives)
            negatives_emb = self.get_embeddings(negatives)
            positive_distances = np.append(positive_distances, distance_fn(anchors_emb, positives_emb))
            negative_distances = np.append(negative_distances, distance_fn(anchors_emb, negatives_emb))
        return positive_distances, negative_distances