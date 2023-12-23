# tensorflow_stnn_lib package

## Main features

## tensorflow_stnn_lib.net module

### *class* tensorflow_stnn_lib.net.SiameseNet(input_shape: tuple, encoder: Model, margin: float = 1.0, optimizer: Optimizer | str = 'adam', distance: str = 'euclidean')

Bases: `object`

Siamese Neural Network

#### fit(training_generator: [PairDataGenerator](#tensorflow_stnn_lib.data.PairDataGenerator) | [PairDataset](#tensorflow_stnn_lib.data.PairDataset), validation_generator: [PairDataGenerator](#tensorflow_stnn_lib.data.PairDataGenerator) | [PairDataset](#tensorflow_stnn_lib.data.PairDataset), epochs: int, start_epoch: int = 1, epoch_end_callback: Callable = None, training_breaker: [TrainingBreaker](#tensorflow_stnn_lib.net.TrainingBreaker) = None)

SNN training method. You must provide the training and validation data via PairDataGenerators. 
The use of these generators is mandatory and they serve to reduce the use of RAM memory. 
In addition, you can provide an end-of-epoch callback function and a TrainingBreaker object, 
which is responsible for stopping the training when there is no more evolution in the validation loss.

* **Parameters:**
  * **training_generator** ([*PairDataGenerator*](#tensorflow_stnn_lib.data.PairDataGenerator) *|* [*PairDataset*](#tensorflow_stnn_lib.data.PairDataset)) – Training dataset or generator
  * **validation_generator** ([*PairDataGenerator*](#tensorflow_stnn_lib.data.PairDataGenerator) *|* [*PairDataset*](#tensorflow_stnn_lib.data.PairDataset)) – Validation dataset or generator
  * **epochs** (*int*) – Number of training epochs
  * **start_epoch** (*int**,* *optional*) – Initial epoch, defaults to 1
  * **epoch_end_callback** (*Callable**,* *optional*) – This function is called up at the end of each training season. This function receives a dictionary as an argument, containing the SiameseNet object, the training epoch, the training loss and the validation loss. Defaults to None
  * **training_breaker** ([*TrainingBreaker*](#tensorflow_stnn_lib.net.TrainingBreaker)*,* *optional*) – Object responsible for signaling that training should be interrupted when there is no progress in the validation loss. Defaults to None

#### get_embeddings(x: ndarray | Tensor)

Gets the embeddings of an input array

* **Parameters:**
  **x** (*np.ndarray* *|* *tf.Tensor*) – Input array or tensor
* **Returns:**
  Embedding array
* **Return type:**
  np.ndarray

#### get_encoder()

* **Returns:**
  Encoder network
* **Return type:**
  keras.Model

#### get_test_distances(generator: [PairDataGenerator](#tensorflow_stnn_lib.data.PairDataGenerator) | [PairDataset](#tensorflow_stnn_lib.data.PairDataset), distance: str = 'euclidean')

Obtains positive and negative pair distances from embeddings to analyze encoder performance.

* **Parameters:**
  * **generator** ([*PairDataGenerator*](#tensorflow_stnn_lib.data.PairDataGenerator) *|* [*PairDataset*](#tensorflow_stnn_lib.data.PairDataset)) – Test dataset or generator
  * **distance** (*str**,* *optional*) – Distance function used (‘euclidean’ or ‘cosine’). Defaults to ‘euclidean’. Defaults to ‘euclidean’
* **Raises:**
  **Exception** – The chosen distance is invalid
* **Returns:**
  Tuple with two arrays: positive_distances and negative_distances. Both arrays are one-dimensional.
* **Return type:**
  Tuple[np.ndarray, np.ndarray]

#### get_training_history_df()

Return the training history as a DataFrame with the following columns: epoch, training_loss, validation_loss and validation_auc.

* **Returns:**
  Training history DataFrame
* **Return type:**
  pd.DataFrame

#### load_encoder(path: str)

Load encoder weights from a file

* **Parameters:**
  **path** (*str*) – file path

#### plot_training_history()

Plot a line chart with the evolution of the training loss, validation loss and validation AUC over the course of the training.

#### save_encoder(path: str)

Save encoder weights to a file

* **Parameters:**
  **path** (*str*) – target file path

### *class* tensorflow_stnn_lib.net.TrainingBreaker(avg_window_size: int = 10, limit: float = -0.001)

Bases: `object`

This class has a method that signals when the training of an neural network should be interrupted. 
The interruption of training is authorized when the moving average of the numerical derivative of the validation 
loss reaches a maximum limit close to 0.

#### eval(val_loss: float)

Computes the moving average of the derivative of the validation loss and returns True if it is time to stop training

* **Parameters:**
  **val_loss** (*float*) – Last validation loss computed
* **Returns:**
  True if it’s time to strop training
* **Return type:**
  bool

#### reset()

Clears the moving average queue

## tensorflow_stnn_lib.data module

### *class* tensorflow_stnn_lib.data.PairDataGenerator(batch_size: int, dataset_df: DataFrame, loader_fn: Callable, name: str = None)

Bases: `object`

This class should be used instead of PairDataset when SiameseNet needs to be trained with a large volume of data 
and this data is in files on disk. The PairDataGenerator class will load the files from disk and convert them to HDF5 
datasets (one for each batch). This way, only the RAM needed to store one batch of the dataset is used at a time, avoiding 
memory overflow problems.

#### get_batch_files(index: int)

#### get_batch_size()

### *class* tensorflow_stnn_lib.data.PairDataset(batch_size: int, dataset_x: ndarray, dataset_y: ndarray)

Bases: `object`

This class provides SiameseNet with the sample pairs from a dataset in NumPy format. 
It is ideal for small practices and experiments. For large volumes of data, use PairDataGenerator instead.

#### get_batch(index: int)

#### get_batch_size()

### tensorflow_stnn_lib.data.array_dataset_to_pairs_df(dataset_x: ndarray, dataset_y: ndarray)

Internal function that generates a dataframe of pairs for a dataset in NumPy format (in the MNIST digit dataset standard).

* **Parameters:**
  * **dataset_x** (*np.ndarray*) – NumPy array containing the input samples. This array must have more than one dimension, and the index of the first dimension must be the index of the sample, following the pattern of the MNIST digit dataset.
  * **dataset_y** (*np.ndarray*) – NumPy array containing the classes/labels of the samples. This array must have only one dimension, following the pattern of the MNIST digit dataset.
* **Returns:**
  Output pairs dataframe
* **Return type:**
  pd.DataFrame

### tensorflow_stnn_lib.data.dataset_df_to_pairs_df(dataset_df: DataFrame)

Internal function responsible for transforming an addr-class dataframe into a pair dataframe with addr_left, 
class_left, addr_right, class_right and label columns.

* **Parameters:**
  **dataset_df** (*pd.DataFrame*) – Input addr-class dataframe.
* **Returns:**
  Output pairs dataframe
* **Return type:**
  pd.DataFrame

## tensorflow_stnn_lib.metrics module

### tensorflow_stnn_lib.metrics.get_roc_auc(positive_distances: ndarray, negative_distances: ndarray)

Obtains the area under the ROC curve (AUC).

* **Parameters:**
  * **positive_distances** (*np.ndarray*) – Array of positive distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.
  * **negative_distances** (*np.ndarray*) – Array of negative distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.
* **Returns:**
  AUC
* **Return type:**
  float

### tensorflow_stnn_lib.metrics.plot_histogram(positive_distances: ndarray, negative_distances: ndarray)

Plot a histogram showing the distributions of positive and negative distances.

* **Parameters:**
  * **positive_distances** (*np.ndarray*) – Array of positive distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.
  * **negative_distances** (*np.ndarray*) – Array of negative distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.

### tensorflow_stnn_lib.metrics.plot_roc(positive_distances: ndarray, negative_distances: ndarray)

Plots a ROC curve of the encoder’s predictions.

* **Parameters:**
  * **positive_distances** (*np.ndarray*) – Array of positive distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.
  * **negative_distances** (*np.ndarray*) – Array of negative distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.

## Internal features

## tensorflow_stnn_lib.distance module

### tensorflow_stnn_lib.distance.cosine_distance(vectors: Tuple[Tensor, Tensor])

Cosine distance between two vectors.

* **Parameters:**
  **vectors** (*Tuple**[**tf.Tensor**,* *tf.Tensor**]*) – Tuple with two input tensors
* **Returns:**
  cosine distance tensor
* **Return type:**
  tf.Tensor

### tensorflow_stnn_lib.distance.euclidean_distance(vectors: Tuple[Tensor, Tensor])

Euclidean distance between two vectors.
Ref: [https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/](https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/)

* **Parameters:**
  **vectors** (*Tuple**[**tf.Tensor**,* *tf.Tensor**]*) – Tuple with two input tensors
* **Returns:**
  euclidean distance tensor
* **Return type:**
  tf.Tensor

## tensorflow_stnn_lib.loss module

### tensorflow_stnn_lib.loss.contrastive_loss(y_true: Tensor, y_pred: Tensor, margin: float = 1.0)

Contrastive Loss function

* **Parameters:**
  * **y_true** (*tf.Tensor*) – unused target labels
  * **y_pred** (*tf.Tensor*) – model output (a list with positive and negative distances)
  * **margin** (*float**,* *optional*) – desired separation margin between positive and negative examples, defaults to 1.0
* **Returns:**
  Contastive loss tensor
* **Return type:**
  tf.Tensor

### tensorflow_stnn_lib.loss.triplet_loss(y_true: Tensor, y_pred: Tensor, margin: float = 1.0)

Triplet Loss function

* **Parameters:**
  * **y_true** (*tf.Tensor*) – unused target labels
  * **y_pred** (*tf.Tensor*) – model output (a list with positive and negative distances)
  * **margin** (*float**,* *optional*) – desired separation margin between positive and negative examples, defaults to 1.0
* **Returns:**
  triplet loss tensor
* **Return type:**
  tf.Tensor
