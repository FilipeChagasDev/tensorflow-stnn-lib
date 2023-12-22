# tensorflow_stnn_lib package

## Submodules

## tensorflow_stnn_lib.distance module

### tensorflow_stnn_lib.distance.cosine_distance(vectors)

Cosine distance between two vectors.

* **Parameters:**
  **vectors** (*Tuple**[**tf.Tensor**,* *tf.Tensor**]*) – Tuple with two input tensors
* **Returns:**
  cosine distance tensor
* **Return type:**
  tf.Tensor

### tensorflow_stnn_lib.distance.euclidean_distance(vectors)

Euclidean distance between two vectors.
Ref: [https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/](https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/)

* **Parameters:**
  **vectors** (*Tuple**[**tf.Tensor**,* *tf.Tensor**]*) – Tuple with two input tensors
* **Returns:**
  euclidean distance tensor
* **Return type:**
  tf.Tensor

## tensorflow_stnn_lib.generator module

### *class* tensorflow_stnn_lib.generator.PairDataGenerator(batch_size, pairs_df, loader_fn, name=None)

Bases: `object`

Siamese neural network data generator. 
This class is used to provide the neural network’s training or test data so that it doesn’t consume too much RAM.

#### get_batch_files(index)

* **Return type:**
  `File`

#### get_batch_size()

* **Return type:**
  `int`

### *class* tensorflow_stnn_lib.generator.TripletDataGenerator(batch_size, triplets_df, loader_fn, name=None)

Bases: `object`

Triplet neural network data generator. 
This class is used to provide the neural network’s training or test data so that it doesn’t consume too much RAM.

#### get_batch_files(index)

* **Return type:**
  `File`

#### get_batch_size()

* **Return type:**
  `int`

## tensorflow_stnn_lib.loss module

### tensorflow_stnn_lib.loss.contrastive_loss(y_true, y_pred, margin=1.0)

Contrastive Loss function

* **Parameters:**
  * **y_true** (*tf.Tensor*) – unused target labels
  * **y_pred** (*tf.Tensor*) – model output (a list with positive and negative distances)
  * **margin** (*float**,* *optional*) – desired separation margin between positive and negative examples, defaults to 1.0
* **Returns:**
  Contastive loss tensor
* **Return type:**
  tf.Tensor

### tensorflow_stnn_lib.loss.triplet_loss(y_true, y_pred, margin=1.0)

Triplet Loss function

* **Parameters:**
  * **y_true** (*tf.Tensor*) – unused target labels
  * **y_pred** (*tf.Tensor*) – model output (a list with positive and negative distances)
  * **margin** (*float**,* *optional*) – desired separation margin between positive and negative examples, defaults to 1.0
* **Returns:**
  triplet loss tensor
* **Return type:**
  tf.Tensor

## tensorflow_stnn_lib.metrics module

### tensorflow_stnn_lib.metrics.get_roc_auc(positive_distances, negative_distances)

Obtains the area under the ROC curve (AUC).

* **Parameters:**
  * **positive_distances** (*np.ndarray*) – Array of positive distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.
  * **negative_distances** (*np.ndarray*) – Array of negative distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.
* **Returns:**
  AUC
* **Return type:**
  float

### tensorflow_stnn_lib.metrics.plot_histogram(positive_distances, negative_distances)

Plot a histogram showing the distributions of positive and negative distances.

* **Parameters:**
  * **positive_distances** (*np.ndarray*) – Array of positive distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.
  * **negative_distances** (*np.ndarray*) – Array of negative distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.

### tensorflow_stnn_lib.metrics.plot_roc(positive_distances, negative_distances)

Plots a ROC curve of the encoder’s predictions.

* **Parameters:**
  * **positive_distances** (*np.ndarray*) – Array of positive distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.
  * **negative_distances** (*np.ndarray*) – Array of negative distances obtained with the get_test_distances method of the SiameseNet or TripletNet class.

## tensorflow_stnn_lib.net module

### *class* tensorflow_stnn_lib.net.SiameseNet(input_shape, encoder, margin=1.0, optimizer='adamax', distance='euclidean')

Bases: `object`

Siamese Neural Network

#### fit(training_generator, validation_generator, epochs, start_epoch=1, epoch_end_callback=None, training_breaker=None)

SNN training method. You must provide the training and validation data via PairDataGenerators. 
The use of these generators is mandatory and they serve to reduce the use of RAM memory. 
In addition, you can provide an end-of-epoch callback function and a TrainingBreaker object, 
which is responsible for stopping the training when there is no more evolution in the validation loss.

* **Parameters:**
  * **training_generator** ([*PairDataGenerator*](#tensorflow_stnn_lib.generator.PairDataGenerator)) – Training data generator
  * **validation_generator** ([*PairDataGenerator*](#tensorflow_stnn_lib.generator.PairDataGenerator)) – Validation data generator
  * **epochs** (*int*) – Number of training epochs
  * **start_epoch** (*int**,* *optional*) – Initial epoch, defaults to 1
  * **epoch_end_callback** (*Callable**,* *optional*) – This function is called up at the end of each training season. This function receives a dictionary as an argument, containing the SiameseNet object, the training epoch, the training loss and the validation loss. Defaults to None
  * **training_breaker** ([*TrainingBreaker*](#tensorflow_stnn_lib.net.TrainingBreaker)*,* *optional*) – Object responsible for signaling that training should be interrupted when there is no progress in the validation loss. Defaults to None

#### get_embeddings(x)

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

#### get_test_distances(generator, distance='euclidean')

Obtains positive and negative pair distances from embeddings to analyze encoder performance.

* **Parameters:**
  * **generator** ([*PairDataGenerator*](#tensorflow_stnn_lib.generator.PairDataGenerator)) – Test data generator
  * **distance** (*str**,* *optional*) – Distance function used (‘euclidean’ or ‘cosine’). Defaults to ‘euclidean’. Defaults to ‘euclidean’
* **Raises:**
  **Exception** – The chosen distance is invalid
* **Returns:**
  Tuple with two arrays: positive_distances and negative_distances. Both arrays are one-dimensional.
* **Return type:**
  Tuple[np.ndarray, np.ndarray]

#### load_encoder(path)

Load encoder weights from a file

* **Parameters:**
  **path** (*str*) – file path

#### plot_loss()

Plot a line chart with the evolution of the training and validation losses over the course of the training

#### save_encoder(path)

Save encoder weights to a file

* **Parameters:**
  **path** (*str*) – target file path

### *class* tensorflow_stnn_lib.net.TrainingBreaker(avg_window_size=10, limit=-0.001)

Bases: `object`

#### eval(val_loss)

Computes the moving average of the derivative of the validation loss and returns True if it is time to stop training

* **Parameters:**
  **val_loss** (*float*) – Last validation loss computed
* **Returns:**
  True if it’s time to strop training
* **Return type:**
  bool

#### reset()

Clears the moving average queue

### *class* tensorflow_stnn_lib.net.TripletNet(input_shape, encoder, margin=100.0, optimizer='adamax', distance='euclidean')

Bases: `object`

Triplet Neural Network

#### fit(training_generator, validation_generator, epochs, start_epoch=1, epoch_end_callback=None, training_breaker=None)

TNN training method. You must provide the training and validation data via TripletDataGenerator. 
The use of these generators is mandatory and they serve to reduce the use of RAM memory. 
In addition, you can provide an end-of-epoch callback function and a TrainingBreaker object, 
which is responsible for stopping the training when there is no more evolution in the validation loss.

* **Parameters:**
  * **training_generator** ([*TripletDataGenerator*](#tensorflow_stnn_lib.generator.TripletDataGenerator)) – Training data generator
  * **validation_generator** ([*TripletDataGenerator*](#tensorflow_stnn_lib.generator.TripletDataGenerator)) – Validation data generator
  * **epochs** (*int*) – Number of training epochs
  * **start_epoch** (*int**,* *optional*) – Initial epoch, defaults to 1
  * **epoch_end_callback** (*Callable**,* *optional*) – This function is called up at the end of each training season. This function receives a dictionary as an argument, containing the SiameseNet object, the training epoch, the training loss and the validation loss. Defaults to None
  * **training_breaker** ([*TrainingBreaker*](#tensorflow_stnn_lib.net.TrainingBreaker)*,* *optional*) – Object responsible for signaling that training should be interrupted when there is no progress in the validation loss. Defaults to None

#### get_embeddings(x)

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

#### get_test_distances(generator, distance='euclidean')

Obtains positive and negative pair distances from embeddings to analyze encoder performance.

* **Parameters:**
  * **generator** ([*PairDataGenerator*](#tensorflow_stnn_lib.generator.PairDataGenerator)) – Test data generator
  * **distance** (*str**,* *optional*) – Distance function used (‘euclidean’ or ‘cosine’). Defaults to ‘euclidean’. Defaults to ‘euclidean’
* **Raises:**
  **Exception** – The chosen distance is invalid
* **Returns:**
  Tuple with two arrays: positive_distances and negative_distances. Both arrays are one-dimensional.
* **Return type:**
  Tuple[np.ndarray, np.ndarray]

#### load_encoder(path)

Load encoder weights from a file

* **Parameters:**
  **path** (*str*) – file path

#### plot_loss()

Plot a line chart with the evolution of the training and validation losses over the course of the training

#### save_encoder(path)

Save encoder weights to a file

* **Parameters:**
  **path** (*str*) – target file path

## Module contents
