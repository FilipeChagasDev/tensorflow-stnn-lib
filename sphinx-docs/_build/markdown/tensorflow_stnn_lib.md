# tensorflow_stnn_lib package

## Main features

## tensorflow_stnn_lib.net module

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
