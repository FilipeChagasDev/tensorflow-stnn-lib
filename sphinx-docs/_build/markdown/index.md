<!-- keras-snn-lib documentation master file, created by
sphinx-quickstart on Fri Dec 22 18:42:18 2023.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->

# Welcome to Tensorflow-STNN-lib documentation!

# Contents:

* [tensorflow_stnn_lib package](tensorflow_stnn_lib.md)
  * [Main features](tensorflow_stnn_lib.md#main-features)
  * [tensorflow_stnn_lib.net module](tensorflow_stnn_lib.md#tensorflow-stnn-lib-net-module)
  * [tensorflow_stnn_lib.data module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.data)
    * [`PairDataGenerator`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataGenerator)
      * [`PairDataGenerator.get_batch_files()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataGenerator.get_batch_files)
      * [`PairDataGenerator.get_batch_size()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataGenerator.get_batch_size)
    * [`PairDataset`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataset)
      * [`PairDataset.get_batch()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataset.get_batch)
      * [`PairDataset.get_batch_size()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataset.get_batch_size)
    * [`array_dataset_to_pairs_df()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.array_dataset_to_pairs_df)
    * [`dataset_df_to_pairs_df()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.dataset_df_to_pairs_df)
  * [tensorflow_stnn_lib.metrics module](tensorflow_stnn_lib.md#tensorflow-stnn-lib-metrics-module)
  * [Internal features](tensorflow_stnn_lib.md#internal-features)
  * [tensorflow_stnn_lib.distance module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.distance)
    * [`cosine_distance()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.distance.cosine_distance)
    * [`euclidean_distance()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.distance.euclidean_distance)
  * [tensorflow_stnn_lib.loss module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.loss)
    * [`contrastive_loss()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.loss.contrastive_loss)
    * [`triplet_loss()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.loss.triplet_loss)
