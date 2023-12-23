<!-- keras-snn-lib documentation master file, created by
sphinx-quickstart on Fri Dec 22 18:42:18 2023.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->

# Welcome to Tensorflow-STNN-lib documentation!

# Contents:

* [tensorflow_stnn_lib package](tensorflow_stnn_lib.md)
  * [Main features](tensorflow_stnn_lib.md#main-features)
  * [tensorflow_stnn_lib.net module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.net)
    * [`SiameseNet`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet)
      * [`SiameseNet.fit()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.fit)
      * [`SiameseNet.get_embeddings()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.get_embeddings)
      * [`SiameseNet.get_encoder()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.get_encoder)
      * [`SiameseNet.get_test_distances()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.get_test_distances)
      * [`SiameseNet.get_training_history_df()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.get_training_history_df)
      * [`SiameseNet.load_encoder()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.load_encoder)
      * [`SiameseNet.plot_training_history()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.plot_training_history)
      * [`SiameseNet.save_encoder()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.save_encoder)
    * [`TrainingBreaker`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TrainingBreaker)
      * [`TrainingBreaker.eval()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TrainingBreaker.eval)
      * [`TrainingBreaker.reset()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TrainingBreaker.reset)
  * [tensorflow_stnn_lib.data module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.data)
    * [`PairDataGenerator`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataGenerator)
      * [`PairDataGenerator.get_batch_files()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataGenerator.get_batch_files)
      * [`PairDataGenerator.get_batch_size()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataGenerator.get_batch_size)
    * [`PairDataset`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataset)
      * [`PairDataset.get_batch()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataset.get_batch)
      * [`PairDataset.get_batch_size()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.PairDataset.get_batch_size)
    * [`array_dataset_to_pairs_df()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.array_dataset_to_pairs_df)
    * [`dataset_df_to_pairs_df()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.data.dataset_df_to_pairs_df)
  * [tensorflow_stnn_lib.metrics module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.metrics)
    * [`get_roc_auc()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.metrics.get_roc_auc)
    * [`plot_histogram()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.metrics.plot_histogram)
    * [`plot_roc()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.metrics.plot_roc)
  * [Internal features](tensorflow_stnn_lib.md#internal-features)
  * [tensorflow_stnn_lib.distance module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.distance)
    * [`cosine_distance()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.distance.cosine_distance)
    * [`euclidean_distance()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.distance.euclidean_distance)
  * [tensorflow_stnn_lib.loss module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.loss)
    * [`contrastive_loss()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.loss.contrastive_loss)
    * [`triplet_loss()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.loss.triplet_loss)
