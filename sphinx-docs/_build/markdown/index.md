<!-- keras-snn-lib documentation master file, created by
sphinx-quickstart on Thu Dec 21 23:21:17 2023.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->

# Tensorflow-STNN-Lib Classes, Mathods and Functions documentation

# Contents:

* [tensorflow_stnn_lib package](tensorflow_stnn_lib.md)
  * [Submodules](tensorflow_stnn_lib.md#submodules)
  * [tensorflow_stnn_lib.distance module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.distance)
    * [`cosine_distance()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.distance.cosine_distance)
    * [`euclidean_distance()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.distance.euclidean_distance)
  * [tensorflow_stnn_lib.generator module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.generator)
    * [`PairDataGenerator`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.generator.PairDataGenerator)
      * [`PairDataGenerator.get_batch_files()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.generator.PairDataGenerator.get_batch_files)
      * [`PairDataGenerator.get_batch_size()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.generator.PairDataGenerator.get_batch_size)
    * [`TripletDataGenerator`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.generator.TripletDataGenerator)
      * [`TripletDataGenerator.get_batch_files()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.generator.TripletDataGenerator.get_batch_files)
      * [`TripletDataGenerator.get_batch_size()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.generator.TripletDataGenerator.get_batch_size)
  * [tensorflow_stnn_lib.loss module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.loss)
    * [`contrastive_loss()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.loss.contrastive_loss)
    * [`triplet_loss()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.loss.triplet_loss)
  * [tensorflow_stnn_lib.metrics module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.metrics)
    * [`get_roc_auc()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.metrics.get_roc_auc)
    * [`plot_histogram()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.metrics.plot_histogram)
    * [`plot_roc()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.metrics.plot_roc)
  * [tensorflow_stnn_lib.net module](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib.net)
    * [`SiameseNet`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet)
      * [`SiameseNet.fit()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.fit)
      * [`SiameseNet.get_embeddings()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.get_embeddings)
      * [`SiameseNet.get_encoder()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.get_encoder)
      * [`SiameseNet.get_test_distances()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.get_test_distances)
      * [`SiameseNet.load_encoder()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.load_encoder)
      * [`SiameseNet.plot_loss()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.plot_loss)
      * [`SiameseNet.save_encoder()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.SiameseNet.save_encoder)
    * [`TrainingBreaker`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TrainingBreaker)
      * [`TrainingBreaker.eval()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TrainingBreaker.eval)
      * [`TrainingBreaker.reset()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TrainingBreaker.reset)
    * [`TripletNet`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TripletNet)
      * [`TripletNet.fit()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TripletNet.fit)
      * [`TripletNet.get_embeddings()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TripletNet.get_embeddings)
      * [`TripletNet.get_encoder()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TripletNet.get_encoder)
      * [`TripletNet.get_test_distances()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TripletNet.get_test_distances)
      * [`TripletNet.load_encoder()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TripletNet.load_encoder)
      * [`TripletNet.plot_loss()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TripletNet.plot_loss)
      * [`TripletNet.save_encoder()`](tensorflow_stnn_lib.md#tensorflow_stnn_lib.net.TripletNet.save_encoder)
  * [Module contents](tensorflow_stnn_lib.md#module-tensorflow_stnn_lib)
