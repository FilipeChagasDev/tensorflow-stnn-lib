# Tensorflow-STNN-Lib

Tensorflow-STNN-lib is a Tensorflow-based Python library built to facilitate the implementation of Siamese and Triplet Neural Networks.

**Obs: This library is in the early stages of development and the Siamese Neural Networks features are a priority. Triplet Neural Networks are not yet available.**

## Introduction

**What are Siamese Neural Networks?**

Siamese Neural Networks are a specialized type of neural network architecture designed for tasks that involve measuring the similarity or dissimilarity between pairs of input data. These neural networks are made up of pairs of subnetworks with shared weights (twins), called "encoders", whose outputs are joined by a distance layer. These encoders are neural networks whose inputs are pre-processed data (images, audio, text, etc.) and outputs are **embeddings**.  The aim of the Siamese Neural Network is to train the encoder to map similar data pairs to embedding pairs separated by small distances, and non-similar data pairs to embeddings separated by larger distances. In this way, the encoders can be used in tasks such as facial recognition, voice recognition and signature recognition, especially if associated with vector databases.

**What are Triplet Neural Networks?**

Triplet Neural Networks are an extension of the siamese architecture, specifically designed to enhance the learning process by incorporating triplets of data instances. Unlike siamese networks that work with pairs, triplet networks leverage three instances for training: an anchor, a positive example, and a negative example. The anchor and positive instance belong to the same class or category, while the negative instance belongs to a different class. The network is trained to minimize the distance between the anchor and positive examples while maximizing the distance between the anchor and negative examples in the embedding space. 

**What are Embeddings?** 

Embeddings are vectors of real numbers that represent data such as images, audio and text with very low dimensionality. These vectors are widely used in recognition and semantic search tasks, as they represent complex, unstructured data in a way that makes similarity comparisons and nearest neighbor searches much easier computationally. For example, computing how similar the faces of two different photos are is easy with embeddings: just generate an embedding for each of the photos and calculate the Euclidean distance between the two embeddings. The smaller the distance, the more similar the faces are. Other metrics, such as cosine similarity and inner product, can also be used.

## Documentation

There is documentation of classes, methods and functions in the following document: [Classes, Methods and Functions documentation](sphinx-docs/_build/markdown/index.md). However, I recommend that you learn how to use the library by following the example notebooks.


## Instalation

From GitHub:
```
pip install git+https://github.com/FilipeChagasDev/tensorflow-stnn-lib.git
```

## Example notebooks

TODO

