import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import h5py as h5
import os
from typing import *

class PairDataset():
    def __init__(self, batch_size: int, pairs_df: pd.DataFrame, dataset_x: np.ndarray):
        assert isinstance(batch_size, int)
        assert isinstance(pairs_df, pd.DataFrame)
        assert isinstance(dataset_x, np.ndarray)
        self.__batch_size = batch_size
        self.__pair_df = pairs_df
        self.__dataset_x = dataset_x
        self.__n_batches = (self.__pair_df.shape[0])//self.__batch_size
        self.__dataset_x_left = self.__dataset_x[self.__pairs_df['addr_left']]
        self.__dataset_x_right = self.__dataset_x[self.__pairs_df['addr_right']]
        self.__labels = np.expand_dims(np.array(self.__pair_df['label']), axis=1)
        self.__batch_files = [self.get_batch(i) for i in tqdm(range(self.__n_batches))]
        
    def __len__(self):
        return self.__n_batches
    
    def get_batch_size(self) -> int:
        return self.__batch_size
    
    def get_batch(self, index: int) -> Tuple[List[h5.Dataset], np.ndarray]:
        xl = self.__dataset_x_left[index*self.__batch_size : (index+1)*self.__batch_size]
        xr = self.__dataset_x_right[index*self.__batch_size : (index+1)*self.__batch_size]
        y = self.__labels[index*self.__batch_size : (index+1)*self.__batch_size]
        return [xl, xr], y  
    
    def __getitem__(self, index: int) -> Tuple[List[h5.Dataset], np.ndarray]:
        return self.__batch_files[index]
    
    
class PairDataGenerator():
    """Siamese neural network data generator. 
    This class is used to provide the neural network's training or test data so that it doesn't consume too much RAM.  
    """
    def __init__(self, batch_size: int, pairs_df: pd.DataFrame, loader_fn: Callable, name: str = None):
        """
        :param batch_size: Size of training/test batches
        :type batch_size: int
        :param pairs_df: DataFrame containing the input data pairs and their respective labels. See the examples to see how this DataFrame should be structured.
        :type pairs_df: pd.DataFrame
        :param loader_fn: Function responsible for loading the samples from disk. This function must receive the address of the sample (present in the DataFrame) and return the sample as a NumPy Array.
        :type loader_fn: Callable
        :param name: Generator name. It's important to define this name if you don't want a new generator to be generated every time your script is run. Defaults to None
        :type name: str, optional
        """
        assert isinstance(batch_size, int)
        assert isinstance(pairs_df, pd.DataFrame)
        assert isinstance(loader_fn, Callable)
        assert isinstance(name, (str, type(None)))
        self.__batch_size = batch_size
        self.__pair_df = pairs_df
        self.__loader_fn = loader_fn
        self.__name = name if name is not None else f'gen{id(self)}'
        self.__n_batches = (self.__pair_df.shape[0])//self.__batch_size
        print('Preparing Dataset Generator')
        self.__batch_files = [self.get_batch_files(i) for i in tqdm(range(self.__n_batches))]
        
    def __len__(self):
        return self.__n_batches
    
    def get_batch_size(self) -> int:
        return self.__batch_size
    
    def get_batch_files(self, index: int) -> h5.File:
        assert isinstance(index, int)
        max_index = self.__batch_size*self.__n_batches
        start_index = min(index * self.__batch_size, max_index)
        end_index = min((index + 1) * self.__batch_size, max_index)
        batch_triplets = self.__pair_df.iloc[start_index: end_index]
        
        #output arrays
        left = None
        right = None
        labels = None
        
        batch_file_path = os.path.join(self.__name, f'{self.__name}_{index}.h5')
        if not os.path.exists(batch_file_path):
            #load all the images of the batch
            for i, left_addr, left_id, right_addr, right_id, label in batch_triplets.itertuples():
                #load data
                left_array = np.expand_dims(self.__loader_fn(left_addr), axis=0)
                right_array = np.expand_dims(self.__loader_fn(right_addr), axis=0)

                #append data to collections of anchors, positives and negatives
                if left is None:
                    left = left_array
                else:
                    left = np.append(left, left_array, axis=0)

                if right is None:
                    right = right_array
                else:
                    right = np.append(right, right_array, axis=0)
                
                if labels is None:
                    labels = np.array([[label]])
                else:
                    labels = np.append(labels, np.array([[label]]), axis=0)

            if not os.path.exists(self.__name):
                os.mkdir(self.__name)
            
            batch_h5 = h5.File(batch_file_path, 'a')
            batch_h5.create_dataset('left', data=left)
            batch_h5.create_dataset('right', data=right)
            batch_h5.create_dataset('labels', data=labels)
            batch_h5.close()
        
        batch_h5 = h5.File(batch_file_path, 'r')
        return batch_h5
    
    def __getitem__(self, index: int) -> Tuple[List[h5.Dataset], np.ndarray]:
        batch_h5 = self.__batch_files[index]
        left = batch_h5['left']
        right = batch_h5['right']
        labels = batch_h5['labels']
        return [left, right], labels
    
class TripletDataGenerator():
    """Triplet neural network data generator. 
    This class is used to provide the neural network's training or test data so that it doesn't consume too much RAM.  
    """
    def __init__(self, batch_size: int, triplets_df: pd.DataFrame, loader_fn: Callable, name: str = None):
        """
        :param batch_size: Size of training/test batches
        :type batch_size: int
        :param triplets_df: DataFrame containing the input triplets. See the examples to see how this DataFrame should be structured.
        :type triplets_df: pd.DataFrame
        :param loader_fn: Function responsible for loading the samples from disk. This function must receive the address of the sample (present in the DataFrame) and return the sample as a NumPy Array.
        :type loader_fn: Callable
        :param name: Generator name. It's important to define this name if you don't want a new generator to be generated every time your script is run. Defaults to None
        :type name: str, optional
        """
        assert isinstance(batch_size, int)
        assert isinstance(triplets_df, pd.DataFrame)
        assert isinstance(loader_fn, Callable)
        assert isinstance(name, (str, type(None)))
        self.__batch_size = batch_size
        self.__triplets_df = triplets_df
        self.__loader_fn = loader_fn
        self.__name = name if name is not None else f'gen{id(self)}'
        self.__n_batches = (self.__triplets_df.shape[0])//self.__batch_size
        print('Preparing Dataset Generator')
        self.__batch_files = [self.get_batch_files(i) for i in tqdm(range(self.__n_batches))]
        
    def __len__(self):
        return self.__n_batches
    
    def get_batch_size(self) -> int:
        return self.__batch_size
    
    def get_batch_files(self, index: int) -> h5.File:
        assert isinstance(index, int)
        max_index = self.__batch_size*self.__n_batches
        start_index = min(index * self.__batch_size, max_index)
        end_index = min((index + 1) * self.__batch_size, max_index)
        batch_triplets = self.__triplets_df.iloc[start_index: end_index]
        
        #output arrays
        anchors = None
        positives = None
        negatives = None
        
        batch_file_path = os.path.join(self.__name, f'{self.__name}_{index}.h5')
        if not os.path.exists(batch_file_path):
            #load all the images of the batch
            for i, anchor_addr, anchor_id, pos_addr, pos_id, neg_addr, neg_id in batch_triplets.itertuples():
                #load data
                anchor_array = np.expand_dims(self.__loader_fn(anchor_addr), axis=0)
                pos_array = np.expand_dims(self.__loader_fn(pos_addr), axis=0)
                neg_array = np.expand_dims(self.__loader_fn(neg_addr), axis=0)

                #append data to collections of anchors, positives and negatives
                if anchors is None:
                    anchors = anchor_array
                else:
                    anchors = np.append(anchors, anchor_array, axis=0)

                if positives is None:
                    positives = pos_array
                else:
                    positives = np.append(positives, pos_array, axis=0)

                if negatives is None:
                    negatives = neg_array
                else:
                    negatives = np.append(negatives, neg_array, axis=0)
         
            if not os.path.exists(self.__name):
                os.mkdir(self.__name)
            
            batch_h5 = h5.File(batch_file_path, 'a')
            batch_h5.create_dataset('anchors', data=anchors)
            batch_h5.create_dataset('positives', data=positives)
            batch_h5.create_dataset('negatives', data=negatives)
            batch_h5.close()
        
        batch_h5 = h5.File(batch_file_path, 'r')
        return batch_h5
    
    def __getitem__(self, index: int) -> Tuple[List[h5.Dataset], np.ndarray]:
        batch_h5 = self.__batch_files[index]
        anchors = batch_h5['anchors']
        positives = batch_h5['positives']
        negatives = batch_h5['negatives']
        #return (x,y), where x=[anchor, positive, negative] and y=0 (it's ignored by the triplet loss function)
        return [anchors, positives, negatives], np.zeros(shape=(self.__batch_size,1))