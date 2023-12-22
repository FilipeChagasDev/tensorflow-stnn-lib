import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import h5py as h5
import os
from typing import *

def dataset_df_to_pairs_df(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Internal function responsible for transforming an addr-class dataframe into a pair dataframe with addr_left, 
    class_left, addr_right, class_right and label columns.

    :param dataset_df: Input addr-class dataframe.
    :type dataset_df: pd.DataFrame
    :return: Output pairs dataframe
    :rtype: pd.DataFrame
    """
    #Make a dataframe of positive (same class) pairs
    def make_pos_pairs(group):
        group_left = group.reset_index(drop=True).rename(columns={'addr': 'addr_left', 'class': 'class_left'})
        group_right = group.sample(frac=1).reset_index(drop=True).rename(columns={'addr': 'addr_right', 'class': 'class_right'})
        group_pairs = pd.concat([group_left, group_right], axis=1)
        return group_pairs

    pos_pairs_df = dataset_df.groupby('class', as_index=False, group_keys=False).apply(make_pos_pairs)
    pos_pairs_df['label'] = 1
    
    #Make a dataframe of negative (different classes) pairs
    def make_neg_pairs(group):
        group_class = group['class'].iloc[0]
        group_left = group.reset_index(drop=True).rename(columns={'addr': 'addr_left', 'class': 'class_left'})
        group_right = dataset_df[dataset_df['class'] != group_class].sample(n=group.shape[0])
        group_right = group_right.reset_index(drop=True).rename(columns={'addr': 'addr_right', 'class': 'class_right'})
        group_pairs = pd.concat([group_left, group_right], axis=1)
        return group_pairs

    neg_pairs_df = dataset_df.groupby('class', as_index=False, group_keys=False).apply(make_neg_pairs)
    neg_pairs_df['label'] = 0

    #Concatenate and shuffle positive and negative pairs
    pairs_df = pd.concat([pos_pairs_df, neg_pairs_df]).sample(frac=1).reset_index(drop=True)
    return pairs_df

def array_dataset_to_pairs_df(dataset_x: np.ndarray, dataset_y: np.ndarray) -> pd.DataFrame:
    """Internal function that generates a dataframe of pairs for a dataset in NumPy format (in the MNIST digit dataset standard).

    :param dataset_x: NumPy array containing the input samples. This array must have more than one dimension, and the index of the first dimension must be the index of the sample, following the pattern of the MNIST digit dataset.
    :type dataset_x: np.ndarray
    :param dataset_y: NumPy array containing the classes/labels of the samples. This array must have only one dimension, following the pattern of the MNIST digit dataset.
    :type dataset_y: np.ndarray
    :return: Output pairs dataframe
    :rtype: pd.DataFrame
    """
    dataset_df = pd.DataFrame({'addr': np.arange(dataset_x.shape[0]), 'class': dataset_y})
    return dataset_df_to_pairs_df(dataset_df)
    
class PairDataset():
    """
    This class provides SiameseNet with the sample pairs from a dataset in NumPy format. 
    It is ideal for small practices and experiments. For large volumes of data, use PairDataGenerator instead.
    """
    def __init__(self, batch_size: int, dataset_x: np.ndarray, dataset_y: np.ndarray):
        """
        :param batch_size: Size of training/test batches
        :type batch_size: int
        :param dataset_x: NumPy array containing the input samples. This array must have more than one dimension, and the index of the first dimension must be the index of the sample, following the pattern of the MNIST digit dataset.
        :type dataset_x: np.ndarray
        :param dataset_y: NumPy array containing the classes/labels of the samples. This array must have only one dimension, following the pattern of the MNIST digit dataset.
        :type dataset_y: np.ndarray
        """
        assert isinstance(batch_size, int)
        assert isinstance(dataset_x, np.ndarray)
        assert isinstance(dataset_y, np.ndarray)
        assert dataset_x.ndim > 1
        assert dataset_y.ndim == 1
        assert dataset_x.shape[0] == dataset_y.shape[0]
        self.__batch_size = batch_size
        self.__pairs_df = array_dataset_to_pairs_df(dataset_x, dataset_y)
        self.__dataset_x = dataset_x
        self.__n_batches = (self.__pairs_df.shape[0])//self.__batch_size
        self.__dataset_x_left = self.__dataset_x[self.__pairs_df['addr_left']]
        self.__dataset_x_right = self.__dataset_x[self.__pairs_df['addr_right']]
        self.__labels = np.expand_dims(np.array(self.__pairs_df['label']), axis=1)
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
    """
    This class should be used instead of PairDataset when SiameseNet needs to be trained with a large volume of data 
    and this data is in files on disk. The PairDataGenerator class will load the files from disk and convert them to HDF5 
    datasets (one for each batch). This way, only the RAM needed to store one batch of the dataset is used at a time, avoiding 
    memory overflow problems.
    """
    def __init__(self, batch_size: int, dataset_df: pd.DataFrame, loader_fn: Callable, name: str = None):
        """
        :param batch_size: Size of training/test batches
        :type batch_size: int
        :param dataset_df: DataFrame containing an 'addr' column and a 'class' column. Each row of this DataFrame corresponds to a sample of the dataset. The row's 'addr' attribute contains the name of the file where the sample is stored (or some other information that identifies the sample on disk), and the 'class' attribute contains the sample's class number.
        :type dataset_df: pd.DataFrame
        :param loader_fn: Function responsible for loading the samples from disk. This function must receive the address of the sample (the 'addr' attribute in the dataset_df) and return the sample as a NumPy Array.
        :type loader_fn: Callable
        :param name: Generator name. It's important to define this name if you don't want a new generator to be generated every time your script is run. Defaults to None
        :type name: str, optional
        """
        assert isinstance(batch_size, int)
        assert isinstance(dataset_df, pd.DataFrame)
        assert isinstance(loader_fn, Callable)
        assert isinstance(name, (str, type(None)))
        assert 'addr' in dataset_df.columns
        assert 'class' in dataset_df.columns
        self.__batch_size = batch_size
        self.__pair_df = dataset_df_to_pairs_df(dataset_df)
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
            for i, left_addr, left_class, right_addr, right_class, label in batch_triplets.itertuples():
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
    
'''
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
'''