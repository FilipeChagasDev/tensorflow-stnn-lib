import pandas as pd
from tqdm import tqdm
import numpy as np
import h5py as h5
import os
from typing import *

class TripletSNNDataGenerator():
    def __init__(self, batch_size: int, triplets_df: pd.DataFrame, loader_fn: callable, name: str = None):
        assert isinstance(batch_size, int)
        assert isinstance(triplets_df, pd.DataFrame)
        assert isinstance(loader_fn, callable)
        assert isinstance(name, str)
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
                anchor_array = self.__loader_fn(anchor_addr)
                pos_array = self.__loader_fn(pos_addr)
                neg_array = self.__loader_fn(neg_addr)

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
        #return (x,y), where x=[anchor_image, positive_image, negative_image] and y=0 (it's ignored by the triplet loss function)
        return [anchors, positives, negatives], np.zeros(shape=(self.__batch_size,1))