import tensorflow as tf
import numpy as np



class BatchLoader():

    def __init__(self, data):
        self.data = data
        self.batch_offset = 0
        self._size=self.data.shape[0]

    def reset_batch_index(self):
            self.batch_offset = 0
            return

    @property
    def size(self):
        return self._size

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self._size:
            #start new epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.data[start:end,:,:,:]
