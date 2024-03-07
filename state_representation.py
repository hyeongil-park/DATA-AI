import tensorflow as tf
import numpy as np

class DRRAveStateRepresentation(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):
        # x[1]의 축을 재정렬하여 각 시간 단계에서의 피처 수가 마지막 축에 위치하도록 함
        items_eb = tf.transpose(x[1], perm=(0, 2, 1)) / self.embedding_dim
        #items_eb = x[1] / self.embedding_dim
        wav = self.wav(items_eb)
        #print(wav.shape)
        wav = tf.transpose(wav, perm=(0, 2, 1))
        wav = tf.squeeze(wav, axis=1)
        return self.flatten(wav)