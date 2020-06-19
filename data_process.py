import numpy as np
import matplotlib.pyplot as plt

def cifar_10_reshape(batch_arg):
    output=np.reshape(batch_arg,(10000,3,32,32)).transpose(0,2,3,1)
    return output

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def train_test_data(path):
    batch1 = unpickle(path + b'/data_batch_1')
    batch2 = unpickle(path + b'/data_batch_2')
    batch3 = unpickle(path + b'/data_batch_3')
    batch4 = unpickle(path + b'/data_batch_4')
    batch5 = unpickle(path + b'/data_batch_5')

    batch1_data = cifar_10_reshape(batch1[b'data'])
    batch2_data = cifar_10_reshape(batch2[b'data'])
    batch3_data = cifar_10_reshape(batch3[b'data'])
    batch4_data = cifar_10_reshape(batch4[b'data'])
    batch5_data = cifar_10_reshape(batch5[b'data'])

    batch1_labels = batch1[b'labels']
    batch2_labels = batch2[b'labels']
    batch3_labels = batch3[b'labels']
    batch4_labels = batch4[b'labels']
    batch5_labels = batch5[b'labels']

    test_batch=unpickle('/home/shrawan/Documents/cifar-10-python/cifar-10-batches-py/test_batch')
    test_images = cifar_10_reshape(test_batch[b'data'])
    test_labels_data = test_batch[b'labels']

    train_images = np.concatenate((batch1_data,batch2_data,batch3_data,batch4_data,batch5_data),axis=0)
    train_labels_data = np.concatenate((batch1_labels,batch2_labels,batch3_labels,batch4_labels,batch5_labels),axis=0)
    return train_images, test_images, train_labels_data, test_labels_data













