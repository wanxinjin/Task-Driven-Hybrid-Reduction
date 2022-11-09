import numpy as np
from os import path
import os
import pickle


class Logger:
    def __init__(self, save_dir=None):

        if save_dir is None:
            assert False, 'Please specify a save directory first!'
        else:
            self.path_root = path.join(save_dir, "results")

        # the dir of saving results during each iteration
        self.path_iteration = path.join(self.path_root, 'iterations')

        if not path.exists(self.path_root):
            os.makedirs(self.path_root)
            os.makedirs(self.path_iteration)

    # log the results of each iteration
    def log_iteration(self, data_name, iter_num, data):

        file_name = 'iter' + str(iter_num) + '_' + data_name + '.data'
        saved_path = path.join(self.path_iteration, file_name)

        with open(saved_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    # load the results of each iteration
    def load_iteration(self, data_name, iter_num):

        file_name = 'iter' + str(iter_num) + '_' + data_name + '.data'
        saved_path = path.join(self.path_iteration, file_name)

        with open(saved_path, 'rb') as f:
            data = pickle.load(f)

        return data


# individual logging utility function
def save_data(data_name, data, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()
    else:
        save_dir = path.join(os.getcwd(), save_dir)

    if not path.exists(save_dir):
        os.makedirs(save_dir)

    data_name = data_name + '.data'
    saved_path = path.join(save_dir, data_name)

    with open(saved_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


# individual load utility function
def load_data(data_name, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()

    data_name = data_name + '.data'
    saved_path = path.join(save_dir, data_name)

    try:
        with open(saved_path, 'rb') as f:
            data = pickle.load(f)
    except:
        assert False

    return data
