import json
import pickle
import logging
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from itertools import chain
import tarfile

my_tar = tarfile.open('data.tar')
my_tar.extractall('./data') # specify which folder to extract to
my_tar.close()