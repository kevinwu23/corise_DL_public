import os
from functools import partial
import json
from glob import glob
from tqdm import tqdm
from dask import bag

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pylab as plt
import matplotlib.animation as animation
import emoji
    
import numbers
from math import sqrt

import torch
from torch.utils.data import Dataset

import utils.constants as const

class QuickDrawDataset(Dataset):
    def __init__(self, path_to_emoji_csvs=None, 
                 doodle_to_emoji_map=const.DOODLE_TO_EMOJI_MAP,
                 ims_per_class=1000, imheight=28, imwidth=28, valfrac=0.1, testfrac=0.1,
                 cache_filepath=None, random_seed=42, dataloader_split='train'):
        
        '''
        Dataset class for loading doodle data from the Google QuickDraw dataset on Kaggle.
        https://www.kaggle.com/c/quickdraw-doodle-recognition/data
        Specific methods taken from https://www.kaggle.com/jpmiller/image-based-cnn
        '''
        
        self.path_to_emoji_csvs = path_to_emoji_csvs
        self.doodle_to_emoji_map = doodle_to_emoji_map
        self.ims_per_class = ims_per_class
        self.imheight = imheight
        self.imwidth = imwidth
        self.valfrac = valfrac
        self.testfrac = testfrac
        self.dataloader_split = dataloader_split
        self.random_seed = random_seed
        
        self.labels_list = np.array(list(self.doodle_to_emoji_map.keys()))
        
        self.data_all = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        if cache_filepath is None:
            self.load_data()
            np.random.seed(self.random_seed)
            np.random.shuffle(self.data_all)
            
        else:
            if os.path.isfile(cache_filepath):
                self.data_all = np.load(cache_filepath)
            else: 
                self.load_data()
                np.random.seed(self.random_seed)
                np.random.shuffle(self.data_all)
                np.save(cache_filepath, self.data_all)
        
        self.split_data()
        
    @staticmethod
    def render_image(imheight, imwidth, stroke_vector):
        
        image = Image.new("RGB", (256, 256), color=(255, 255, 255))
        image_draw = ImageDraw.Draw(image)
        for stroke in json.loads(stroke_vector):
            for i in range(len(stroke[0])-1):
                image_draw.line([stroke[0][i], 
                                 stroke[1][i],
                                 stroke[0][i+1], 
                                 stroke[1][i+1]],
                                fill=(0,0,0), width=5)
        return np.array(image.resize((imheight, imwidth)))[:, :, 0]/255.
    
    def load_data(self):
        
        '''
        Loads data into train_grand, a numpy array with shape (ims_per_class * imheight*imwidth+1)
        '''
        
        data_all = []
        class_paths = [os.path.join(self.path_to_emoji_csvs, '{}.csv'.format(c)) for c in self.labels_list]
        for i, c in enumerate(tqdm(class_paths)):
            train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=self.ims_per_class*5//4)
            train = train[train.recognized == True].head(self.ims_per_class)
            render_function = partial(self.render_image, self.imheight, self.imwidth)
            imagebag = bag.from_sequence(train.drawing.values).map(render_function) 
            datarray = np.array(imagebag.compute())
            datarray = np.reshape(datarray, (self.ims_per_class, -1))    
            labelarray = np.full((train.shape[0], 1), i)
            datarray = np.concatenate((labelarray, datarray), axis=1)
            data_all.append(datarray)

        data_all = np.array([data_all.pop() for i in np.arange(len(self.labels_list))])
        data_all = data_all.reshape((-1, (self.imheight * self.imwidth + 1)))
        
        self.data_all = data_all
        
        return
    
                             
    def split_data(self):
        '''
        Split data into training and test
        If test set, just turn into X and y parts
        '''
        
        if self.data_all is None:
            return None
        
        train_cut = int((self.valfrac + self.testfrac) * self.data_all.shape[0])
        val_cut = int(self.valfrac * self.data_all.shape[0])
        
        self.y_train, self.X_train = self.data_all[train_cut: , 0], self.data_all[train_cut: , 1:]
        self.y_val, self.X_val = self.data_all[0:val_cut, 0], self.data_all[0:val_cut, 1:]
        self.y_test, self.X_test = self.data_all[val_cut:train_cut, 0], self.data_all[val_cut:train_cut, 1:]
        
        return
    
    def get_example(self, class_name, example_idx, train_test_split='train'):
        
        '''
        Get one example of the data, returned in imheight x imwidth form
        '''
        
        assert class_name in self.labels_list, 'Class name not found'
            
        class_label_num = np.where(self.labels_list==class_name)[0][0]
        images = self.X_train[np.where(self.y_train==class_label_num)[0]]
        image = images[example_idx].reshape(self.imheight, self.imwidth)
        
        return image, class_label_num
    
    def __len__(self):
        
        if self.dataloader_split == 'train':
            return len(self.X_train)
        elif self.dataloader_split == 'val':
            return len(self.X_val)
        elif self.dataloader_split == 'test':
            return len(self.X_test)
        else:
            return 0
    
    def __getitem__(self, idx):
        
        if self.dataloader_split == 'train':
            data_to_use_X = self.X_train
            data_to_use_y = self.y_train
        
        elif self.dataloader_split == 'val':
            data_to_use_X = self.X_val
            data_to_use_y = self.y_val
            
        elif self.dataloader_split == 'test':
            data_to_use_X = self.X_test
            data_to_use_y = self.y_test
        
        image_data = np.array([data_to_use_X[idx].reshape(self.imheight, self.imwidth)])
        image = torch.Tensor(image_data)
        label = torch.LongTensor([data_to_use_y[idx]])[0]
            
        return image, label

def create_video(video):

    video = np.array(video)

    fig = plt.figure()
    im = plt.imshow(video[0,:,:,:])

    plt.close() # this is required to not display the generated image

    def init():
        im.set_data(video[0,:,:,:])

    def animate(i):
        im.set_data(video[i,:,:,:])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                                   interval=200)
    
    return anim


def get_dimensions(value):
    if value % 2 == 1:
        value = value + 1
    smallest_diff = value-1
    smallest_pair = (1, value)
    for i in range(2, int(value**0.5)+1):
        if value % i == 0:
            if int(value / i) - i < smallest_diff:
                smallest_diff = int(value / i) - i
                smallest_pair = (i, int(value / i))
    return smallest_pair

def evaluate_doodle_to_emoji(doodle_to_emoji, test_dataloader):

    correct_guesses = 0

    for idx in range(len(test_dataloader)):
        x, y = test_dataloader[idx]
        x_as_np = x[0].detach().numpy()
        emoji_guess = doodle_to_emoji(x_as_np)
        actual_emoji = f':{const.DOODLE_TO_EMOJI_MAP[test_dataloader.labels_list[y.detach().numpy()]]}:'
        actual_emoji = emoji.emojize(actual_emoji, use_aliases=True)
        if emoji_guess == actual_emoji:
            correct_guesses += 1

    total_accuracy = correct_guesses / len(test_dataloader)

    tier_dict = {'C': .5, 'B': .6, 'A': .63, 'S': .68, 'S+': .7}
    for tier in ['C', 'B', 'A', 'S', 'S+']:
        if total_accuracy < tier_dict[tier]:
            break

    print('--------------')
    print(f'Results of evaluation: {total_accuracy: .2f}%')
    print(f'Performance Tier: {tier}. Good job!')

    return