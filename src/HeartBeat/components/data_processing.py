import os 
import glob
import pandas as pd 
import numpy as np
import librosa 
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import IPython.display as ipd
import fnmatch
import math

from HeartBeat.config.configuration import DataPreprocessingConfig


class preprocessing():
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config


    def load_files(self, base_path):
        self.unlable_data    = base_path + "/unlabel/"
        self.normal_data     = base_path + "/normal/"
        self.murmur_data     = base_path + "/murmur/"
        self.extrastole_data = base_path + "/extrastole/"
        self.artifact_data   = base_path + "/artifact/"
        self.extrahls_data   = base_path + "/extrahls/"

    # function for adding noise
    def noise(self, data, x):
        noise = np.random.randn(len(data))
        data_noise = data + x * noise
        return data_noise

    def shift(self, data, x):
        return np.roll(data, x)

    def stretch(self, data, rate):
        return librosa.effects.time_stretch(y = data, rate = rate)

    def pitch(self, data, rate, sr = 22050):
        return librosa.effects.pitch_shift(y = data, sr = sr, n_steps = rate)
    
    def create_label_maps(self, classes):
        """
        classes: list of class names e.g. ["artifact", "murmur", "normal"]
        returns: label_to_int, int_to_label, num_classes
        """
        label_to_int = {k: v for v, k in enumerate(classes)}
        int_to_label = {v: k for k, v in label_to_int.items()}

        print(f"Label to int: {label_to_int}")
        print(f"Int to label: {int_to_label}")

        return label_to_int, int_to_label, len(classes)
    
    def load_file_data(self, folder, file_names, duration = 10, sr = 22050):

        '''
        1) orignal audio to MFCC
        2) Slowed audio to MFCC
        3) sped up audio to MFCC

        '''
        input_length = sr * duration
        features = 52
        data = []

        for file_name in file_names:
            try:
                sound_file = folder + file_name
                X, sr = librosa.load(sound_file, sr = sr, duration = duration)
                dur = librosa.get_duration(y = X, sr = sr)

                # pad audio file to same duration
                if round(dur) < duration:
                    print("Fixing Audio length",file_name)
                    X = librosa.util.fix_length(data = X, size = input_length)

                # Orignal MFCC
                mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sr, n_mfcc = features).T, axis = 0)
                data.append(mfccs.reshape([-1, 1]))

                # stretch 0.8
                stretch_1 = self.stretch(X, 0.8)
                mfcc_1 = np.mean(librosa.feature.mfcc(y = stretch_1, sr = sr, n_mfcc = features).T, axis = 0)
                data.append(mfcc_1.reshape([-1, 1]))

                # Stretch 1.2
                stretch_2 = self.stretch(X, 1.2)
                mfcc_2 = np.mean(librosa.feature.mfcc(y = stretch_2, sr = sr, n_mfcc = features).T, axis = 0)
                data.append(mfcc_2.reshape([-1, 1]))

            except Exception as e:
                print("Error in", file_name, "=>", e)

        return data
    
    def load_all_categories(self, categories, sample_rate=22050, duration=10):
        """
        categories: list of dicts with keys:
            - 'folder': path to audio folder
            - 'prefix': filename prefix e.g. "murmur"
            - 'label' : integer label
        """
        all_sounds = []
        all_labels = []

        for cat in categories:
            files = fnmatch.filter(os.listdir(cat['folder']), f"{cat['prefix']}*.wav")
            sounds = self.load_file_data(folder=cat['folder'], file_names=files, duration=duration, sr=sample_rate)
            labels = [cat['label'] for _ in sounds]

            all_sounds.extend(sounds)
            all_labels.extend(labels)

            print(f"Loaded {len(sounds)} samples for '{cat['prefix']}'")

        print("Loading Done")
        return all_sounds, all_labels
    
    def load_unlabeled_data(self, folder, prefixes, duration=10, sr=22050):
        """
        folder:   path to unlabeled audio folder
        prefixes: list of filename prefixes e.g. ["Aunlabelledtest", "Bunlabelledtest"]
        """
        all_sounds = []
        all_labels = []

        for prefix in prefixes:
            files = fnmatch.filter(os.listdir(folder), f"{prefix}*.wav")
            sounds = self.load_file_data(folder=folder, file_names=files, duration=duration, sr=sr)
            labels = [-1 for _ in sounds]

            all_sounds.extend(sounds)
            all_labels.extend(labels)

            print(f"Loaded {len(sounds)} unlabeled samples for '{prefix}'")

        print("Loading of Unlabeled data Done")
        return all_sounds, all_labels
    
    
    def combine_data(self, labeled_sounds, labeled_labels, unlabeled_sounds, unlabeled_labels):
        x_data = np.array(labeled_sounds)
        y_data = np.array(labeled_labels)

        test_x = np.array(unlabeled_sounds)
        test_y = np.array(unlabeled_labels)

        print(f"Labeled samples:   {len(y_data)}")
        print(f"Unlabeled samples: {len(test_y)}")

        return x_data, y_data, test_x, test_y
    
    def save_preprocessed(self, x_data, y_data, test_x, test_y):
        save_path = self.config.local_data_file
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "x_data.npy"),  x_data)
        np.save(os.path.join(save_path, "y_data.npy"),  y_data)
        np.save(os.path.join(save_path, "test_x.npy"),  test_x)
        np.save(os.path.join(save_path, "test_y.npy"),  test_y)
        print(f"Preprocessed data saved to {save_path}")
    