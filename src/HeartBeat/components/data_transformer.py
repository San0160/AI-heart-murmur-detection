# Components

from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn.functional as F
from HeartBeat.config.configuration import DataTransformerConfig
import os

class DataTransformer:
    def __init__(self, config: DataTransformerConfig):
        self.config = config

    def split_and_encode(self, x_data, y_data, test_y, classes, train_size=0.8, val_size=0.9, random_state=80085):
        """
        x_data:       feature array
        y_data:       label array
        test_y:       unlabeled test labels
        classes:      list of classes e.g. [0, 1, 2]
        train_size:   proportion for train split (default 0.8)
        val_size:     proportion of train set for validation (default 0.9)
        random_state: reproducibility seed
        """
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=train_size, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, train_size=val_size, random_state=random_state
        )

        # One hot encode using PyTorch
        num_classes = len(classes)

        def one_hot(labels, num_classes):
            labels = np.array(labels)
            
            # handle -1 unlabeled by temporarily replacing with 0
            mask = labels == -1
            labels_fixed = labels.copy()
            labels_fixed[mask] = 0
            
            t = torch.tensor(labels_fixed, dtype=torch.long)
            encoded = F.one_hot(t, num_classes=num_classes).numpy()
            
            # zero out the rows that were -1 (unlabeled)
            encoded[mask] = 0
            
            return encoded

        y_train = one_hot(y_train, num_classes)
        y_test  = one_hot(y_test,  num_classes)
        y_val   = one_hot(y_val,   num_classes)
        test_y  = one_hot(test_y,  num_classes)

        print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
        print(f"X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
        print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test, test_y
    
    def load_preprocessed(self):
        load_path = self.config.preprocessed_data_path
        x_data = np.load(os.path.join(load_path, "x_data.npy"))
        y_data = np.load(os.path.join(load_path, "y_data.npy"))
        test_x = np.load(os.path.join(load_path, "test_x.npy"))
        test_y = np.load(os.path.join(load_path, "test_y.npy"))
        print("Preprocessed data loaded successfully")
        return x_data, y_data, test_x, test_y

    def save_transformed(self, X_train, X_val, X_test, y_train, y_val, y_test, test_y):
        save_path = self.config.root_dir
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "X_train.npy"), X_train)
        np.save(os.path.join(save_path, "X_val.npy"),   X_val)
        np.save(os.path.join(save_path, "X_test.npy"),  X_test)
        np.save(os.path.join(save_path, "y_train.npy"), y_train)
        np.save(os.path.join(save_path, "y_val.npy"),   y_val)
        np.save(os.path.join(save_path, "y_test.npy"),  y_test)
        np.save(os.path.join(save_path, "test_y.npy"),  test_y)
        print(f"Transformed data saved to {save_path}")