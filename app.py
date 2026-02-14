"""
EvoNAS Tool - Automated Neural Architecture Search with Evolutionary Algorithms
Supports both Tabular (CSV) and Image (Computer Vision) datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import copy
import random
import time
import plotly.graph_objects as go
import plotly.express as px
from collections import namedtuple
from io import StringIO, BytesIO
import json
import zipfile
import os
from pathlib import Path
from PIL import Image

# ML Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EvoNAS Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# IMAGE DATASET CLASS
# ============================================================================
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# IMAGE AUGMENTATION PIPELINE
# ============================================================================
class ImageAugmentationPipeline:
    def __init__(self, config, img_size=32):
        self.config = config
        self.img_size = img_size
        
    def get_transform(self, is_training=True):
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize((self.img_size, self.img_size)))
        
        if is_training:
            # Augmentation operations
            if self.config.get('random_flip', False):
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            if self.config.get('random_rotation', False):
                angle = self.config.get('rotation_angle', 15)
                transform_list.append(transforms.RandomRotation(angle))
            
            if self.config.get('color_jitter', False):
                brightness = self.config.get('brightness', 0.2)
                contrast = self.config.get('contrast', 0.2)
                saturation = self.config.get('saturation', 0.2)
                transform_list.append(transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation
                ))
            
            if self.config.get('random_crop', False):
                crop_size = int(self.img_size * 0.875)
                transform_list.append(transforms.RandomCrop(crop_size))
                transform_list.append(transforms.Resize((self.img_size, self.img_size)))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if self.config.get('normalize', True):
            mean = self.config.get('normalize_mean', [0.485, 0.456, 0.406])
            std = self.config.get('normalize_std', [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)

# ============================================================================
# CNN MODEL
# ============================================================================
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn=True, activation='relu'):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x

class CNN(nn.Module):
    def __init__(self, arch, num_classes, input_channels=3, img_size=32):
        super(CNN, self).__init__()
        
        layers = []
        in_channels = input_channels
        current_size = img_size
        
        # Build convolutional layers
        for block_config in arch:
            out_channels = block_config['filters']
            kernel_size = block_config['kernel_size']
            stride = block_config.get('stride', 1)
            padding = (kernel_size - 1) // 2
            use_bn = block_config.get('use_bn', True)
            activation = block_config.get('activation', 'relu')
            
            layers.append(CNNBlock(in_channels, out_channels, kernel_size, stride, padding, use_bn, activation))
            
            # Pooling
            if block_config.get('pool', False):
                pool_type = block_config.get('pool_type', 'max')
                pool_size = block_config.get('pool_size', 2)
                if pool_type == 'max':
                    layers.append(nn.MaxPool2d(pool_size))
                else:
                    layers.append(nn.AvgPool2d(pool_size))
                current_size = current_size // pool_size
            
            # Dropout
            dropout = block_config.get('dropout', 0.0)
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate the flattened size
        self.flatten_size = in_channels * current_size * current_size
        
        # Fully connected layers
        fc_layers = []
        fc_config = arch[0].get('fc_layers', [128])  # Default FC layer
        
        current_dim = self.flatten_size
        for fc_units in fc_config:
            fc_layers.append(nn.Linear(current_dim, fc_units))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.5))
            current_dim = fc_units
        
        fc_layers.append(nn.Linear(current_dim, num_classes))
        
        self.classifier = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# PREPROCESSING PIPELINE (for Tabular Data)
# ============================================================================
class EDA_Pipeline:
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.label_encoders = {}
        self.feature_transformer = None
        self.selected_features = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.rf_model = None
        
    def identify_column_types(self, X):
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def handle_missing_values(self, X, fit=True):
        X = X.copy()
        
        if len(self.numeric_cols) > 0 and self.config.get('numeric_impute') != 'none':
            strategy = self.config['numeric_impute']
            if strategy == 'mode':
                strategy = 'most_frequent'
            if fit:
                self.numeric_imputer = SimpleImputer(strategy=strategy)
                X[self.numeric_cols] = self.numeric_imputer.fit_transform(X[self.numeric_cols])
            else:
                if self.numeric_imputer:
                    X[self.numeric_cols] = self.numeric_imputer.transform(X[self.numeric_cols])
        
        if len(self.categorical_cols) > 0:
            if self.config['categorical_impute'] == 'mode':
                if fit:
                    self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                    X[self.categorical_cols] = self.categorical_imputer.fit_transform(X[self.categorical_cols])
                else:
                    if self.categorical_imputer:
                        X[self.categorical_cols] = self.categorical_imputer.transform(X[self.categorical_cols])
            else:
                if fit:
                    self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
                    X[self.categorical_cols] = self.categorical_imputer.fit_transform(X[self.categorical_cols])
                else:
                    if self.categorical_imputer:
                        X[self.categorical_cols] = self.categorical_imputer.transform(X[self.categorical_cols])
        
        return X
    
    def handle_outliers(self, X, fit=True):
        X = X.copy()
        if self.config['outlier_method'] == 'none' or len(self.numeric_cols) == 0:
            return X
        
        for col in self.numeric_cols:
            if self.config['outlier_method'] == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.config['outlier_threshold'] * IQR
                upper = Q3 + self.config['outlier_threshold'] * IQR
                X[col] = X[col].clip(lower, upper)
            elif self.config['outlier_method'] == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                if std > 0:
                    threshold = self.config['outlier_threshold']
                    X[col] = X[col].clip(mean - threshold * std, mean + threshold * std)
            elif self.config['outlier_method'] == 'clip':
                lower = X[col].quantile(0.01)
                upper = X[col].quantile(0.99)
                X[col] = X[col].clip(lower, upper)
        
        return X
    
    def encode_categorical(self, X, fit=True):
        X = X.copy()
        if len(self.categorical_cols) == 0:
            return X
        
        if self.config['categorical_encoding'] == 'label':
            for col in self.categorical_cols:
                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        X[col] = X[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        X[col] = le.transform(X[col])
        elif self.config['categorical_encoding'] == 'frequency':
            for col in self.categorical_cols:
                if fit:
                    freq_map = X[col].value_counts(normalize=True).to_dict()
                    self.label_encoders[col] = freq_map
                    X[col] = X[col].map(freq_map).fillna(0)
                else:
                    freq_map = self.label_encoders.get(col, {})
                    X[col] = X[col].map(freq_map).fillna(0)
        
        return X
    
    def feature_engineering(self, X):
        X = X.copy()
        current_numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(current_numeric) < 2:
            return X
        
        cols_to_use = current_numeric[:min(5, len(current_numeric))]
        
        if self.config.get('create_interactions', False):
            for i in range(len(cols_to_use)):
                for j in range(i+1, min(i+3, len(cols_to_use))):
                    col1, col2 = cols_to_use[i], cols_to_use[j]
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        
        if self.config.get('create_ratios', False):
            for i in range(len(cols_to_use)):
                for j in range(i+1, min(i+3, len(cols_to_use))):
                    col1, col2 = cols_to_use[i], cols_to_use[j]
                    X[f'{col1}_div_{col2}'] = X[col1] / (X[col2].abs() + 1e-5)
        
        if self.config.get('create_binning', False):
            for col in cols_to_use[:3]:
                try:
                    X[f'{col}_binned'] = pd.cut(X[col], bins=self.config['n_bins'], labels=False, duplicates='drop')
                except:
                    pass
        
        if self.config.get('create_polynomial', False):
            X_poly = X[cols_to_use] ** 2
            X_poly.columns = [f'{col}_squared' for col in cols_to_use]
            X = pd.concat([X, X_poly], axis=1)
        
        return X
    
    def scale_features(self, X, fit=True):
        X = X.copy()
        if self.config['scaling'] == 'none':
            return X
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            return X
        
        if self.config['scaling'] == 'standard':
            if fit:
                self.scaler = StandardScaler()
                X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            else:
                if self.scaler:
                    X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        elif self.config['scaling'] == 'minmax':
            if fit:
                self.scaler = MinMaxScaler()
                X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            else:
                if self.scaler:
                    X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        elif self.config['scaling'] == 'robust':
            if fit:
                self.scaler = RobustScaler()
                X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            else:
                if self.scaler:
                    X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        return X
    
    def select_features(self, X, y, fit=True):
        X = X.copy()
        if not self.config.get('feature_selection', False):
            return X
        
        if fit:
            method = self.config.get('feature_selection_method', 'rf_importance')
            
            if method == 'rf_importance':
                self.rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                self.rf_model.fit(X, y)
                importances = pd.Series(self.rf_model.feature_importances_, index=X.columns)
                threshold = self.config.get('selection_threshold', 0.01)
                self.selected_features = importances[importances >= threshold].index.tolist()
                
                if len(self.selected_features) < 5:
                    self.selected_features = importances.nlargest(min(10, len(X.columns))).index.tolist()
            
            elif method == 'selectk':
                k = min(self.config.get('n_features_to_select', 10), X.shape[1])
                selector = SelectKBest(f_classif, k=k)
                selector.fit(X, y)
                self.selected_features = X.columns[selector.get_support()].tolist()
                self.feature_transformer = selector
            
            elif method == 'pca':
                n_components = min(self.config.get('n_components', 10), X.shape[1])
                self.feature_transformer = PCA(n_components=n_components)
                X_transformed = self.feature_transformer.fit_transform(X)
                return pd.DataFrame(X_transformed, columns=[f'PC{i+1}' for i in range(n_components)])
            
            return X[self.selected_features]
        else:
            if self.selected_features:
                available_features = [f for f in self.selected_features if f in X.columns]
                missing_features = [f for f in self.selected_features if f not in X.columns]
                
                for f in missing_features:
                    X[f] = 0
                
                return X[self.selected_features]
            elif self.feature_transformer:
                X_transformed = self.feature_transformer.transform(X)
                return pd.DataFrame(X_transformed, columns=[f'PC{i+1}' for i in range(X_transformed.shape[1])])
            
            return X
    
    def fit_transform(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        
        self.identify_column_types(X)
        X = self.handle_missing_values(X, fit=True)
        X = self.handle_outliers(X, fit=True)
        X = self.encode_categorical(X, fit=True)
        X = self.feature_engineering(X)
        X = self.scale_features(X, fit=True)
        X = self.select_features(X, y.values, fit=True)
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X.values, y.values
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.reset_index(drop=True)
        
        X = self.handle_missing_values(X, fit=False)
        X = self.handle_outliers(X, fit=False)
        X = self.encode_categorical(X, fit=False)
        X = self.feature_engineering(X)
        X = self.scale_features(X, fit=False)
        
        if self.config.get('feature_selection', False) and self.selected_features:
            available_features = [f for f in self.selected_features if f in X.columns]
            missing_features = [f for f in self.selected_features if f not in X.columns]
            
            for f in missing_features:
                X[f] = 0
            
            X = X[self.selected_features]
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X.values

# ============================================================================
# MLP MODEL (for Tabular Data)
# ============================================================================
class MLP(nn.Module):
    def __init__(self, arch, input_dim, num_classes):
        super(MLP, self).__init__()
        layers = []
        cur_dim = input_dim
        
        for layer_config in arch:
            layers.append(nn.Linear(cur_dim, layer_config['units']))
            
            if layer_config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif layer_config['activation'] == 'tanh':
                layers.append(nn.Tanh())
            elif layer_config['activation'] == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            if layer_config.get('dropout', 0) > 0:
                layers.append(nn.Dropout(layer_config['dropout']))
            
            cur_dim = layer_config['units']
        
        layers.append(nn.Linear(cur_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# EVOLUTIONARY ALGORITHM - RANDOM GENERATORS
# ============================================================================
Individual = namedtuple('Individual', ['preprocessing', 'augmentation', 'arch', 'model_type', 'fitness', 'metrics'])

def random_preprocessing_tabular():
    return {
        'numeric_impute': random.choice(['mean', 'median', 'mode']),
        'categorical_impute': random.choice(['mode', 'constant']),
        'outlier_method': random.choice(['none', 'iqr', 'zscore', 'clip']),
        'outlier_threshold': random.choice([1.5, 2.0, 3.0]),
        'create_interactions': random.choice([True, False]),
        'create_ratios': random.choice([True, False]),
        'create_binning': random.choice([True, False]),
        'create_polynomial': random.choice([True, False]),
        'n_bins': random.choice([3, 5, 10]),
        'categorical_encoding': random.choice(['label', 'frequency']),
        'scaling': random.choice(['standard', 'minmax', 'robust', 'none']),
        'feature_selection': random.choice([True, False]),
        'feature_selection_method': random.choice(['rf_importance', 'selectk', 'pca']),
        'selection_threshold': random.choice([0.01, 0.05, 0.1]),
        'n_features_to_select': random.choice([10, 15, 20]),
        'n_components': random.choice([5, 10, 15])
    }

def random_augmentation_image():
    return {
        'random_flip': random.choice([True, False]),
        'random_rotation': random.choice([True, False]),
        'rotation_angle': random.choice([10, 15, 30]),
        'color_jitter': random.choice([True, False]),
        'brightness': random.choice([0.1, 0.2, 0.3]),
        'contrast': random.choice([0.1, 0.2, 0.3]),
        'saturation': random.choice([0.1, 0.2, 0.3]),
        'random_crop': random.choice([True, False]),
        'normalize': True,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    }

def random_mlp_layer():
    return {
        'units': random.choice([16, 32, 64, 128, 256]),
        'activation': random.choice(['relu', 'tanh', 'sigmoid']),
        'dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
    }

def random_mlp_arch(min_layers=1, max_layers=4):
    return [random_mlp_layer() for _ in range(random.randint(min_layers, max_layers))]

def random_cnn_block():
    return {
        'filters': random.choice([16, 32, 64, 128]),
        'kernel_size': random.choice([3, 5]),
        'stride': 1,
        'use_bn': random.choice([True, False]),
        'activation': random.choice(['relu', 'leaky_relu', 'elu']),
        'pool': random.choice([True, False]),
        'pool_type': random.choice(['max', 'avg']),
        'pool_size': 2,
        'dropout': random.choice([0.0, 0.1, 0.2, 0.3])
    }

def random_cnn_arch(min_blocks=2, max_blocks=6):
    arch = [random_cnn_block() for _ in range(random.randint(min_blocks, max_blocks))]
    # Add FC layers configuration to first block
    arch[0]['fc_layers'] = random.choice([[128], [256], [128, 64], [256, 128]])
    return arch

def random_model_type_tabular():
    return random.choice(['mlp', 'rf', 'logreg', 'gbm', 'svm'])

# ============================================================================
# MUTATION OPERATORS
# ============================================================================
def mutate_preprocessing_tabular(config):
    new = copy.deepcopy(config)
    field = random.choice(list(config.keys()))
    
    if field == 'numeric_impute':
        new['numeric_impute'] = random.choice(['mean', 'median', 'mode'])
    elif field == 'categorical_impute':
        new['categorical_impute'] = random.choice(['mode', 'constant'])
    elif field == 'outlier_method':
        new['outlier_method'] = random.choice(['none', 'iqr', 'zscore', 'clip'])
    elif field == 'outlier_threshold':
        new['outlier_threshold'] = random.choice([1.5, 2.0, 3.0])
    elif field in ['create_interactions', 'create_ratios', 'create_binning', 'create_polynomial', 'feature_selection']:
        new[field] = not new[field]
    elif field == 'n_bins':
        new['n_bins'] = random.choice([3, 5, 10])
    elif field == 'categorical_encoding':
        new['categorical_encoding'] = random.choice(['label', 'frequency'])
    elif field == 'scaling':
        new['scaling'] = random.choice(['standard', 'minmax', 'robust', 'none'])
    elif field == 'feature_selection_method':
        new['feature_selection_method'] = random.choice(['rf_importance', 'selectk', 'pca'])
    elif field == 'selection_threshold':
        new['selection_threshold'] = random.choice([0.01, 0.05, 0.1])
    elif field == 'n_features_to_select':
        new['n_features_to_select'] = random.choice([10, 15, 20])
    elif field == 'n_components':
        new['n_components'] = random.choice([5, 10, 15])
    
    return new

def mutate_augmentation_image(config):
    new = copy.deepcopy(config)
    field = random.choice(['random_flip', 'random_rotation', 'rotation_angle', 
                          'color_jitter', 'brightness', 'contrast', 'saturation', 'random_crop'])
    
    if field in ['random_flip', 'random_rotation', 'color_jitter', 'random_crop']:
        new[field] = not new[field]
    elif field == 'rotation_angle':
        new['rotation_angle'] = random.choice([10, 15, 30])
    elif field in ['brightness', 'contrast', 'saturation']:
        new[field] = random.choice([0.1, 0.2, 0.3])
    
    return new

def mutate_mlp_arch(arch, max_layers=6):
    new = copy.deepcopy(arch)
    ops = ['add', 'remove', 'modify']
    op = random.choice(ops)
    
    if op == 'add' and len(new) < max_layers:
        pos = random.randint(0, len(new))
        new.insert(pos, random_mlp_layer())
    elif op == 'remove' and len(new) > 1:
        pos = random.randrange(len(new))
        new.pop(pos)
    else:
        pos = random.randrange(len(new))
        field = random.choice(['units', 'activation', 'dropout'])
        if field == 'units':
            new[pos]['units'] = random.choice([16, 32, 64, 128, 256])
        elif field == 'activation':
            new[pos]['activation'] = random.choice(['relu', 'tanh', 'sigmoid'])
        else:
            new[pos]['dropout'] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
    
    return new

def mutate_cnn_arch(arch, max_blocks=8):
    new = copy.deepcopy(arch)
    ops = ['add', 'remove', 'modify']
    op = random.choice(ops)
    
    if op == 'add' and len(new) < max_blocks:
        pos = random.randint(0, len(new))
        new.insert(pos, random_cnn_block())
    elif op == 'remove' and len(new) > 2:  # Minimum 2 blocks
        pos = random.randrange(len(new))
        new.pop(pos)
    else:
        pos = random.randrange(len(new))
        field = random.choice(['filters', 'kernel_size', 'activation', 'pool', 'dropout'])
        if field == 'filters':
            new[pos]['filters'] = random.choice([16, 32, 64, 128])
        elif field == 'kernel_size':
            new[pos]['kernel_size'] = random.choice([3, 5])
        elif field == 'activation':
            new[pos]['activation'] = random.choice(['relu', 'leaky_relu', 'elu'])
        elif field == 'pool':
            new[pos]['pool'] = not new[pos]['pool']
        elif field == 'dropout':
            new[pos]['dropout'] = random.choice([0.0, 0.1, 0.2, 0.3])
    
    return new

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
def train_and_evaluate_model_tabular(X_train, y_train, X_val, y_val, model_type, arch, device, num_classes, train_epochs=3, lr=0.001):
    try:
        if model_type == 'mlp':
            model = MLP(arch, X_train.shape[1], num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            
            # Training
            model.train()
            for epoch in range(train_epochs):
                n_samples = len(X_train)
                indices = np.random.permutation(n_samples)
                batch_size = min(32, n_samples)
                
                for i in range(0, n_samples, batch_size):
                    batch_idx = indices[i:i+batch_size]
                    X_batch = torch.FloatTensor(X_train[batch_idx]).to(device)
                    y_batch = torch.LongTensor(y_train[batch_idx]).to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                outputs = model(X_val_tensor)
                _, y_pred = outputs.max(1)
                y_pred = y_pred.cpu().numpy()
        
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        
        elif model_type == 'logreg':
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        
        elif model_type == 'gbm':
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        
        elif model_type == 'svm':
            model = SVC(kernel='rbf', random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        fitness = 0.7 * acc + 0.3 * f1
        
        return fitness, metrics, model
    
    except Exception as e:
        st.warning(f"Error in tabular model training: {str(e)}")
        return 0.0, {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}, None

def train_and_evaluate_model_image(train_loader, val_loader, arch, augmentation_config, device, num_classes, 
                                   train_epochs=10, lr=0.001, img_size=32, classification_type='multiclass'):
    try:
        model = CNN(arch, num_classes, input_channels=3, img_size=img_size).to(device)
        
        if classification_type == 'multilabel':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Training
        model.train()
        for epoch in range(train_epochs):
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                if classification_type == 'multilabel':
                    loss = criterion(outputs, labels.float())
                else:
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                
                if classification_type == 'multilabel':
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        if classification_type == 'multilabel':
            # For multilabel, calculate metrics differently
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
            precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
        else:
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        fitness = 0.7 * acc + 0.3 * f1
        
        return fitness, metrics, model
    
    except Exception as e:
        st.warning(f"Error in image model training: {str(e)}")
        return 0.0, {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}, None

# ============================================================================
# EVOLUTION FOR TABULAR DATA
# ============================================================================
def run_evolution_tabular(X_train, y_train, X_val, y_val, args, device, num_classes, progress_callback=None):
    population = []
    evolution_history = []
    
    # Initialize population
    if progress_callback:
        progress_callback(0, "Initializing population...")
    
    for i in range(args['pop_size']):
        prep_config = random_preprocessing_tabular()
        model_type = random_model_type_tabular()
        arch = random_mlp_arch(args['min_layers'], args['max_layers']) if model_type == 'mlp' else None
        
        try:
            pipeline = EDA_Pipeline(prep_config)
            X_train_proc, y_train_proc = pipeline.fit_transform(X_train.copy(), y_train.copy())
            X_val_proc = pipeline.transform(X_val.copy())
            
            fitness, metrics, _ = train_and_evaluate_model_tabular(
                X_train_proc, y_train_proc, X_val_proc, y_val,
                model_type, arch, device, num_classes,
                args['train_epochs'], args['lr']
            )
            
            individual = Individual(
                preprocessing=prep_config,
                augmentation=None,
                arch=arch,
                model_type=model_type,
                fitness=fitness,
                metrics=metrics
            )
            population.append(individual)
            
            if progress_callback:
                progress_callback((i + 1) / args['pop_size'] * 0.3, f"Initialized {i+1}/{args['pop_size']}")
        
        except Exception as e:
            st.warning(f"Failed to initialize individual {i+1}: {str(e)}")
    
    # Evolution
    for gen in range(args['generations']):
        if progress_callback:
            base_progress = 0.3 + (gen / args['generations']) * 0.7
            progress_callback(base_progress, f"Generation {gen+1}/{args['generations']}")
        
        # Sort population
        population = sorted(population, key=lambda x: x.fitness if x.fitness is not None else 0.0, reverse=True)
        
        # Record best of generation
        best = population[0]
        evolution_history.append({
            'generation': gen + 1,
            'best_fitness': best.fitness,
            'best_accuracy': best.metrics['accuracy'],
            'best_f1': best.metrics['f1'],
            'model_type': best.model_type
        })
        
        # Create next generation
        next_pop = []
        
        # Elitism
        elite_count = max(1, int(args['elitism'] * len(population)))
        next_pop.extend(population[:elite_count])
        
        # Generate children
        while len(next_pop) < args['pop_size']:
            # Tournament selection
            tournament = random.sample(population, k=min(args['tournament_k'], len(population)))
            parent = max(tournament, key=lambda x: x.fitness if x.fitness is not None else 0.0)
            
            # Mutation
            mutation_type = random.random()
            
            if mutation_type < 0.33:  # Mutate preprocessing only
                child_prep = mutate_preprocessing_tabular(parent.preprocessing)
                child_model_type = parent.model_type
                child_arch = parent.arch
            elif mutation_type < 0.66:  # Mutate architecture only (if MLP)
                child_prep = parent.preprocessing
                child_model_type = parent.model_type
                child_arch = mutate_mlp_arch(parent.arch, args['max_layers']) if parent.model_type == 'mlp' else None
            else:  # Change model type
                child_prep = parent.preprocessing
                child_model_type = random_model_type_tabular()
                child_arch = random_mlp_arch(args['min_layers'], args['max_layers']) if child_model_type == 'mlp' else None
            
            try:
                pipeline = EDA_Pipeline(child_prep)
                X_train_proc, y_train_proc = pipeline.fit_transform(X_train.copy(), y_train.copy())
                X_val_proc = pipeline.transform(X_val.copy())
                
                fitness, metrics, _ = train_and_evaluate_model_tabular(
                    X_train_proc, y_train_proc, X_val_proc, y_val,
                    child_model_type, child_arch, device, num_classes,
                    args['train_epochs'], args['lr']
                )
                
                child = Individual(
                    preprocessing=child_prep,
                    augmentation=None,
                    arch=child_arch,
                    model_type=child_model_type,
                    fitness=fitness,
                    metrics=metrics
                )
                next_pop.append(child)
            
            except Exception as e:
                if len(next_pop) < args['pop_size']:
                    next_pop.append(parent)
        
        population = next_pop[:args['pop_size']]
    
    # Final best
    population = sorted(population, key=lambda x: x.fitness if x.fitness is not None else 0.0, reverse=True)
    best = population[0]
    
    return best, evolution_history

# ============================================================================
# EVOLUTION FOR IMAGE DATA
# ============================================================================
def run_evolution_image(train_data, val_data, args, device, num_classes, img_size, classification_type, progress_callback=None):
    population = []
    evolution_history = []
    
    # Initialize population
    if progress_callback:
        progress_callback(0, "Initializing population...")
    
    for i in range(args['pop_size']):
        augmentation_config = random_augmentation_image()
        arch = random_cnn_arch(args['min_cnn_blocks'], args['max_cnn_blocks'])
        
        try:
            # Create data loaders with augmentation
            aug_pipeline = ImageAugmentationPipeline(augmentation_config, img_size)
            train_transform = aug_pipeline.get_transform(is_training=True)
            val_transform = aug_pipeline.get_transform(is_training=False)
            
            train_dataset = ImageDataset(train_data['paths'], train_data['labels'], train_transform)
            val_dataset = ImageDataset(val_data['paths'], val_data['labels'], val_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2)
            
            fitness, metrics, _ = train_and_evaluate_model_image(
                train_loader, val_loader, arch, augmentation_config, device, num_classes,
                args['train_epochs_cnn'], args['lr'], img_size, classification_type
            )
            
            individual = Individual(
                preprocessing=None,
                augmentation=augmentation_config,
                arch=arch,
                model_type='cnn',
                fitness=fitness,
                metrics=metrics
            )
            population.append(individual)
            
            if progress_callback:
                progress_callback((i + 1) / args['pop_size'] * 0.3, f"Initialized {i+1}/{args['pop_size']}")
        
        except Exception as e:
            st.warning(f"Failed to initialize individual {i+1}: {str(e)}")
    
    # Evolution
    for gen in range(args['generations']):
        if progress_callback:
            base_progress = 0.3 + (gen / args['generations']) * 0.7
            progress_callback(base_progress, f"Generation {gen+1}/{args['generations']}")
        
        # Sort population
        population = sorted(population, key=lambda x: x.fitness if x.fitness is not None else 0.0, reverse=True)
        
        # Record best of generation
        best = population[0]
        evolution_history.append({
            'generation': gen + 1,
            'best_fitness': best.fitness,
            'best_accuracy': best.metrics['accuracy'],
            'best_f1': best.metrics['f1'],
            'model_type': 'cnn'
        })
        
        # Create next generation
        next_pop = []
        
        # Elitism
        elite_count = max(1, int(args['elitism'] * len(population)))
        next_pop.extend(population[:elite_count])
        
        # Generate children
        while len(next_pop) < args['pop_size']:
            # Tournament selection
            tournament = random.sample(population, k=min(args['tournament_k'], len(population)))
            parent = max(tournament, key=lambda x: x.fitness if x.fitness is not None else 0.0)
            
            # Mutation
            mutation_type = random.random()
            
            if mutation_type < 0.5:  # Mutate augmentation
                child_aug = mutate_augmentation_image(parent.augmentation)
                child_arch = parent.arch
            else:  # Mutate architecture
                child_aug = parent.augmentation
                child_arch = mutate_cnn_arch(parent.arch, args['max_cnn_blocks'])
            
            try:
                # Create data loaders with augmentation
                aug_pipeline = ImageAugmentationPipeline(child_aug, img_size)
                train_transform = aug_pipeline.get_transform(is_training=True)
                val_transform = aug_pipeline.get_transform(is_training=False)
                
                train_dataset = ImageDataset(train_data['paths'], train_data['labels'], train_transform)
                val_dataset = ImageDataset(val_data['paths'], val_data['labels'], val_transform)
                
                train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2)
                
                fitness, metrics, _ = train_and_evaluate_model_image(
                    train_loader, val_loader, child_arch, child_aug, device, num_classes,
                    args['train_epochs_cnn'], args['lr'], img_size, classification_type
                )
                
                child = Individual(
                    preprocessing=None,
                    augmentation=child_aug,
                    arch=child_arch,
                    model_type='cnn',
                    fitness=fitness,
                    metrics=metrics
                )
                next_pop.append(child)
            
            except Exception as e:
                if len(next_pop) < args['pop_size']:
                    next_pop.append(parent)
        
        population = next_pop[:args['pop_size']]
    
    # Final best
    population = sorted(population, key=lambda x: x.fitness if x.fitness is not None else 0.0, reverse=True)
    best = population[0]
    
    return best, evolution_history

# ============================================================================
# IMAGE DATA LOADING
# ============================================================================
def load_image_dataset_from_zip(zip_file):
    """
    Load image dataset from uploaded zip file
    Expected format: zip contains folders named by class labels, each containing images
    """
    try:
        # Create temp directory
        temp_dir = "temp_dataset"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract zip
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find all image files and their labels
        image_paths = []
        labels = []
        label_names = []
        
        # Get all subdirectories (class folders)
        class_folders = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        class_folders = sorted(class_folders)  # Sort for consistent label encoding
        
        label_to_idx = {label: idx for idx, label in enumerate(class_folders)}
        
        # Iterate through class folders
        for class_folder in class_folders:
            class_path = os.path.join(temp_dir, class_folder)
            
            # Get all image files in this class folder
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(class_path, img_file)
                    image_paths.append(img_path)
                    labels.append(label_to_idx[class_folder])
                    label_names.append(class_folder)
        
        return {
            'paths': image_paths,
            'labels': np.array(labels),
            'label_names': label_names,
            'num_classes': len(class_folders),
            'class_names': class_folders,
            'label_to_idx': label_to_idx
        }
    
    except Exception as e:
        st.error(f"Error loading image dataset: {str(e)}")
        return None

# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    st.markdown('<div class="main-header">🧬 EvoNAS Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evolutionary Neural Architecture Search for Automated ML</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📊 Data Type Selection")
        data_type = st.radio("Select Data Type", ["Tabular (CSV)", "Image (Computer Vision)"], index=0)
        
        if data_type == "Tabular (CSV)":
            st.subheader("📊 Data Upload")
            uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
        else:
            st.subheader("🖼️ Image Dataset Upload")
            uploaded_file = st.file_uploader("Upload ZIP file with image dataset", type=['zip'])
            st.info("📁 Expected format: ZIP file containing folders named by class labels, each with images")
            
            classification_type = st.radio(
                "Classification Type",
                ["Multi-class", "Multi-label"],
                help="Multi-class: Each image belongs to one class. Multi-label: Each image can belong to multiple classes."
            )
        
        st.subheader("🧬 Evolution Parameters")
        pop_size = st.slider("Population Size", 4, 20, 6 if data_type == "Image (Computer Vision)" else 8, 2)
        generations = st.slider("Generations", 2, 15, 3 if data_type == "Image (Computer Vision)" else 5, 1)
        elitism = st.slider("Elitism Rate", 0.1, 0.5, 0.25, 0.05)
        tournament_k = st.slider("Tournament Size", 2, 5, 3, 1)
        
        if data_type == "Tabular (CSV)":
            st.subheader("🏗️ Architecture Constraints (MLP)")
            min_layers = st.slider("Min Layers", 1, 3, 1, 1)
            max_layers = st.slider("Max Layers", 3, 8, 5, 1)
        else:
            st.subheader("🏗️ Architecture Constraints (CNN)")
            min_cnn_blocks = st.slider("Min CNN Blocks", 2, 4, 2, 1)
            max_cnn_blocks = st.slider("Max CNN Blocks", 4, 8, 6, 1)
            img_size = st.select_slider("Image Size", options=[28, 32, 64, 128], value=32)
        
        st.subheader("🎯 Training Parameters")
        if data_type == "Tabular (CSV)":
            train_epochs = st.slider("Epochs per Evaluation", 1, 10, 3, 1)
        else:
            train_epochs_cnn = st.slider("Epochs per Evaluation (CNN)", 5, 30, 10, 5)
            batch_size = st.slider("Batch Size", 16, 128, 32, 16)
        
        learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
        
        st.subheader("🔧 Advanced Options")
        random_seed = st.number_input("Random Seed", 0, 9999, 42)
        use_gpu = st.checkbox("Use GPU (if available)", True)
    
    # Main content
    if uploaded_file is None:
        st.info(f"👈 Please upload a {'CSV' if data_type == 'Tabular (CSV)' else 'ZIP'} dataset to begin")
        
        with st.expander("📖 How to Use EvoNAS Tool"):
            if data_type == "Tabular (CSV)":
                st.markdown("""
                ### Tabular Data (CSV) Mode
                
                1. **Upload CSV Dataset**: Your data in CSV format
                2. **Select Target Column**: Choose prediction target
                3. **Configure Evolution**: Set population, generations, epochs
                4. **Start Evolution**: Click button and wait for optimization
                5. **Review Results**: Examine best configuration and metrics
                6. **Export**: Download configuration and history
                
                **Optimizes**: Preprocessing, Feature Engineering, Model Selection (MLP, RF, LogReg, GBM, SVM)
                """)
            else:
                st.markdown("""
                ### Image Data (Computer Vision) Mode
                
                1. **Prepare Dataset**: Create ZIP file with this structure:
```
                   dataset.zip
                   ├── class_1_name/
                   │   ├── image1.jpg
                   │   ├── image2.jpg
                   │   └── ...
                   ├── class_2_name/
                   │   ├── image1.jpg
                   │   └── ...
                   └── class_3_name/
                       └── ...
```
                2. **Upload ZIP File**: Drag and drop your prepared dataset
                3. **Select Classification Type**: Multi-class or Multi-label
                4. **Configure Evolution**: Set population, generations, epochs
                5. **Start Evolution**: CNN architecture search begins
                6. **Review Results**: Best CNN architecture and augmentation strategy
                
                **Optimizes**: CNN Architecture, Data Augmentation, Image Preprocessing
                """)
        
        with st.expander("📊 Example Formats"):
            if data_type == "Tabular (CSV)":
                example_df = pd.DataFrame({
                    'feature1': [1.2, 2.3, 3.1, 4.5, 5.2],
                    'feature2': [10, 20, 15, 30, 25],
                    'feature3': ['A', 'B', 'A', 'C', 'B'],
                    'target': [0, 1, 0, 1, 1]
                })
                st.dataframe(example_df)
            else:
                st.markdown("""
                **Image Dataset Structure:**
```
                MNIST_dataset.zip
                ├── digit_0/
                │   ├── img_001.png
                │   ├── img_002.png
                │   └── ...
                ├── digit_1/
                │   ├── img_101.png
                │   └── ...
                └── digit_9/
                    └── ...
```
                """)
        
        return
    
    # ============================================================================
    # TABULAR DATA PROCESSING
    # ============================================================================
    if data_type == "Tabular (CSV)":
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # [Rest of tabular processing code remains the same as before]
            # ... (keeping all the existing tabular data processing code)
            
            st.markdown("---")
            st.header("🎯 Data Configuration")
            
            # Column removal feature
            with st.expander("🗑️ Remove Unwanted Columns (Optional)", expanded=False):
                st.write("Select columns you want to **remove** from the dataset before training:")
                
                cols_per_row = 4
                all_columns = df.columns.tolist()
                
                if 'columns_to_remove' not in st.session_state:
                    st.session_state['columns_to_remove'] = []
                
                num_cols = len(all_columns)
                num_rows = (num_cols + cols_per_row - 1) // cols_per_row
                
                selected_to_remove = []
                
                for row in range(num_rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        idx = row * cols_per_row + col_idx
                        if idx < num_cols:
                            column_name = all_columns[idx]
                            with cols[col_idx]:
                                if st.checkbox(
                                    column_name, 
                                    key=f"remove_{column_name}",
                                    value=column_name in st.session_state['columns_to_remove']
                                ):
                                    selected_to_remove.append(column_name)
                
                st.session_state['columns_to_remove'] = selected_to_remove
                
                if selected_to_remove:
                    st.warning(f"⚠️ {len(selected_to_remove)} column(s) will be removed: {', '.join(selected_to_remove)}")
                    
                    col_btn1, col_btn2 = st.columns([1, 4])
                    with col_btn1:
                        if st.button("✅ Apply Removal", type="primary"):
                            df = df.drop(columns=selected_to_remove)
                            st.session_state['columns_to_remove'] = []
                            st.success(f"Removed {len(selected_to_remove)} column(s)")
                            st.rerun()
                    with col_btn2:
                        if st.button("🔄 Clear Selection"):
                            st.session_state['columns_to_remove'] = []
                            st.rerun()
                else:
                    st.info("ℹ️ No columns selected for removal. All columns will be used.")
            
            remaining_columns = [col for col in df.columns if col not in st.session_state.get('columns_to_remove', [])]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_column = st.selectbox(
                    "Select Target Column",
                    options=remaining_columns,
                    help="Choose the column you want to predict"
                )
            
            with col2:
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            
            with col3:
                val_size = st.slider("Validation Size", 0.1, 0.4, 0.15, 0.05)
            
            # Data preview
            with st.expander("👀 Data Preview", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**First 10 rows:**")
                    display_df = df[[col for col in df.columns if col not in st.session_state.get('columns_to_remove', [])]]
                    st.dataframe(display_df.head(10))
                with col2:
                    st.write("**Dataset Info:**")
                    st.write(f"- Shape: {display_df.shape}")
                    st.write(f"- Missing values: {display_df.isnull().sum().sum()}")
                    st.write(f"- Numeric columns: {len(display_df.select_dtypes(include=[np.number]).columns)}")
                    st.write(f"- Categorical columns: {len(display_df.select_dtypes(exclude=[np.number]).columns)}")
            
            if target_column is None:
                st.warning("⚠️ Please select a target column to continue")
                return
            
            if st.session_state.get('columns_to_remove'):
                df = df.drop(columns=st.session_state['columns_to_remove'])
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                st.info(f"Target column encoded. Classes: {list(le.classes_)}")
            
            num_classes = len(np.unique(y))
            st.info(f"🎯 Classification task detected: {num_classes} classes")
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size+val_size, random_state=random_seed, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_size/(test_size+val_size), 
                random_state=random_seed, stratify=y_temp
            )
            
            st.write("**Data Split:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Training Samples", len(X_train))
            col2.metric("Validation Samples", len(X_val))
            col3.metric("Test Samples", len(X_test))
            
            device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            if use_gpu and torch.cuda.is_available():
                st.success(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.info("💻 Using CPU")
            
            st.markdown("---")
            if st.button("🚀 Start Evolution", type="primary", use_container_width=True):
                args = {
                    'pop_size': pop_size,
                    'generations': generations,
                    'elitism': elitism,
                    'tournament_k': tournament_k,
                    'min_layers': min_layers,
                    'max_layers': max_layers,
                    'train_epochs': train_epochs,
                    'lr': learning_rate
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                with st.spinner("🧬 Evolution in progress..."):
                    start_time = time.time()
                    
                    best_individual, evolution_history = run_evolution_tabular(
                        X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
                        y_train,
                        X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
                        y_val,
                        args,
                        device,
                        num_classes,
                        progress_callback=update_progress
                    )
                    
                    elapsed_time = time.time() - start_time
                
                progress_bar.progress(1.0)
                status_text.text("✅ Evolution completed!")
                
                st.success(f"🎉 Evolution completed in {elapsed_time:.2f} seconds!")
                
                # Results display (same as before)
                st.markdown("---")
                st.header("📊 Results")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Best Accuracy", f"{best_individual.metrics['accuracy']:.4f}")
                col2.metric("Best F1 Score", f"{best_individual.metrics['f1']:.4f}")
                col3.metric("Precision", f"{best_individual.metrics['precision']:.4f}")
                col4.metric("Recall", f"{best_individual.metrics['recall']:.4f}")
                
                # Evolution history plot
                st.subheader("📈 Evolution Progress")
                history_df = pd.DataFrame(evolution_history)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history_df['generation'], y=history_df['best_accuracy'],
                                        mode='lines+markers', name='Accuracy', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=history_df['generation'], y=history_df['best_f1'],
                                        mode='lines+markers', name='F1 Score', line=dict(color='red', width=2)))
                fig.update_layout(title='Best Performance per Generation',
                                xaxis_title='Generation',
                                yaxis_title='Score',
                                hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                # Best configuration details
                st.subheader("🏆 Best Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Model Type:**")
                    st.info(f"🤖 {best_individual.model_type.upper()}")
                    
                    if best_individual.model_type == 'mlp' and best_individual.arch:
                        st.write("**Architecture:**")
                        arch_data = []
                        for i, layer in enumerate(best_individual.arch):
                            arch_data.append({
                                'Layer': f'Layer {i+1}',
                                'Units': layer['units'],
                                'Activation': layer['activation'],
                                'Dropout': layer['dropout']
                            })
                        st.dataframe(pd.DataFrame(arch_data), use_container_width=True)
                
                with col2:
                    st.write("**Preprocessing Configuration:**")
                    prep_config = best_individual.preprocessing
                    prep_display = {
                        'Numeric Imputation': prep_config['numeric_impute'],
                        'Categorical Imputation': prep_config['categorical_impute'],
                        'Outlier Method': prep_config['outlier_method'],
                        'Scaling': prep_config['scaling'],
                        'Feature Selection': '✓' if prep_config['feature_selection'] else '✗',
                    }
                    st.json(prep_display)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    # ============================================================================
    # IMAGE DATA PROCESSING
    # ============================================================================
    else:  # Image mode
        try:
            st.info("Loading image dataset... This may take a moment.")
            
            dataset_info = load_image_dataset_from_zip(uploaded_file)
            
            if dataset_info is None:
                st.error("Failed to load dataset")
                return
            
            st.success(f"✅ Dataset loaded: {len(dataset_info['paths'])} images, {dataset_info['num_classes']} classes")
            
            st.markdown("---")
            st.header("🖼️ Dataset Information")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Images", len(dataset_info['paths']))
            col2.metric("Number of Classes", dataset_info['num_classes'])
            col3.metric("Image Size", f"{img_size}×{img_size}")
            
            # Class distribution
            with st.expander("📊 Class Distribution", expanded=True):
                class_counts = pd.Series(dataset_info['labels']).value_counts().sort_index()
                class_names_ordered = [dataset_info['class_names'][i] for i in class_counts.index]
                
                fig = px.bar(x=class_names_ordered, y=class_counts.values,
                           labels={'x': 'Class', 'y': 'Count'},
                           title='Images per Class')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show sample images
            with st.expander("🖼️ Sample Images", expanded=False):
                num_samples = min(10, len(dataset_info['paths']))
                sample_indices = random.sample(range(len(dataset_info['paths'])), num_samples)
                
                cols = st.columns(5)
                for idx, sample_idx in enumerate(sample_indices):
                    img_path = dataset_info['paths'][sample_idx]
                    label = dataset_info['class_names'][dataset_info['labels'][sample_idx]]
                    
                    with cols[idx % 5]:
                        img = Image.open(img_path)
                        st.image(img, caption=f"Class: {label}", use_container_width=True)
            
            # Train/val/test split
            st.subheader("🔄 Data Split Configuration")
            col1, col2 = st.columns(2)
            with col1:
                test_size_img = st.slider("Test Size", 0.1, 0.3, 0.15, 0.05, key="test_size_img")
            with col2:
                val_size_img = st.slider("Validation Size", 0.1, 0.3, 0.15, 0.05, key="val_size_img")
            
            # Split data
            indices = np.arange(len(dataset_info['paths']))
            train_idx, temp_idx = train_test_split(
                indices, test_size=test_size_img+val_size_img, 
                random_state=random_seed, stratify=dataset_info['labels']
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=test_size_img/(test_size_img+val_size_img),
                random_state=random_seed, stratify=dataset_info['labels'][temp_idx]
            )
            
            train_data = {
                'paths': [dataset_info['paths'][i] for i in train_idx],
                'labels': dataset_info['labels'][train_idx]
            }
            val_data = {
                'paths': [dataset_info['paths'][i] for i in val_idx],
                'labels': dataset_info['labels'][val_idx]
            }
            test_data = {
                'paths': [dataset_info['paths'][i] for i in test_idx],
                'labels': dataset_info['labels'][test_idx]
            }
            
            st.write("**Data Split:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Training Images", len(train_data['paths']))
            col2.metric("Validation Images", len(val_data['paths']))
            col3.metric("Test Images", len(test_data['paths']))
            
            device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            if use_gpu and torch.cuda.is_available():
                st.success(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.warning("💻 Using CPU (Image training will be slow. GPU highly recommended!)")
            
            st.markdown("---")
            if st.button("🚀 Start Evolution", type="primary", use_container_width=True):
                classification_type_value = 'multilabel' if classification_type == "Multi-label" else 'multiclass'
                
                args = {
                    'pop_size': pop_size,
                    'generations': generations,
                    'elitism': elitism,
                    'tournament_k': tournament_k,
                    'min_cnn_blocks': min_cnn_blocks,
                    'max_cnn_blocks': max_cnn_blocks,
                    'train_epochs_cnn': train_epochs_cnn,
                    'batch_size': batch_size,
                    'lr': learning_rate
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                with st.spinner("🧬 CNN Evolution in progress..."):
                    start_time = time.time()
                    
                    best_individual, evolution_history = run_evolution_image(
                        train_data, val_data, args, device, 
                        dataset_info['num_classes'], img_size, 
                        classification_type_value,
                        progress_callback=update_progress
                    )
                    
                    elapsed_time = time.time() - start_time
                
                progress_bar.progress(1.0)
                status_text.text("✅ Evolution completed!")
                
                st.success(f"🎉 CNN Evolution completed in {elapsed_time/60:.2f} minutes!")
                
                # Results
                st.markdown("---")
                st.header("📊 Results")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Best Accuracy", f"{best_individual.metrics['accuracy']:.4f}")
                col2.metric("Best F1 Score", f"{best_individual.metrics['f1']:.4f}")
                col3.metric("Precision", f"{best_individual.metrics['precision']:.4f}")
                col4.metric("Recall", f"{best_individual.metrics['recall']:.4f}")
                
                # Evolution history plot
                st.subheader("📈 Evolution Progress")
                history_df = pd.DataFrame(evolution_history)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history_df['generation'], y=history_df['best_accuracy'],
                                        mode='lines+markers', name='Accuracy', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=history_df['generation'], y=history_df['best_f1'],
                                        mode='lines+markers', name='F1 Score', line=dict(color='red', width=2)))
                fig.update_layout(title='Best CNN Performance per Generation',
                                xaxis_title='Generation',
                                yaxis_title='Score',
                                hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                # Best configuration
                st.subheader("🏆 Best CNN Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**CNN Architecture:**")
                    arch_data = []
                    for i, block in enumerate(best_individual.arch):
                        arch_data.append({
                            'Block': f'Block {i+1}',
                            'Filters': block['filters'],
                            'Kernel': f"{block['kernel_size']}×{block['kernel_size']}",
                            'Activation': block['activation'],
                            'Pool': '✓' if block['pool'] else '✗',
                            'Dropout': block['dropout']
                        })
                    st.dataframe(pd.DataFrame(arch_data), use_container_width=True)
                
                with col2:
                    st.write("**Data Augmentation:**")
                    aug_config = best_individual.augmentation
                    aug_display = {
                        'Random Flip': '✓' if aug_config['random_flip'] else '✗',
                        'Random Rotation': '✓' if aug_config['random_rotation'] else '✗',
                        'Rotation Angle': f"{aug_config['rotation_angle']}°" if aug_config['random_rotation'] else 'N/A',
                        'Color Jitter': '✓' if aug_config['color_jitter'] else '✗',
                        'Random Crop': '✓' if aug_config['random_crop'] else '✗',
                        'Normalization': '✓' if aug_config['normalize'] else '✗'
                    }
                    st.json(aug_display)
                
                # Export results
                st.subheader("💾 Export Results")
                
                results_dict = {
                    'model_type': 'cnn',
                    'architecture': best_individual.arch,
                    'augmentation': best_individual.augmentation,
                    'validation_metrics': best_individual.metrics,
                    'evolution_history': evolution_history,
                    'configuration': args,
                    'dataset_info': {
                        'num_classes': dataset_info['num_classes'],
                        'class_names': dataset_info['class_names'],
                        'total_images': len(dataset_info['paths']),
                        'img_size': img_size
                    }
                }
                
                results_json = json.dumps(results_dict, indent=2, default=str)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Configuration (JSON)",
                        data=results_json,
                        file_name="evonas_cnn_config.json",
                        mime="application/json"
                    )
                
                with col2:
                    history_csv = pd.DataFrame(evolution_history).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Evolution History (CSV)",
                        data=history_csv,
                        file_name="evonas_cnn_history.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    main()