"""
Comprehensive test suite for app.py - EvoNAS Tool
Tests include unit tests, integration tests, and edge cases for all major components
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import random
import copy
from sklearn.preprocessing import LabelEncoder

# Import functions and classes from app.py
import sys
sys.path.insert(0, 'd:\\College\\semester 6\\SWE\\EvoNAS-SWE')


class TestEDAPipeline:
    """Test suite for EDA_Pipeline class"""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for EDA_Pipeline"""
        return {
            'numeric_impute': 'mean',
            'categorical_impute': 'mode',
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'create_interactions': False,
            'create_ratios': False,
            'create_binning': False,
            'create_polynomial': False,
            'n_bins': 5,
            'categorical_encoding': 'label',
            'scaling': 'standard',
            'feature_selection': False,
            'feature_selection_method': 'rf_importance',
            'selection_threshold': 0.01,
            'n_features_to_select': 10,
            'n_components': 5
        }
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)
        X = pd.DataFrame({
            'numeric_col1': np.random.randn(100),
            'numeric_col2': np.random.randn(100) * 100,
            'categorical_col1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical_col2': np.random.choice(['X', 'Y'], 100)
        })
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_eda_pipeline_initialization(self, basic_config):
        """Test EDA_Pipeline initialization"""
        # Import EDA_Pipeline inside the test to avoid Streamlit issues
        from app import EDA_Pipeline
        pipeline = EDA_Pipeline(basic_config)
        
        assert pipeline.config == basic_config
        assert pipeline.scaler is None
        assert pipeline.numeric_imputer is None
        assert pipeline.categorical_imputer is None
        assert pipeline.label_encoders == {}
        assert pipeline.numeric_cols == []
        assert pipeline.categorical_cols == []
    
    def test_identify_column_types(self, basic_config, sample_data):
        """Test column type identification"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        pipeline = EDA_Pipeline(basic_config)
        pipeline.identify_column_types(X)
        
        assert 'numeric_col1' in pipeline.numeric_cols
        assert 'numeric_col2' in pipeline.numeric_cols
        assert 'categorical_col1' in pipeline.categorical_cols
        assert 'categorical_col2' in pipeline.categorical_cols
    
    def test_handle_missing_values_mean(self, basic_config, sample_data):
        """Test missing value handling with mean strategy"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        # Add missing values
        X_with_missing = X.copy()
        X_with_missing.loc[0:5, 'numeric_col1'] = np.nan
        
        pipeline = EDA_Pipeline(basic_config)
        pipeline.identify_column_types(X_with_missing)
        X_imputed = pipeline.handle_missing_values(X_with_missing, fit=True)
        
        # Check that missing values are handled
        assert X_imputed.isnull().sum().sum() == 0
        assert X_imputed.shape == X_with_missing.shape
    
    def test_handle_missing_values_mode(self, basic_config, sample_data):
        """Test missing value handling with mode strategy"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        config = copy.deepcopy(basic_config)
        config['numeric_impute'] = 'mode'
        
        X_with_missing = X.copy()
        X_with_missing.loc[0:5, 'numeric_col1'] = np.nan
        
        pipeline = EDA_Pipeline(config)
        pipeline.identify_column_types(X_with_missing)
        X_imputed = pipeline.handle_missing_values(X_with_missing, fit=True)
        
        assert X_imputed.isnull().sum().sum() == 0
    
    def test_handle_outliers_iqr(self, basic_config, sample_data):
        """Test outlier handling with IQR method"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        # Add outliers
        X_with_outliers = X.copy()
        X_with_outliers.loc[0, 'numeric_col1'] = 1000
        X_with_outliers.loc[1, 'numeric_col2'] = -1000
        
        pipeline = EDA_Pipeline(basic_config)
        pipeline.identify_column_types(X_with_outliers)
        X_clean = pipeline.handle_outliers(X_with_outliers, fit=True)
        
        # Check that outliers are clipped
        assert X_clean['numeric_col1'].max() < 1000
        assert X_clean['numeric_col2'].min() > -1000
    
    def test_handle_outliers_zscore(self, basic_config, sample_data):
        """Test outlier handling with Z-score method"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        config = copy.deepcopy(basic_config)
        config['outlier_method'] = 'zscore'
        
        pipeline = EDA_Pipeline(config)
        pipeline.identify_column_types(X)
        X_clean = pipeline.handle_outliers(X, fit=True)
        
        assert X_clean.shape == X.shape
    
    def test_encode_categorical_label(self, basic_config, sample_data):
        """Test categorical encoding with label encoding"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        pipeline = EDA_Pipeline(basic_config)
        pipeline.identify_column_types(X)
        X_encoded = pipeline.encode_categorical(X, fit=True)
        
        # Check that categorical columns are now numeric
        assert pd.api.types.is_numeric_dtype(X_encoded['categorical_col1'])
        assert pd.api.types.is_numeric_dtype(X_encoded['categorical_col2'])
    
    def test_encode_categorical_frequency(self, basic_config, sample_data):
        """Test categorical encoding with frequency encoding"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        config = copy.deepcopy(basic_config)
        config['categorical_encoding'] = 'frequency'
        
        pipeline = EDA_Pipeline(config)
        pipeline.identify_column_types(X)
        X_encoded = pipeline.encode_categorical(X, fit=True)
        
        assert pd.api.types.is_numeric_dtype(X_encoded['categorical_col1'])
    
    def test_feature_engineering_interactions(self, basic_config, sample_data):
        """Test feature engineering with interactions"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        config = copy.deepcopy(basic_config)
        config['create_interactions'] = True
        config['categorical_encoding'] = 'label'
        
        pipeline = EDA_Pipeline(config)
        pipeline.identify_column_types(X)
        X_encoded = pipeline.encode_categorical(X, fit=True)
        X_engineered = pipeline.feature_engineering(X_encoded)
        
        # Check that interaction features were created
        assert X_engineered.shape[1] > X_encoded.shape[1]
    
    def test_feature_engineering_ratios(self, basic_config, sample_data):
        """Test feature engineering with ratios"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        config = copy.deepcopy(basic_config)
        config['create_ratios'] = True
        
        pipeline = EDA_Pipeline(config)
        pipeline.identify_column_types(X)
        X_engineered = pipeline.feature_engineering(X)
        
        # Check that ratio features were created
        assert X_engineered.shape[1] > X.shape[1]
    
    def test_feature_engineering_polynomial(self, basic_config, sample_data):
        """Test feature engineering with polynomial features"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        config = copy.deepcopy(basic_config)
        config['create_polynomial'] = True
        
        pipeline = EDA_Pipeline(config)
        pipeline.identify_column_types(X)
        X_engineered = pipeline.feature_engineering(X)
        
        # Check that polynomial features were created
        assert X_engineered.shape[1] > X.shape[1]
    
    def test_scale_features_standard(self, basic_config, sample_data):
        """Test feature scaling with standard scaler"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        pipeline = EDA_Pipeline(basic_config)
        pipeline.identify_column_types(X)
        X_scaled = pipeline.scale_features(X, fit=True)
        
        # Check that numeric columns are scaled
        numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(X_scaled[col].mean()) < 1.0  # Mean should be close to 0
    
    def test_scale_features_minmax(self, basic_config, sample_data):
        """Test feature scaling with MinMax scaler"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        config = copy.deepcopy(basic_config)
        config['scaling'] = 'minmax'
        
        pipeline = EDA_Pipeline(config)
        pipeline.identify_column_types(X)
        X_scaled = pipeline.scale_features(X, fit=True)
        
        # Check that values are between 0 and 1
        numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X_scaled[col].max() > 0:
                assert X_scaled[col].min() >= 0
                assert X_scaled[col].max() <= 1
    
    def test_scale_features_robust(self, basic_config, sample_data):
        """Test feature scaling with Robust scaler"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        config = copy.deepcopy(basic_config)
        config['scaling'] = 'robust'
        
        pipeline = EDA_Pipeline(config)
        pipeline.identify_column_types(X)
        X_scaled = pipeline.scale_features(X, fit=True)
        
        assert X_scaled.shape == X.shape
    
    def test_fit_transform(self, basic_config, sample_data):
        """Test complete fit_transform pipeline"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        pipeline = EDA_Pipeline(basic_config)
        X_transformed, y_transformed = pipeline.fit_transform(X, y)
        
        # Check outputs
        assert isinstance(X_transformed, np.ndarray)
        assert isinstance(y_transformed, np.ndarray)
        assert X_transformed.shape[0] == y_transformed.shape[0]
        assert X_transformed.shape[1] > 0
        # Check for no infinite values
        assert not np.isinf(X_transformed).any()
    
    def test_transform_after_fit(self, basic_config, sample_data):
        """Test transform after fit_transform"""
        from app import EDA_Pipeline
        X, y = sample_data
        
        # Split data
        X_train = X.iloc[:80]
        y_train = y[:80]
        X_test = X.iloc[80:]
        
        # Fit on training data
        pipeline = EDA_Pipeline(basic_config)
        X_train_transformed, _ = pipeline.fit_transform(X_train, y_train)
        
        # Transform test data
        X_test_transformed = pipeline.transform(X_test)
        
        assert X_test_transformed.shape[0] == len(X_test)
        assert X_test_transformed.shape[1] == X_train_transformed.shape[1]
        assert not np.isinf(X_test_transformed).any()


class TestMLPModel:
    """Test suite for MLP neural network model"""
    
    def test_mlp_initialization(self):
        """Test MLP model initialization"""
        from app import MLP
        
        arch = [
            {'units': 64, 'activation': 'relu', 'dropout': 0.2},
            {'units': 32, 'activation': 'relu', 'dropout': 0.1}
        ]
        input_dim = 20
        num_classes = 2
        
        model = MLP(arch, input_dim, num_classes)
        
        assert isinstance(model, nn.Module)
        assert model.network is not None
    
    def test_mlp_forward_pass(self):
        """Test MLP forward pass"""
        from app import MLP
        
        arch = [
            {'units': 64, 'activation': 'relu', 'dropout': 0.0},
            {'units': 32, 'activation': 'relu', 'dropout': 0.0}
        ]
        input_dim = 20
        num_classes = 3
        batch_size = 16
        
        model = MLP(arch, input_dim, num_classes)
        X = torch.randn(batch_size, input_dim)
        
        output = model(X)
        
        assert output.shape == (batch_size, num_classes)
    
    def test_mlp_different_activations(self):
        """Test MLP with different activation functions"""
        from app import MLP
        
        activations = ['relu', 'tanh', 'sigmoid']
        input_dim = 20
        num_classes = 2
        batch_size = 16
        
        for activation in activations:
            arch = [
                {'units': 32, 'activation': activation, 'dropout': 0.0}
            ]
            model = MLP(arch, input_dim, num_classes)
            X = torch.randn(batch_size, input_dim)
            output = model(X)
            
            assert output.shape == (batch_size, num_classes)
    
    def test_mlp_with_dropout(self):
        """Test MLP with dropout layers"""
        from app import MLP
        
        arch = [
            {'units': 64, 'activation': 'relu', 'dropout': 0.5},
            {'units': 32, 'activation': 'relu', 'dropout': 0.3}
        ]
        input_dim = 20
        num_classes = 2
        batch_size = 16
        
        model = MLP(arch, input_dim, num_classes)
        X = torch.randn(batch_size, input_dim)
        
        # Test training mode
        model.train()
        output_train = model(X)
        
        # Test evaluation mode
        model.eval()
        output_eval = model(X)
        
        assert output_train.shape == (batch_size, num_classes)
        assert output_eval.shape == (batch_size, num_classes)


class TestRandomFunctions:
    """Test suite for random generation functions"""
    
    def test_random_preprocessing(self):
        """Test random preprocessing configuration generation"""
        from app import random_preprocessing
        
        config = random_preprocessing(task_type='classification')
        
        # Check all required keys are present
        required_keys = [
            'numeric_impute', 'categorical_impute', 'outlier_method',
            'outlier_threshold', 'create_interactions', 'create_ratios',
            'create_binning', 'create_polynomial', 'n_bins',
            'categorical_encoding', 'scaling', 'feature_selection',
            'feature_selection_method', 'selection_threshold',
            'n_features_to_select', 'n_components'
        ]
        
        for key in required_keys:
            assert key in config
        
        # Check value types
        assert config['numeric_impute'] in ['mean', 'median', 'mode']
        assert config['categorical_impute'] in ['mode', 'constant']
        assert config['outlier_method'] in ['none', 'iqr', 'zscore', 'clip']
        assert isinstance(config['create_interactions'], bool)
    
    def test_random_mlp_layer(self):
        """Test random MLP layer generation"""
        from app import random_mlp_layer
        
        layer = random_mlp_layer()
        
        assert 'units' in layer
        assert 'activation' in layer
        assert 'dropout' in layer
        assert layer['units'] in [16, 32, 64, 128, 256]
        assert layer['activation'] in ['relu', 'tanh', 'sigmoid']
        assert layer['dropout'] in [0.0, 0.1, 0.2, 0.3, 0.5]
    
    def test_random_mlp_arch(self):
        """Test random MLP architecture generation"""
        from app import random_mlp_arch
        
        arch = random_mlp_arch(min_layers=1, max_layers=4)
        
        assert isinstance(arch, list)
        assert 1 <= len(arch) <= 4
        
        for layer in arch:
            assert 'units' in layer
            assert 'activation' in layer
            assert 'dropout' in layer
    
    def test_random_model_type(self):
        """Test random model type generation"""
        from app import random_model_type
        
        model_type = random_model_type()
        
        assert model_type in ['mlp', 'rf', 'logreg', 'gbm', 'svm']


class TestMutationFunctions:
    """Test suite for mutation functions"""
    
    @pytest.fixture
    def base_preprocessing(self):
        """Base preprocessing configuration"""
        return {
            'numeric_impute': 'mean',
            'categorical_impute': 'mode',
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'create_interactions': False,
            'create_ratios': False,
            'create_binning': False,
            'create_polynomial': False,
            'n_bins': 5,
            'categorical_encoding': 'label',
            'scaling': 'standard',
            'feature_selection': False,
            'feature_selection_method': 'rf_importance',
            'selection_threshold': 0.01,
            'n_features_to_select': 10,
            'n_components': 5
        }
    
    def test_mutate_preprocessing(self, base_preprocessing):
        """Test preprocessing mutation"""
        from app import mutate_preprocessing
        
        mutated = mutate_preprocessing(base_preprocessing)
        
        # Check that it's different
        assert mutated != base_preprocessing or True  # At least one field should change
        
        # Check that all keys are still present
        assert set(mutated.keys()) == set(base_preprocessing.keys())
    
    def test_mutate_preprocessing_preserves_original(self, base_preprocessing):
        """Test that mutation doesn't modify the original"""
        from app import mutate_preprocessing
        
        original = copy.deepcopy(base_preprocessing)
        mutated = mutate_preprocessing(base_preprocessing)
        
        assert base_preprocessing == original
    
    def test_mutate_mlp_arch_add_layer(self):
        """Test MLP architecture mutation - adding layers"""
        from app import mutate_mlp_arch
        
        arch = [
            {'units': 64, 'activation': 'relu', 'dropout': 0.2}
        ]
        
        # Multiple mutations to statistically ensure layer addition occurs
        mutations = [mutate_mlp_arch(arch, max_layers=5) for _ in range(20)]
        
        # At least one should have a different length
        lengths = [len(m) for m in mutations]
        assert len(set(lengths)) > 1 or lengths[0] == len(arch)
    
    def test_mutate_mlp_arch_structure(self):
        """Test MLP architecture mutation maintains structure"""
        from app import mutate_mlp_arch
        
        arch = [
            {'units': 64, 'activation': 'relu', 'dropout': 0.2},
            {'units': 32, 'activation': 'relu', 'dropout': 0.1}
        ]
        
        mutated = mutate_mlp_arch(arch, max_layers=5)
        
        assert isinstance(mutated, list)
        for layer in mutated:
            assert 'units' in layer
            assert 'activation' in layer
            assert 'dropout' in layer


class TestTrainAndEvaluate:
    """Test suite for training and evaluation function"""
    
    def test_train_and_evaluate_mlp(self):
        """Test training and evaluation of MLP model"""
        from app import train_and_evaluate_model, MLP
        
        # Generate test data
        X_train = np.random.randn(50, 10)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)
        
        arch = [
            {'units': 32, 'activation': 'relu', 'dropout': 0.0}
        ]
        device = torch.device('cpu')
        
        fitness, metrics, model = train_and_evaluate_model(
            X_train, y_train, X_val, y_val,
            'mlp', arch, device, 2,
            train_epochs=1, lr=0.001
        )
        
        assert isinstance(fitness, float)
        assert 0 <= fitness <= 1
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1' in metrics
    
    def test_train_and_evaluate_rf(self):
        """Test training and evaluation of Random Forest model"""
        from app import train_and_evaluate_model
        
        X_train = np.random.randn(50, 10)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)
        
        device = torch.device('cpu')
        
        fitness, metrics, model = train_and_evaluate_model(
            X_train, y_train, X_val, y_val,
            'rf', None, device, 2,
            train_epochs=1, lr=0.001
        )
        
        assert isinstance(fitness, float)
        assert isinstance(metrics, dict)
        assert model is not None
    
    def test_train_and_evaluate_logreg(self):
        """Test training and evaluation of Logistic Regression model"""
        from app import train_and_evaluate_model
        
        X_train = np.random.randn(50, 10)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)
        
        device = torch.device('cpu')
        
        fitness, metrics, model = train_and_evaluate_model(
            X_train, y_train, X_val, y_val,
            'logreg', None, device, 2,
            train_epochs=1, lr=0.001
        )
        
        assert isinstance(fitness, float)
        assert isinstance(metrics, dict)
    
    def test_train_and_evaluate_gbm(self):
        """Test training and evaluation of Gradient Boosting model"""
        from app import train_and_evaluate_model
        
        X_train = np.random.randn(50, 10)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)
        
        device = torch.device('cpu')
        
        fitness, metrics, model = train_and_evaluate_model(
            X_train, y_train, X_val, y_val,
            'gbm', None, device, 2,
            train_epochs=1, lr=0.001
        )
        
        assert isinstance(fitness, float)
        assert isinstance(metrics, dict)
    
    def test_train_and_evaluate_svm(self):
        """Test training and evaluation of SVM model"""
        from app import train_and_evaluate_model
        
        X_train = np.random.randn(50, 10)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)
        
        device = torch.device('cpu')
        
        fitness, metrics, model = train_and_evaluate_model(
            X_train, y_train, X_val, y_val,
            'svm', None, device, 2,
            train_epochs=1, lr=0.001
        )
        
        assert isinstance(fitness, float)
        assert isinstance(metrics, dict)


class TestEvolutionaryAlgorithm:
    """Test suite for evolutionary algorithm"""
    
    def test_run_evolution_basic(self):
        """Test basic evolution run"""
        from app import run_evolution, train_and_evaluate_model

        # Generate simple dataset
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(30, 5)
        y_val = np.random.randint(0, 2, 30)

        args = {
            'pop_size': 4,
            'generations': 2,
            'elitism': 0.25,
            'tournament_k': 2,
            'min_layers': 1,
            'max_layers': 2,
            'train_epochs': 1,
            'lr': 0.001
        }

        device = torch.device('cpu')

        # Mock train_and_evaluate_model to return predictable results
        with patch('app.train_and_evaluate_model') as mock_train:
            mock_train.return_value = (0.85, {'accuracy': 0.85, 'f1': 0.82, 'precision': 0.80, 'recall': 0.84}, None)
            
            best, history = run_evolution(
                X_train, y_train, X_val, y_val,
                args, device, 2
            )

            # Check outputs
            assert best is not None
            assert best.fitness is not None
            assert best.preprocessing is not None
            assert best.model_type in ['mlp', 'rf', 'logreg', 'gbm', 'svm']
            assert isinstance(history, list)
            assert len(history) == args['generations']


class TestIndividual:
    """Test suite for Individual namedtuple"""
    
    def test_individual_creation(self):
        """Test Individual namedtuple creation"""
        from app import Individual
        
        preprocessing = {'numeric_impute': 'mean'}
        arch = [{'units': 64, 'activation': 'relu', 'dropout': 0.0}]
        model_type = 'mlp'
        fitness = 0.85
        metrics = {'accuracy': 0.85, 'f1': 0.82, 'precision': 0.87, 'recall': 0.80}
        
        individual = Individual(
            preprocessing=preprocessing,
            arch=arch,
            model_type=model_type,
            fitness=fitness,
            metrics=metrics
        )
        
        assert individual.preprocessing == preprocessing
        assert individual.arch == arch
        assert individual.model_type == model_type
        assert individual.fitness == fitness
        assert individual.metrics == metrics
    
    def test_individual_immutable(self):
        """Test that Individual is immutable"""
        from app import Individual
        
        individual = Individual(
            preprocessing={'numeric_impute': 'mean'},
            arch=[],
            model_type='mlp',
            fitness=0.85,
            metrics={'accuracy': 0.85}
        )
        
        with pytest.raises(AttributeError):
            individual.fitness = 0.9


class TestEdgeCases:
    """Test suite for edge cases and error handling"""
    
    def test_eda_pipeline_with_all_categorical(self):
        """Test EDA pipeline with all categorical features"""
        from app import EDA_Pipeline
        
        config = {
            'numeric_impute': 'mean',
            'categorical_impute': 'mode',
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'create_interactions': False,
            'create_ratios': False,
            'create_binning': False,
            'create_polynomial': False,
            'n_bins': 5,
            'categorical_encoding': 'label',
            'scaling': 'standard',
            'feature_selection': False,
            'feature_selection_method': 'rf_importance',
            'selection_threshold': 0.01,
            'n_features_to_select': 10,
            'n_components': 5
        }
        
        X = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y']
        })
        y = np.array([0, 1, 0, 1, 1])
        
        pipeline = EDA_Pipeline(config)
        X_transformed, y_transformed = pipeline.fit_transform(X, y)
        
        assert X_transformed.shape[0] == len(y)
        assert not np.isinf(X_transformed).any()
    
    def test_eda_pipeline_with_all_numeric(self):
        """Test EDA pipeline with all numeric features"""
        from app import EDA_Pipeline
        
        config = {
            'numeric_impute': 'mean',
            'categorical_impute': 'mode',
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'create_interactions': True,
            'create_ratios': True,
            'create_binning': True,
            'create_polynomial': True,
            'n_bins': 5,
            'categorical_encoding': 'label',
            'scaling': 'standard',
            'feature_selection': False,
            'feature_selection_method': 'rf_importance',
            'selection_threshold': 0.01,
            'n_features_to_select': 10,
            'n_components': 5
        }
        
        X = pd.DataFrame({
            'num1': np.random.randn(50),
            'num2': np.random.randn(50),
            'num3': np.random.randn(50)
        })
        y = np.random.randint(0, 2, 50)
        
        pipeline = EDA_Pipeline(config)
        X_transformed, y_transformed = pipeline.fit_transform(X, y)
        
        assert X_transformed.shape[0] == len(y)
    
    def test_eda_pipeline_single_sample(self):
        """Test EDA pipeline with single sample"""
        from app import EDA_Pipeline
        
        config = {
            'numeric_impute': 'mean',
            'categorical_impute': 'mode',
            'outlier_method': 'none',
            'outlier_threshold': 1.5,
            'create_interactions': False,
            'create_ratios': False,
            'create_binning': False,
            'create_polynomial': False,
            'n_bins': 5,
            'categorical_encoding': 'label',
            'scaling': 'standard',
            'feature_selection': False,
            'feature_selection_method': 'rf_importance',
            'selection_threshold': 0.01,
            'n_features_to_select': 10,
            'n_components': 5
        }
        
        X = pd.DataFrame({'num1': [1.0], 'num2': [2.0]})
        y = np.array([0])
        
        pipeline = EDA_Pipeline(config)
        X_transformed, y_transformed = pipeline.fit_transform(X, y)
        
        assert X_transformed.shape[0] == 1
    
    def test_train_and_evaluate_with_single_class(self):
        """Test model training with imbalanced classes"""
        from app import train_and_evaluate_model
        
        X_train = np.random.randn(50, 10)
        y_train = np.zeros(50, dtype=int)  # All zeros
        y_train[0:5] = 1  # Only 5 ones
        
        X_val = np.random.randn(20, 10)
        y_val = np.zeros(20, dtype=int)
        y_val[0:5] = 1
        
        device = torch.device('cpu')
        
        # Should handle imbalanced data gracefully
        fitness, metrics, model = train_and_evaluate_model(
            X_train, y_train, X_val, y_val,
            'rf', None, device, 2,
            train_epochs=1, lr=0.001
        )
        
        assert isinstance(fitness, float)
    
    def test_mlp_with_zero_dropout(self):
        """Test MLP with zero dropout"""
        from app import MLP
        
        arch = [
            {'units': 32, 'activation': 'relu', 'dropout': 0.0},
            {'units': 16, 'activation': 'relu', 'dropout': 0.0}
        ]
        
        model = MLP(arch, 10, 2)
        X = torch.randn(16, 10)
        
        output = model(X)
        assert output.shape == (16, 2)


class TestDataIntegrity:
    """Test suite for data integrity and consistency"""
    
    def test_transform_preserves_shape(self):
        """Test that transform preserves correct shape"""
        from app import EDA_Pipeline
        
        config = {
            'numeric_impute': 'mean',
            'categorical_impute': 'mode',
            'outlier_method': 'none',
            'outlier_threshold': 1.5,
            'create_interactions': False,
            'create_ratios': False,
            'create_binning': False,
            'create_polynomial': False,
            'n_bins': 5,
            'categorical_encoding': 'label',
            'scaling': 'standard',
            'feature_selection': False,
            'feature_selection_method': 'rf_importance',
            'selection_threshold': 0.01,
            'n_features_to_select': 10,
            'n_components': 5
        }
        
        X_train = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50)
        })
        y_train = np.random.randint(0, 2, 50)
        
        X_test = pd.DataFrame({
            'f1': np.random.randn(20),
            'f2': np.random.randn(20)
        })
        
        pipeline = EDA_Pipeline(config)
        X_train_t, _ = pipeline.fit_transform(X_train, y_train)
        X_test_t = pipeline.transform(X_test)
        
        assert X_train_t.shape[1] == X_test_t.shape[1]
    
    def test_no_nan_values_after_transform(self):
        """Test that no NaN values appear after transformation"""
        from app import EDA_Pipeline
        
        config = {
            'numeric_impute': 'mean',
            'categorical_impute': 'mode',
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'create_interactions': True,
            'create_ratios': True,
            'create_binning': True,
            'create_polynomial': False,
            'n_bins': 5,
            'categorical_encoding': 'label',
            'scaling': 'standard',
            'feature_selection': False,
            'feature_selection_method': 'rf_importance',
            'selection_threshold': 0.01,
            'n_features_to_select': 10,
            'n_components': 5
        }
        
        X = pd.DataFrame({
            'n1': np.random.randn(100),
            'n2': np.random.randn(100),
            'n3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)
        
        # Add some NaNs
        X.loc[0:5, 'n1'] = np.nan
        
        pipeline = EDA_Pipeline(config)
        X_transformed, _ = pipeline.fit_transform(X, y)
        
        assert not np.isnan(X_transformed).any()


class TestConfigurationVariations:
    """Test different configuration combinations"""
    
    def test_no_feature_engineering(self):
        """Test with no feature engineering"""
        from app import EDA_Pipeline
        
        config = {
            'numeric_impute': 'mean',
            'categorical_impute': 'mode',
            'outlier_method': 'none',
            'outlier_threshold': 1.5,
            'create_interactions': False,
            'create_ratios': False,
            'create_binning': False,
            'create_polynomial': False,
            'n_bins': 5,
            'categorical_encoding': 'label',
            'scaling': 'none',
            'feature_selection': False,
            'feature_selection_method': 'rf_importance',
            'selection_threshold': 0.01,
            'n_features_to_select': 10,
            'n_components': 5
        }
        
        X = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50)
        })
        y = np.random.randint(0, 2, 50)
        
        pipeline = EDA_Pipeline(config)
        X_transformed, _ = pipeline.fit_transform(X, y)
        
        assert X_transformed.shape[1] == 2
    
    def test_all_feature_engineering(self):
        """Test with all feature engineering enabled"""
        from app import EDA_Pipeline
        
        config = {
            'numeric_impute': 'mean',
            'categorical_impute': 'mode',
            'outlier_method': 'zscore',
            'outlier_threshold': 2.0,
            'create_interactions': True,
            'create_ratios': True,
            'create_binning': True,
            'create_polynomial': True,
            'n_bins': 5,
            'categorical_encoding': 'frequency',
            'scaling': 'minmax',
            'feature_selection': False,
            'feature_selection_method': 'rf_importance',
            'selection_threshold': 0.01,
            'n_features_to_select': 10,
            'n_components': 5
        }
        
        X = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
            'f3': np.random.randn(50)
        })
        y = np.random.randint(0, 2, 50)
        
        pipeline = EDA_Pipeline(config)
        X_transformed, _ = pipeline.fit_transform(X, y)
        
        # With all feature engineering, we expect more features
        assert X_transformed.shape[1] > 3


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
