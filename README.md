# 🧬 EvoNAS: Evolutionary Neural Architecture Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

Automated ML pipeline optimization using evolutionary algorithms. Simultaneously searches preprocessing strategies, feature engineering, and model architectures.

---

## 🎯 What It Does

EvoNAS automates the entire ML pipeline design process:
- **Data Preprocessing**: Imputation, outlier handling, scaling
- **Feature Engineering**: Interactions, ratios, polynomial features, binning
- **Feature Selection**: RF importance, SelectKBest, PCA
- **Architecture Search**: Neural network topology (depth, width, activations, dropout)
- **Model Selection**: MLP, Random Forest, Gradient Boosting, Logistic Regression, SVM

**Key Advantage**: Co-evolves preprocessing and architecture together, recognizing their interdependence.

---

## 🧠 Neural Architecture Search

### The Problem

Designing neural networks involves vast combinatorial spaces:
- 5 layers × 5 unit choices × 3 activations × 5 dropout rates = **759M configurations**
- Add preprocessing options → **billions of possible pipelines**

Traditional approaches:
- **RL-based NAS**: Requires thousands of GPUs (Google's approach)
- **Gradient-based NAS**: Limited to differentiable operations
- **Random Search**: No learning from past evaluations

### Our Solution: Evolutionary Algorithms

Evolutionary NAS uses genetic algorithms inspired by natural selection:

```
Population → Fitness Evaluation → Selection → Mutation → Next Generation
```

**Why Evolution Works Here**:
1. **Handles Discrete Spaces**: "Add layer" or "remove feature" aren't differentiable
2. **No Gradient Required**: Works with any fitness function
3. **Population Diversity**: Explores multiple solutions simultaneously
4. **Interpretable Results**: Human-readable configurations

---

## 🔬 Algorithm Mechanics

### 1. Individual Representation

Each candidate solution is a complete pipeline:

```python
{
  'preprocessing': {
    'numeric_impute': 'median',
    'outlier_method': 'iqr',
    'scaling': 'standard',
    'create_interactions': True,
    'feature_selection': True,
    'feature_selection_method': 'rf_importance'
  },
  'architecture': [
    {'units': 64, 'activation': 'relu', 'dropout': 0.2},
    {'units': 32, 'activation': 'relu', 'dropout': 0.1}
  ],
  'model_type': 'mlp'
}
```

### 2. Fitness Function

```
Fitness = 0.7 × Accuracy + 0.3 × F1-Score
```

Balances overall correctness with class-balanced performance. Each individual trains for N epochs (typically 3-5) and evaluates on validation set.

### 3. Tournament Selection

For each offspring:
1. Randomly sample k individuals (k=2-5)
2. Select highest fitness as parent
3. Apply mutation to create child

Balances selection pressure with diversity maintenance.

### 4. Mutation Operators

**Preprocessing Mutations** (probability 0.33):
- Switch imputation: mean → median → mode
- Change outlier method: none → IQR → Z-score → clip
- Toggle features: interactions, ratios, binning, polynomial
- Modify scaling: standard → minmax → robust → none
- Adjust feature selection: method and thresholds

**Architecture Mutations** (probability 0.33, MLP only):
- **Add Layer**: Insert at random position (respects max_layers)
- **Remove Layer**: Delete random layer (maintains min_layers)
- **Modify Layer**: Change units (16/32/64/128/256), activation, or dropout

**Model Type Mutation** (probability 0.33):
- Switch between MLP, RF, Logistic Regression, GBM, SVM

### 5. Elitism

Top `elitism_rate × population_size` individuals automatically advance to next generation. Ensures best solution never degrades.

### 6. Search Space Summary

| Component | Options | Size |
|-----------|---------|------|
| **Preprocessing** | Imputation (3×2), Outliers (4×3), Scaling (4), Encoding (2) | ~288 |
| **Feature Engineering** | Interactions, Ratios, Binning, Polynomial | 16 |
| **Feature Selection** | Methods (3), Thresholds (3), Components (3) | ~27 |
| **MLP Architecture** | Layers (1-8), Units (5), Activations (3), Dropout (5) | 10^9+ |
| **Model Type** | 5 options | 5 |
| **Total Search Space** | | >10^12 |

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Requirements**: Python 3.8+, 4GB RAM (8GB recommended), GPU optional

---

## 📖 Usage

### Quick Start

1. **Upload CSV**: Drag-and-drop your dataset
2. **Remove Columns** (optional): Exclude IDs, timestamps, irrelevant features
3. **Select Target**: Choose prediction column from dropdown
4. **Configure Split**: Set validation (15%) and test (20%) sizes
5. **Set Parameters**:
   - Population: 8-12 (quick), 15-20 (thorough)
   - Generations: 5-8 (quick), 10-15 (thorough)
   - Epochs: 3-5 per evaluation
6. **Run Evolution**: Monitor real-time progress
7. **Export**: Download JSON config and evolution history CSV

### Configuration Guidelines

| Dataset Size | Population | Generations | Epochs | Expected Time |
|--------------|------------|-------------|--------|---------------|
| <5K samples | 6-8 | 3-5 | 2-3 | 5-15 min |
| 5K-50K samples | 10-12 | 5-8 | 3-5 | 15-45 min |
| >50K samples | 15-20 | 8-12 | 3-5 | 45-120 min |

**Elitism**: 0.25 (default) - preserves top 25% each generation  
**Tournament Size**: 3 (default) - moderate selection pressure

### Understanding Results

**Evolution Patterns**:
- **Healthy**: Gen 1: 0.75 → Gen 5: 0.87 (gradual improvement)
- **Fast Convergence**: Gen 1: 0.88 → Gen 2+: 0.88 (early optimum found)
- **Noisy**: Fluctuating fitness (increase epochs or population)

**Model Selection Insights**:
- **MLP chosen** → Complex patterns, large dataset, non-linear
- **Random Forest** → Feature interactions critical, mixed data types
- **Gradient Boosting** → Best overall performance achievable
- **Logistic Regression** → Linear separability, need interpretability
- **SVM** → Clear class margins, medium-sized dataset

**Preprocessing Insights**:
- **StandardScaler** → Gaussian-distributed features
- **MinMaxScaler** → Bounded ranges important
- **IQR outlier removal** → Extreme values present
- **Interactions enabled** → Multiplicative relationships exist
- **Aggressive selection (threshold=0.05)** → High dimensionality, many weak features

---

## 🔧 Advanced Topics

### Imbalanced Datasets

Monitor F1 > Accuracy. Adjust fitness function:
```python
Fitness = 0.3 × Accuracy + 0.7 × F1  # Prioritize F1
```

### High-Dimensional Data (>100 features)

- Enable feature selection (critical)
- Use PCA if features highly correlated
- Increase population size (15-20) to explore selection space
- Consider longer training for selection models

### Small Datasets (<1000 samples)

- Lower epochs (1-2) to prevent overfitting
- Higher dropout (0.3-0.5) for regularization
- Prefer simpler models (Logistic Regression, shallow MLPs)
- Increase validation size to 20-25%

### GPU Acceleration

Automatically uses CUDA if available. Provides 2-10× speedup for MLP training. Verify:
```python
import torch
print(torch.cuda.is_available())
```

### Performance Optimization

**Time Complexity**: `O(population_size × generations × (preprocessing + training + evaluation))`

**Speed Tips**:
1. Start with small population (6) and generations (3) for prototyping
2. Reduce epochs to 1-2 for initial experiments
3. Use GPU for MLP-heavy populations
4. Remove unnecessary columns before upload

---

## 📊 Example: Customer Churn

```
Dataset: 10,000 customers × 25 features
Target: churned (0: retained, 1: churned) - 75%/25% split
Action: Removed customer_id, signup_date

Configuration:
- Population: 12
- Generations: 8  
- Epochs: 5
- Runtime: 28 minutes (GPU)

Best Solution:
- Model: Gradient Boosting
- Preprocessing: StandardScaler, IQR outliers, interaction features
- Feature Selection: RF importance (threshold=0.05), 12 features kept
- Performance: Val F1=0.72, Test F1=0.70, Test Acc=85%
```

---

## 🔬 Technical Details

### Fitness Evaluation Pipeline

```python
1. Apply preprocessing config to training data
2. Transform validation data using fitted pipeline
3. Train model for N epochs
4. Predict on validation set
5. Compute: acc, f1, precision, recall
6. Return: fitness = 0.7*acc + 0.3*f1
```

### Mutation Rate Impact

- **Low rate** (single change per mutation) → Fine-grained search, slower convergence
- **High rate** (multiple changes) → Broad exploration, risk of destroying good solutions

Current implementation: Single-component mutation (optimal for this search space)

### Convergence Criteria

Fixed generation count (no early stopping). Monitor stagnation:
- If best fitness unchanged for 3+ generations → Consider increasing population or mutation diversity
- If still improving at final generation → Run longer

---

## 🚀 Future Enhancements

- **Multi-objective optimization**: Pareto fronts for accuracy vs. complexity/speed
- **Parallel evaluation**: Distribute individuals across workers
- **Warm starting**: Initialize with domain-specific architectures
- **Adaptive mutation**: Adjust rates based on population diversity
- **Regression support**: Extend beyond classification
- **Time series**: Add LSTM/GRU architecture search

---

## 📚 References

1. Real, E., et al. (2019). "Regularized Evolution for Image Classifier Architecture Search". ICML.
2. Zoph, B., & Le, Q. V. (2017). "Neural Architecture Search with Reinforcement Learning". ICLR.
3. Elsken, T., et al. (2019). "Neural Architecture Search: A Survey". JMLR.
