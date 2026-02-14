# Credit Card Fraud Detection - Complete Analysis in One Notebook
# This notebook contains all the code in a single file for easy execution in Google Colab

# PART 1: INSTALL REQUIRED PACKAGES

# Run this cell first to install all required packages
import subprocess
import sys

packages = [
    'imbalanced-learn',
    'scikit-learn',
    'pandas',
    'numpy', 
    'matplotlib',
    'seaborn'
]

print("Installing required packages...")
for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
print("✓ All packages installed successfully!\n")


# PART 2: IMPORT LIBRARIES


import pandas as pd
import numpy as np
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("✓ All libraries imported successfully!\n")


# PART 3: DATA PREPROCESSING CLASS


class DataPreprocessor:
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_and_split(self, file_path):
        df = pd.read_csv(file_path)
        
        if df.isnull().sum().sum() > 0:
            print("Warning: Missing values found. Filling with mean...")
            df = df.fillna(df.mean())
        
        X = df.drop('Class', axis=1)
        y = df['Class'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


# PART 4: SAMPLING TECHNIQUES CLASS


class SamplingTechniques:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def apply_sampling(self, X, y, method):
        if method == 'RandomOverSampling':
            return self._random_oversampling(X, y)
        elif method == 'SMOTE':
            return self._smote_sampling(X, y)
        elif method == 'ADASYN':
            return self._adasyn_sampling(X, y)
        elif method == 'RandomUnderSampling':
            return self._random_undersampling(X, y)
        elif method == 'TomekLinks':
            return self._tomek_links(X, y)
    
    def _random_oversampling(self, X, y):
        ros = RandomOverSampler(random_state=self.random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def _smote_sampling(self, X, y):
        min_samples = min(Counter(y).values())
        k_neighbors = min(5, min_samples - 1)
        smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def _adasyn_sampling(self, X, y):
        min_samples = min(Counter(y).values())
        k_neighbors = min(5, min_samples - 1)
        try:
            adasyn = ADASYN(random_state=self.random_state, n_neighbors=k_neighbors)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
        except:
            print("    (ADASYN failed, using SMOTE as fallback)")
            smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def _random_undersampling(self, X, y):
        rus = RandomUnderSampler(random_state=self.random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def _tomek_links(self, X, y):
        tomek = TomekLinks()
        X_resampled, y_resampled = tomek.fit_resample(X, y)
        class_counts = Counter(y_resampled)
        if max(class_counts.values()) / min(class_counts.values()) > 2:
            ros = RandomOverSampler(random_state=self.random_state)
            X_resampled, y_resampled = ros.fit_resample(X_resampled, y_resampled)
        return X_resampled, y_resampled


# PART 5: MODEL TRAINING CLASS


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def train_model(self, X_train, y_train, model_name):
        if model_name == 'LogisticRegression':
            model = LogisticRegression(random_state=self.random_state, max_iter=1000, solver='liblinear')
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, 
                                          max_depth=10, min_samples_split=5)
        elif model_name == 'GradientBoosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state,
                                              learning_rate=0.1, max_depth=5)
        elif model_name == 'SVM':
            model = SVC(kernel='rbf', random_state=self.random_state, C=1.0, gamma='scale')
        elif model_name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100
        return accuracy


# PART 6: EVALUATION CLASS


class ModelEvaluator:
    def create_results_table(self, all_results):
        results_dict = {'Model': ['M1', 'M2', 'M3', 'M4', 'M5']}
        
        for sampling_idx in range(1, 6):
            sampling_name = f'Sampling{sampling_idx}'
            accuracies = []
            for model_idx in range(1, 6):
                model_name = f'M{model_idx}'
                accuracy = all_results[sampling_name][model_name]['accuracy']
                accuracies.append(accuracy)
            results_dict[sampling_name] = accuracies
        
        df = pd.DataFrame(results_dict)
        df = df.set_index('Model')
        return df
    
    def find_best_combinations(self, all_results):
        combinations = []
        for sampling_name, models in all_results.items():
            for model_name, result in models.items():
                combinations.append({
                    'sampling': sampling_name,
                    'model': model_name,
                    'model_name': result['model_name'],
                    'accuracy': result['accuracy']
                })
        combinations.sort(key=lambda x: x['accuracy'], reverse=True)
        return combinations


# PART 7: VISUALIZATION CLASS


class ResultsVisualizer:
    def create_heatmap(self, results_df):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(results_df, annot=True, fmt='.2f', cmap='YlGnBu', 
                    cbar_kws={'label': 'Accuracy (%)'}, ax=ax,
                    linewidths=0.5, linecolor='gray')
        plt.title('Model Accuracy Across Different Sampling Techniques', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Sampling Technique', fontsize=12, fontweight='bold')
        plt.ylabel('Model', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_plots(self, all_results):
        sampling_names = [f'Sampling{i}' for i in range(1, 6)]
        model_names = ['M1', 'M2', 'M3', 'M4', 'M5']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for model_idx in range(1, 6):
            model_name = f'M{model_idx}'
            accuracies = [all_results[f'Sampling{i}'][model_name]['accuracy'] 
                         for i in range(1, 6)]
            ax1.plot(sampling_names, accuracies, marker='o', linewidth=2, 
                    label=model_name, markersize=8)
        
        ax1.set_xlabel('Sampling Technique', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Across Sampling Techniques', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        for sampling_idx in range(1, 6):
            sampling_name = f'Sampling{sampling_idx}'
            accuracies = [all_results[sampling_name][f'M{i}']['accuracy'] 
                         for i in range(1, 6)]
            ax2.plot(model_names, accuracies, marker='s', linewidth=2, 
                    label=sampling_name, markersize=8)
        
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Sampling Technique Performance Across Models', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


# PART 8: MAIN EXECUTION


def main():
    print("="*80)
    print("CREDIT CARD FRAUD DETECTION - SAMPLING TECHNIQUES COMPARISON")
    print("="*80)
    
    # Upload dataset
    print("\n[STEP 1] Upload your dataset...")
    print("Please upload the 'Creditcard_data.csv' file")
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    print(f"\n✓ File '{file_name}' uploaded successfully!")
    
    # Load and preprocess
    print("\n[STEP 2] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.load_and_split(file_name)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Apply sampling
    print("\n[STEP 3] Applying 5 different sampling techniques...")
    sampler = SamplingTechniques()
    sampling_results = {}
    sampling_methods = ['RandomOverSampling', 'SMOTE', 'ADASYN', 
                       'RandomUnderSampling', 'TomekLinks']
    
    for idx, method in enumerate(sampling_methods, 1):
        print(f"\n  [{idx}/5] Applying {method}...")
        X_sampled, y_sampled = sampler.apply_sampling(X_train, y_train, method)
        sampling_results[f'Sampling{idx}'] = {
            'X': X_sampled, 'y': y_sampled, 'method': method
        }
        print(f"      Balanced dataset size: {X_sampled.shape[0]} samples")
    
    # Train models
    print("\n[STEP 4] Training 5 different ML models on each sampled dataset...")
    trainer = ModelTrainer()
    all_results = {}
    model_names = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'SVM', 'KNN']
    
    for sampling_name, sampling_data in sampling_results.items():
        print(f"\n  Training models on {sampling_data['method']} ({sampling_name})...")
        all_results[sampling_name] = {}
        
        for idx, model_name in enumerate(model_names, 1):
            print(f"    [{idx}/5] Training {model_name}...", end=' ')
            model = trainer.train_model(sampling_data['X'], sampling_data['y'], model_name)
            accuracy = trainer.evaluate_model(model, X_test, y_test)
            all_results[sampling_name][f'M{idx}'] = {
                'model': model, 'accuracy': accuracy, 'model_name': model_name
            }
            print(f"Accuracy: {accuracy:.2f}%")
    
    # Create results
    print("\n[STEP 5] Generating results...")
    evaluator = ModelEvaluator()
    results_df = evaluator.create_results_table(all_results)
    
    print("\n" + "="*80)
    print("RESULTS TABLE - Accuracy Scores (%)")
    print("="*80)
    print(results_df.to_string())
    
    # Best combinations
    best_combos = evaluator.find_best_combinations(all_results)
    print("\n[STEP 6] Top 10 Best Combinations:")
    print("-" * 60)
    for idx, combo in enumerate(best_combos[:10], 1):
        print(f"{idx}. {combo['sampling']} + {combo['model_name']}: {combo['accuracy']:.2f}%")
    
    # Visualizations
    print("\n[STEP 7] Creating visualizations...")
    visualizer = ResultsVisualizer()
    visualizer.create_heatmap(results_df)
    visualizer.create_comparison_plots(all_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


# RUN THE ANALYSIS


if __name__ == "__main__":
    main()
