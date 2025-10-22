import pandas as pd
import numpy as np
import re
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Optional imports
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


# ---------- Utility Functions ----------
def parse_total_sqft(value):
    try:
        value = str(value).strip().lower()
        if '-' in value:
            low, high = value.split('-')
            low = float(re.sub(r'[^\d.]', '', low))
            high = float(re.sub(r'[^\d.]', '', high))
            return (low + high) / 2
        match = re.match(r'(\d*\.?\d*)\s*([a-zA-Z\s.]*)', value)
        if not match:
            return None
        num = float(match.group(1))
        unit = match.group(2).replace(' ', '').replace('.', '')
        if unit in ['sqyard', 'sqyards', 'yards']:
            return num * 9
        elif unit in ['acre', 'acres']:
            return num * 43560
        elif unit in ['sqmeter', 'sqmeters', 'meter']:
            return num * 10.7639
        elif unit in ['cent', 'cents']:
            return num * 435.6
        elif unit in ['guntha', 'gunthas']:
            return num * 1089
        elif unit in ['perch', 'perches']:
            return num * 272.25
        elif unit in ['ground', 'grounds']:
            return num * 2400.35
        elif unit == '' or unit == 'sqft':
            return num
        else:
            return None
    except Exception:
        return None


def perform_eda(df, output_dir='./static/plots/'):
    """Perform comprehensive EDA and save plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    eda_summary = {}
    
    print("📊 Generating EDA plots...")
    
    # 1. Dataset Overview
    eda_summary['total_rows'] = int(df.shape[0])
    eda_summary['total_columns'] = int(df.shape[1])
    eda_summary['missing_values'] = df.isnull().sum().to_dict()
    eda_summary['data_types'] = df.dtypes.astype(str).to_dict()
    
    # 2. Price Distribution (Before Outlier Removal)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(df['price'].dropna(), bins=50, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Price (Lakhs)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Price Distribution (Original)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(np.log1p(df['price'].dropna()), bins=50, color='coral', edgecolor='black')
    axes[1].set_xlabel('Log(Price)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-Transformed Price Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    eda_summary['price_stats'] = {
        'mean': float(df['price'].mean()),
        'median': float(df['price'].median()),
        'std': float(df['price'].std()),
        'min': float(df['price'].min()),
        'max': float(df['price'].max())
    }
    
    # 3. Categorical Variables Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Area Type
    area_counts = df['area_type'].value_counts()
    axes[0, 0].bar(range(len(area_counts)), area_counts.values, color='teal')
    axes[0, 0].set_xticks(range(len(area_counts)))
    axes[0, 0].set_xticklabels(area_counts.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Area Types')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Size (BHK)
    size_counts = df['size'].value_counts().head(10)
    axes[0, 1].bar(range(len(size_counts)), size_counts.values, color='salmon')
    axes[0, 1].set_xticks(range(len(size_counts)))
    axes[0, 1].set_xticklabels(size_counts.index, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Top 10 Size Categories (BHK)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top Locations
    location_counts = df['location'].value_counts().head(15)
    axes[1, 0].barh(range(len(location_counts)), location_counts.values, color='mediumseagreen')
    axes[1, 0].set_yticks(range(len(location_counts)))
    axes[1, 0].set_yticklabels(location_counts.index)
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Top 15 Locations')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bath Distribution
    bath_counts = df['bath'].value_counts().sort_index().head(10)
    axes[1, 1].bar(bath_counts.index, bath_counts.values, color='mediumpurple')
    axes[1, 1].set_xlabel('Number of Bathrooms')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Bathroom Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation Heatmap (for numerical features after preprocessing)
    df_temp = df.copy()
    df_temp['size_num'] = df_temp['size'].str.extract(r'(\d+)').astype(float)
    df_temp['total_sqft_num'] = df_temp['total_sqft'].apply(parse_total_sqft)
    
    numerical_features = ['price', 'total_sqft_num', 'size_num', 'bath', 'balcony']
    corr_matrix = df_temp[numerical_features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig(f'{output_dir}correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    eda_summary['correlations'] = corr_matrix.to_dict()
    
    # 5. Price vs Key Features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Price vs Total Sqft
    valid_data = df_temp[['total_sqft_num', 'price']].dropna()
    valid_data = valid_data[(valid_data['total_sqft_num'] < valid_data['total_sqft_num'].quantile(0.99))]
    axes[0, 0].scatter(valid_data['total_sqft_num'], valid_data['price'], 
                       alpha=0.3, s=10, color='steelblue')
    axes[0, 0].set_xlabel('Total Square Feet')
    axes[0, 0].set_ylabel('Price (Lakhs)')
    axes[0, 0].set_title('Price vs Total Square Feet')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Price vs Size (BHK)
    valid_data = df_temp[['size_num', 'price']].dropna()
    valid_data = valid_data[valid_data['size_num'] <= 10]
    axes[0, 1].scatter(valid_data['size_num'], valid_data['price'], 
                       alpha=0.3, s=10, color='coral')
    axes[0, 1].set_xlabel('Size (BHK)')
    axes[0, 1].set_ylabel('Price (Lakhs)')
    axes[0, 1].set_title('Price vs Size (BHK)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Price by Area Type
    df_temp.boxplot(column='price', by='area_type', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Area Type')
    axes[1, 0].set_ylabel('Price (Lakhs)')
    axes[1, 0].set_title('Price Distribution by Area Type')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45, ha='right')
    axes[1, 0].get_figure().suptitle('')
    
    # Price vs Bathrooms
    valid_data = df_temp[['bath', 'price']].dropna()
    valid_data = valid_data[valid_data['bath'] <= 10]
    axes[1, 1].scatter(valid_data['bath'], valid_data['price'], 
                       alpha=0.3, s=10, color='mediumseagreen')
    axes[1, 1].set_xlabel('Number of Bathrooms')
    axes[1, 1].set_ylabel('Price (Lakhs)')
    axes[1, 1].set_title('Price vs Bathrooms')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}price_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ EDA plots saved to {output_dir}")
    return eda_summary


def calculate_vif(X_processed_df, output_dir='./static/plots/'):
    """Calculate VIF for multicollinearity detection"""
    print("📐 Calculating VIF scores...")
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_processed_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_processed_df.values, i) 
                       for i in range(X_processed_df.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    # Plot VIF
    plt.figure(figsize=(12, max(6, len(vif_data) * 0.3)))
    colors = ['red' if x > 10 else 'orange' if x > 5 else 'green' for x in vif_data['VIF']]
    plt.barh(range(len(vif_data)), vif_data['VIF'], color=colors)
    plt.yticks(range(len(vif_data)), vif_data['Feature'])
    plt.xlabel('VIF Score')
    plt.title('Variance Inflation Factor (VIF) for Features\n(Red: VIF>10, Orange: VIF>5, Green: VIF≤5)')
    plt.axvline(x=5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='VIF=5')
    plt.axvline(x=10, color='red', linestyle='--', linewidth=1, alpha=0.7, label='VIF=10')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}vif_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ VIF analysis saved to {output_dir}")
    return vif_data


def preprocess_data(df):
    df = df.dropna(subset=['price'])

    # Outlier removal
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['price'] < Q1 - 1.5 * IQR) | (df['price'] > Q3 + 1.5 * IQR))]

    X = df.drop('price', axis=1)
    y = np.log1p(df['price'])

    # Fill missing values
    for col in ['location', 'size', 'bath', 'balcony']:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    X['society'] = X['society'].fillna('Unknown')

    # Feature engineering
    X['size_num'] = X['size'].str.extract(r'(\d+)').astype(float)
    X['total_sqft_num'] = X['total_sqft'].apply(parse_total_sqft)
    X['total_sqft_num'] = X['total_sqft_num'].fillna(X['total_sqft_num'].median())
    X['bath_per_size'] = X['bath'] / X['size_num'].replace(0, 1)

    numerical_cols = ['total_sqft_num', 'size_num', 'bath', 'balcony', 'bath_per_size']
    categorical_low_card = ['area_type']
    categorical_high_card = ['location', 'society']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat_low', OneHotEncoder(drop='first', sparse_output=False), categorical_low_card),
            ('cat_high', TargetEncoder(), categorical_high_card)
        ]
    )

    X_processed = preprocessor.fit_transform(X, y)
    
    # Create feature names for VIF analysis
    feature_names = numerical_cols.copy()
    
    # Get OneHotEncoder feature names
    ohe = preprocessor.named_transformers_['cat_low']
    ohe_features = ohe.get_feature_names_out(categorical_low_card)
    feature_names.extend(ohe_features)
    
    # Add high cardinality categorical features
    feature_names.extend(categorical_high_card)
    
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    return X_processed, y, preprocessor, X_processed_df

def plot_model_overview(predictions, results, outputdir):
    """
    Create a high-level overview plot for the home page.
    Shows actual vs. predicted prices for the best model.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Identify the best model by Test R2 score
    best_model_name = max(results.items(), key=lambda x: x[1]['TestR2'])[0]
    true = np.array(predictions[best_model_name]['testtrue'])
    pred = np.array(predictions[best_model_name]['testpred'])

    plt.figure(figsize=(10,7))
    plt.scatter(true, pred, alpha=0.4, color='royalblue', s=15, label="Predictions")
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', lw=2, label="Perfect Prediction")
    plt.xlabel("Actual Price (Lakhs)")
    plt.ylabel("Predicted Price (Lakhs)")
    plt.title(f"Best Model Overview: {best_model_name} (Test $R^2$ = {results[best_model_name]['TestR2']:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outputdir}/modeloverview.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model overview plot saved to {outputdir}/modeloverview.png")

# # Insert at the appropriate point after other plots
# plot_model_overview(predictions, results, outputdir="./static/plots")


def train_models(X_train, y_train, X_test, y_test):
    models = {
        'LinearRegression': LinearRegression(),
        'RidgeRegression': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    if XGBRegressor:
        models['XGBoost'] = XGBRegressor(random_state=42, eval_metric='rmse')
    if LGBMRegressor:
        models['LightGBM'] = LGBMRegressor(random_state=42, verbose=-1)
    if CatBoostRegressor:
        models['CatBoost'] = CatBoostRegressor(verbose=0, random_state=42)

    # Hyperparameter tuning for Random Forest
    print("🔧 Tuning Random Forest hyperparameters...")
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [None, 10, 20, 30, 40, 50]},
        cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    models['RandomForest'] = grid_search.best_estimator_
    print(f"✅ Best Random Forest params: {grid_search.best_params_}")

    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"🤖 Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        train_preds = np.expm1(model.predict(X_train))
        test_preds = np.expm1(model.predict(X_test))
        
        train_true = np.expm1(y_train)
        test_true = np.expm1(y_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(train_true, train_preds))
        test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))
        train_mae = mean_absolute_error(train_true, train_preds)
        test_mae = mean_absolute_error(test_true, test_preds)
        train_r2 = r2_score(train_true, train_preds)
        test_r2 = r2_score(test_true, test_preds)
        
        results[name] = {
            'Train_RMSE': round(train_rmse, 2),
            'Test_RMSE': round(test_rmse, 2),
            'Train_MAE': round(train_mae, 2),
            'Test_MAE': round(test_mae, 2),
            'Train_R2': round(train_r2, 3),
            'Test_R2': round(test_r2, 3)
        }
        
        predictions[name] = {
            'test_true': test_true.tolist() if hasattr(test_true, 'tolist') else list(test_true),
            'test_pred': test_preds.tolist() if hasattr(test_preds, 'tolist') else list(test_preds)
        }

    return models, results, predictions


def plot_model_comparison(results, output_dir='./static/plots/'):
    """Create model comparison plots"""
    print("📊 Creating model comparison plots...")
    
    # Extract metrics
    model_names = list(results.keys())
    test_rmse = [results[m]['Test_RMSE'] for m in model_names]
    test_r2 = [results[m]['Test_R2'] for m in model_names]
    test_mae = [results[m]['Test_MAE'] for m in model_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # RMSE Comparison
    axes[0].bar(model_names, test_rmse, color='steelblue')
    axes[0].set_ylabel('RMSE (Lakhs)')
    axes[0].set_title('Model Comparison: RMSE (Lower is Better)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # R² Comparison
    axes[1].bar(model_names, test_r2, color='coral')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Model Comparison: R² (Higher is Better)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([min(test_r2) - 0.05, 1.0])
    
    # MAE Comparison
    axes[2].bar(model_names, test_mae, color='mediumseagreen')
    axes[2].set_ylabel('MAE (Lakhs)')
    axes[2].set_title('Model Comparison: MAE (Lower is Better)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model comparison saved to {output_dir}")


def plot_predictions(predictions, results, output_dir='./static/plots/'):
    """Plot actual vs predicted for best models"""
    print("📊 Creating prediction plots...")
    
    # Find best model by R²
    best_model = max(results.items(), key=lambda x: x[1]['Test_R2'])[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, preds) in enumerate(predictions.items()):
        if idx >= 6:
            break
            
        true_vals = preds['test_true']
        pred_vals = preds['test_pred']
        
        axes[idx].scatter(true_vals, pred_vals, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(min(true_vals), min(pred_vals))
        max_val = max(max(true_vals), max(pred_vals))
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[idx].set_xlabel('Actual Price (Lakhs)')
        axes[idx].set_ylabel('Predicted Price (Lakhs)')
        axes[idx].set_title(f'{name}\nR² = {results[name]["Test_R2"]}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(predictions), 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}prediction_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Prediction plots saved to {output_dir}")


def save_summary(eda_summary, vif_data, results, output_dir='./static/results/'):
    """Save comprehensive summary"""
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'eda_summary': eda_summary,
        'vif_analysis': vif_data.to_dict('records'),
        'model_results': results
    }
    
    # Save as JSON
    with open(f'{output_dir}analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save as readable text
    with open(f'{output_dir}analysis_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BENGALURU HOUSE PRICE PREDICTION - ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📊 DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Rows: {eda_summary['total_rows']}\n")
        f.write(f"Total Columns: {eda_summary['total_columns']}\n\n")
        
        f.write("💰 PRICE STATISTICS\n")
        f.write("-" * 80 + "\n")
        for key, value in eda_summary['price_stats'].items():
            f.write(f"{key.capitalize()}: {value:.2f} Lakhs\n")
        f.write("\n")
        
        f.write("📐 VIF ANALYSIS (Top 10 Features)\n")
        f.write("-" * 80 + "\n")
        for idx, row in vif_data.head(10).iterrows():
            status = "⚠️ HIGH" if row['VIF'] > 10 else "⚡ MODERATE" if row['VIF'] > 5 else "✅ LOW"
            f.write(f"{row['Feature']}: {row['VIF']:.2f} {status}\n")
        f.write("\n")
        
        f.write("🤖 MODEL PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<20} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<10}\n")
        f.write("-" * 80 + "\n")
        for name, metrics in results.items():
            f.write(f"{name:<20} {metrics['Test_RMSE']:<12} {metrics['Test_MAE']:<12} {metrics['Test_R2']:<10}\n")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['Test_R2'])
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"🏆 BEST MODEL: {best_model[0]} (R² = {best_model[1]['Test_R2']})\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Summary saved to {output_dir}")


def main():
    print("=" * 80)
    print("🏡 BENGALURU HOUSE PRICE PREDICTION - ML PIPELINE")
    print("=" * 80 + "\n")
    
    # Load dataset
    print("📂 Loading dataset...")
    df = pd.read_csv("Bengaluru_House_Data.csv")
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.\n")

    # EDA
    print("=" * 80)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    eda_summary = perform_eda(df)
    print()

    # Preprocessing
    print("=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)
    print("🧹 Preprocessing data...")
    X, y, preprocessor, X_processed_df = preprocess_data(df)
    print(f"✅ Processed data shape: {X.shape}")
    print(f"✅ Removed outliers, filled missing values, engineered features\n")

    # VIF Analysis
    print("=" * 80)
    print("STEP 3: VIF ANALYSIS")
    print("=" * 80)
    vif_data = calculate_vif(X_processed_df)
    print()

    # Train-Test Split
    print("=" * 80)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples\n")

    # Model Training
    print("=" * 80)
    print("STEP 5: MODEL TRAINING")
    print("=" * 80)
    models, results, predictions = train_models(X_train, y_train, X_test, y_test)
    print()

    # Performance Summary
    print("=" * 80)
    print("STEP 6: MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Train RMSE':<12} {'Test RMSE':<12} {'Test R²':<10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['Train_RMSE']:<12} {metrics['Test_RMSE']:<12} {metrics['Test_R2']:<10}")
    print()

    # Visualization
    print("=" * 80)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("=" * 80)
    plot_model_comparison(results)
    plot_predictions(predictions, results)
    print()

    # Save Models
    print("=" * 80)
    print("STEP 8: SAVING MODELS & ARTIFACTS")
    print("=" * 80)
    model_dir = "./saved_models/"
    os.makedirs(model_dir, exist_ok=True)
    
    for name, model in models.items():
        filename = f"{model_dir}{name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump({'model': model, 'preprocessor': preprocessor}, f)
        print(f"✅ Saved {name} → {filename}")
    
    # Save Results
    results_dir = "./static/results/"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}model_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"✅ Saved model results → {results_dir}model_results.pkl")
    
    with open(f"{results_dir}predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)
    print(f"✅ Saved predictions → {results_dir}predictions.pkl")
    
    # Save Summary
    print()
    print("=" * 80)
    print("STEP 9: GENERATING COMPREHENSIVE SUMMARY")
    print("=" * 80)
    save_summary(eda_summary, vif_data, results)
    
    # Save metadata for Flask app
    metadata = {
        'locations': sorted(df['location'].dropna().unique().tolist()),
        'area_types': sorted(df['area_type'].dropna().unique().tolist()),
        'size_options': sorted([str(x) for x in df['size'].dropna().unique()]),
        'bath_range': [int(df['bath'].min()), int(df['bath'].max())],
        'balcony_range': [int(df['balcony'].min()), int(df['balcony'].max())],
        'sqft_range': [float(df['total_sqft'].apply(parse_total_sqft).min()), 
                       float(df['total_sqft'].apply(parse_total_sqft).max())],
        'best_model': max(results.items(), key=lambda x: x[1]['Test_R2'])[0],
        'model_list': list(models.keys()),
        'rows': int(df.shape[0]),
        'features': int(df.shape[1])
    }
    
    with open(f"{results_dir}metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Saved Flask app metadata → {results_dir}metadata.json")
    
    print()
    print("=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("   📊 Plots:")
    print("      - static/plots/price_distribution.png")
    print("      - static/plots/categorical_analysis.png")
    print("      - static/plots/correlation_heatmap.png")
    print("      - static/plots/price_relationships.png")
    print("      - static/plots/vif_analysis.png")
    print("      - static/plots/model_comparison.png")
    print("      - static/plots/prediction_plots.png")
    print("   🤖 Models:")
    for name in models.keys():
        print(f"      - saved_models/{name}.pkl")
    print("   📈 Results:")
    print("      - static/results/model_results.pkl")
    print("      - static/results/predictions.pkl")
    print("      - static/results/analysis_summary.json")
    print("      - static/results/analysis_summary.txt")
    print("      - static/results/metadata.json")
    print("\n🚀 Ready for Flask App Development!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
