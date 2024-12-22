import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic data
data = {
    'Soil Type': np.random.choice(['Sandy', 'Clay', 'Loam'], 1000),
    'Rainfall (mm)': np.random.uniform(200, 1500, 1000),
    'Temperature (°C)': np.random.uniform(10, 40, 1000),
    'Humidity (%)': np.random.uniform(20, 90, 1000),
    'pH Level': np.random.uniform(5.0, 8.5, 1000),
    'Nitrogen (N)': np.random.uniform(0, 100, 1000),
    'Phosphorus (P)': np.random.uniform(0, 100, 1000),
    'Potassium (K)': np.random.uniform(0, 100, 1000),
    'Irrigation Type': np.random.choice(['Drip', 'Flood', 'Sprinkler'], 1000),
    'Farm Size (hectares)': np.random.uniform(0.5, 20, 1000),
    'Season': np.random.choice(['Winter', 'Summer', 'Rainy'], 1000),
    'Altitude (m)': np.random.uniform(50, 1500, 1000),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'Sunlight Hours': np.random.uniform(6, 12, 1000),
    'Water Availability (liters)': np.random.uniform(1000, 5000, 1000),
    'Recommended Crop': np.random.choice(
        ['Wheat', 'Rice', 'Maize', 'Sugarcane', 'Barley', 'Soybean', 'Cotton', 'Tomato', 'Potato'], 1000
    )
}

# Create the dataset
dataset = pd.DataFrame(data)

# Save the clean dataset
dataset.to_csv("crop_recommendation_data.csv", index=False)
print("Synthetic dataset generated and saved as 'crop_recommendation_data.csv'.")

# Step 2: Introduce inconsistencies
# Introduce Missing Values
missing_indices_rainfall = np.random.choice(dataset.index, size=int(0.05 * len(dataset)), replace=False)
dataset.loc[missing_indices_rainfall, 'Rainfall (mm)'] = np.nan

missing_indices_ph = np.random.choice(dataset.index, size=int(0.03 * len(dataset)), replace=False)
dataset.loc[missing_indices_ph, 'pH Level'] = np.nan

# Introduce Duplicates
duplicate_indices = np.random.choice(dataset.index, size=int(0.02 * len(dataset)), replace=False)
duplicates = dataset.loc[duplicate_indices]
dataset = pd.concat([dataset, duplicates], ignore_index=True)

# Introduce Outliers
outlier_indices_nitrogen = np.random.choice(dataset.index, size=int(0.01 * len(dataset)), replace=False)
dataset.loc[outlier_indices_nitrogen, 'Nitrogen (N)'] = dataset['Nitrogen (N)'].max() * 5

outlier_indices_temp = np.random.choice(dataset.index, size=int(0.01 * len(dataset)), replace=False)
dataset.loc[outlier_indices_temp, 'Temperature (°C)'] = np.random.choice([-50, 100], size=len(outlier_indices_temp))

# Introduce Noisy Data
noise_indices_humidity = np.random.choice(dataset.index, size=int(0.03 * len(dataset)), replace=False)
dataset.loc[noise_indices_humidity, 'Humidity (%)'] += np.random.uniform(-20, 20, size=len(noise_indices_humidity))

# Save the inconsistent dataset
dataset.to_csv("crop_recommendation_data_inconsistent.csv", index=False)
print("Inconsistent dataset saved as 'crop_recommendation_data_inconsistent.csv'.")

# Summary of inconsistencies introduced
print("\nSummary of Inconsistencies Introduced:")
print(f"Missing values in 'Rainfall (mm)': {dataset['Rainfall (mm)'].isna().sum()}")
print(f"Missing values in 'pH Level': {dataset['pH Level'].isna().sum()}")
print(f"Total duplicates: {dataset.duplicated().sum()}")
print(f"Outliers added to 'Nitrogen (N)': {len(outlier_indices_nitrogen)}")
print(f"Outliers added to 'Temperature (°C)': {len(outlier_indices_temp)}")
print(f"Noisy values added to 'Humidity (%)': {len(noise_indices_humidity)}")
