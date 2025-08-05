
# src/change_point_analysis.py

import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import os

# Define file paths relative to the current working directory
# We're running the script from /content/Birhan-Energies- so the path is relative to there
data_file_path = 'data/BrentOilPrices.csv'
reports_dir = os.path.join('reports', 'figures')

# Check and print the current working directory to debug the file path
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Check for and create the reports/figures directory if it doesn't exist
if not os.path.exists(reports_dir):
    print(f"Creating '{reports_dir}' directory...")
    os.makedirs(reports_dir)

# Attempt to read the data
try:
    df = pd.read_csv(data_file_path)
except FileNotFoundError:
    print(f"Error: The file was not found at the expected location: {data_file_path}")
    print("Please ensure your 'BrentOilPrices.csv' file is in the 'data' directory.")
    print("Full path being checked:", os.path.abspath(data_file_path))
    # Exit the script gracefully to prevent further errors
    exit()

# Data preprocessing
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])
df['Log_Returns'] = np.log(df['Price']).diff()
clean_df = df.dropna(subset=['Log_Returns'])

y = clean_df['Log_Returns'].values
n_points = len(y)
dates = clean_df.index.values

print(f"Data has been cleaned. Number of data points for modeling: {n_points}")

y_std = np.std(y)
y_mean = np.mean(y)

# Task 2.1: Define and sample from the change point model
with pm.Model() as change_point_model:
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_points - 1)
    mu_1 = pm.Normal("mu_1", mu=y_mean, sigma=y_std*5)
    mu_2 = pm.Normal("mu_2", mu=y_mean, sigma=y_std*5)
    sigma_1 = pm.HalfNormal("sigma_1", sigma=y_std*5)
    sigma_2 = pm.HalfNormal("sigma_2", sigma=y_std*5)
    mu = pm.math.switch(tau > np.arange(n_points), mu_1, mu_2)
    sigma = pm.math.switch(tau > np.arange(n_points), sigma_1, sigma_2)
    likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)
    idata_change = pm.sample(
        draws=2000,
        tune=1000,
        chains=2,
        cores=1,
        init='auto',
        idata_kwargs={'log_likelihood': True}
    )

print("\nSampling for change point model complete.")
print("\n--- Summary of Model Parameters ---")
print(az.summary(idata_change, var_names=["tau", "mu_1", "mu_2", "sigma_1", "sigma_2"]))

tau_posterior = idata_change.posterior["tau"].values.flatten()
most_likely_tau = int(np.mean(tau_posterior))
most_likely_date = dates[most_likely_tau]

print(f"\nMost likely change point index: {most_likely_tau}")
print(f"Most likely change point date: {pd.to_datetime(most_likely_date).strftime('%Y-%m-%d')}")

# Visualize the posterior distribution of the change point
plt.figure(figsize=(12, 6))
plt.hist(tau_posterior, bins=50, density=True, alpha=0.7, color='skyblue')
plt.title('Posterior Distribution of the Change Point (tau)')
plt.xlabel('Time Index')
plt.ylabel('Probability Density')
plt.axvline(most_likely_tau, color='red', linestyle='--', label=f'Most Likely Tau: {most_likely_tau}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(reports_dir, 'tau_posterior.png'))
plt.show()

# Visualize the data and the model's fit
plt.figure(figsize=(12, 6))
plt.plot(dates, y, 'o', label='Log Returns', alpha=0.5)
plt.title('Log Returns with Detected Change Point')
plt.xlabel('Date')
plt.ylabel('Log Returns')

mu_1_mean = idata_change.posterior["mu_1"].values.mean()
mu_2_mean = idata_change.posterior["mu_2"].values.mean()

plt.plot(dates[:most_likely_tau], [mu_1_mean] * most_likely_tau, 'r--', label=f'Mean before tau: {mu_1_mean:.4f}')
plt.plot(dates[most_likely_tau:], [mu_2_mean] * (n_points - most_likely_tau), 'g--', label=f'Mean after tau: {mu_2_mean:.4f}')

plt.legend()
plt.grid(True)
plt.savefig(os.path.join(reports_dir, 'log_returns_fit.png'))
plt.show()


# Task 2.2: Model comparison
print("\n--- Performing Model Comparison ---")

with pm.Model() as null_model:
    mu_null = pm.Normal("mu_null", mu=y_mean, sigma=y_std * 5)
    sigma_null = pm.HalfNormal("sigma_null", sigma=y_std * 5)
    likelihood_null = pm.Normal("likelihood_null", mu=mu_null, sigma=sigma_null, observed=y)
    idata_null = pm.sample(
        draws=2000,
        tune=1000,
        chains=2,
        cores=1,
        init='auto',
        idata_kwargs={'log_likelihood': True}
    )

print("\nSampling for null model complete.")

print("\n--- Model Comparison Summary (using WAIC and LOO) ---")
df_comp_waic = az.compare({"change_point_model": idata_change, "null_model": idata_null}, ic="waic", scale="log")
print(df_comp_waic)

print("\n")

df_comp_loo = az.compare({"change_point_model": idata_change, "null_model": idata_null}, ic="loo", scale="log")
print(df_comp_loo)
