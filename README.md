# UKF Data Extraction & Baseline Model ($VO_2$ Max Fusion Project)

## Overview
This script processes the raw clinical breath-by-breath dataset to establish our baseline "Smartwatch-Only" $VO_2$ max estimation accuracy. Crucially, it simulates realistic smartwatch PPG noise (sensor inaccuracy + motion artifacts) and exports perfectly aligned time-series arrays ($HR$, $Speed$, $VO_2$) that are ready to be fed directly into our Unscented Kalman Filter (UKF).

## Prerequisites
Ensure you have the following packages installed in your environment:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Import and Run
You do not need to rewrite the extraction logic in your UKF file. You can import this script as a module. 

Assuming this script is saved as `ppg-mapping.py` in the same directory, use the following code in your UKF script:

```python
# Import the function from our prep file
from ppg-mapping import evaluate_vo2_max_clinical_method

# Run the function to get the baseline metrics and the UKF dataset
# You can tweak the noise parameters to test the UKF's robustness
results_df, ukf_data = evaluate_vo2_max_clinical_method(
    subject_info_path='./dataset_ppg/subject-info.csv', 
    measure_path='./dataset_ppg/test_measure.csv',
    noise_std=6.0,       # Standard deviation of baseline PPG noise (bpm)
    artifact_prob=0.05   # 5% chance of a large motion artifact spike
)
```

## Understanding the UKF Data Output
The `ukf_data` object is a Python dictionary. The **keys** are the unique `ID_test` strings (representing individual treadmill tests). The **values** are dictionaries containing the time-series arrays for that specific test.

Here is how the exported data maps to our UKF architecture:

| Dictionary Key | Kalman Filter Variable | Description |
| :--- | :--- | :--- |
| `'time'` | $t$ | The time step array (in seconds) since the test started. |
| `'u_speed'` | $u_k$ (Control Input) | The treadmill speed (km/h) driving the physiological changes. |
| `'z_hr_noisy'` | $z_k$ (Measurement) | The simulated noisy PPG heart rate (bpm). **This is the primary input to the UKF.** |
| `'z_hr_clean'` | Reference Only | The clean ECG heart rate, useful if you want to plot the noise profile. |
| `'x_true_vo2'` | Ground Truth ($x_{true}$) | The actual, smoothed $VO_2$ (mL/kg/min). **Do not feed this to the UKF.** Use it to calculate RMSE at the end. |
| `'R_hr_noise'` | $R$ (Measurement Noise) | Estimated variance of the heart rate sensor. Use this to populate the Measurement Noise Covariance matrix. |
| `'initial_hr_rest'`| Initial State ($x_0$) | The resting HR calculated from the first 60 seconds, useful for initializing the filter state. |

## UKF Integration Template
Here is a skeleton loop showing how you should iterate through the data to run the UKF for every valid test:

```python
import numpy as np

# Iterate through every valid treadmill test
for test_id, test_data in ukf_data.items():
    print(f"Running UKF for Test ID: {test_id}")
    
    # Extract arrays for this specific test
    time_steps = test_data['time']
    u_speed = test_data['u_speed']
    z_hr = test_data['z_hr_noisy']
    true_vo2 = test_data['x_true_vo2']
    
    # 1. Initialize your UKF states here using test_data['initial_hr_rest']
    # ukf = UnscentedKalmanFilter(dim_x=2, dim_z=2) ...
    
    # Array to store the UKF's VO2 estimates over time
    estimated_vo2_trajectory = []
    
    # 2. Run the UKF step-by-step
    for k in range(len(time_steps)):
        # --- PREDICT STEP ---
        # Pass the control input (speed) to your state transition function f(x, u)
        # ukf.predict(u=u_speed[k])
        
        # --- UPDATE STEP ---
        # We also need to inject the generated time-varying lactate signal here!
        # current_lactate = generate_lactate_measurement(time_steps[k], u_speed[k])
        # measurement_vector = np.array([z_hr[k], current_lactate])
        
        # Update the filter with the new measurement
        # ukf.update(measurement_vector)
        
        # Save the current state estimate
        # estimated_vo2_trajectory.append(ukf.x[0]) # Assuming VO2 is state index 0
        pass 
        
    # 3. Calculate MAE and RMSE against 'true_vo2' array for this test to prove the fusion works!
```