import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import HuberRegressor 

def evaluate_vo2_max_clinical_method(subject_info_path, measure_path, noise_std=6.0, artifact_prob=0.05):
    sub_info = pd.read_csv(subject_info_path)
    sub_info.columns = sub_info.columns.str.strip()
    test_measure = pd.read_csv(measure_path)
    
    results = []
    ukf_export_data = {} 
    
    for test_id in sub_info['ID_test'].unique():
        try:
            meta = sub_info[sub_info['ID_test'] == test_id].iloc[0]
            age, weight = meta['Age'], meta['Weight']
            hr_max_theor = 220 - age
            
            data = test_measure[test_measure['ID_test'] == test_id].copy()
            data = data.dropna(subset=['HR', 'Speed', 'VO2'])
            data['VO2_smooth'] = data['VO2'].rolling(window=15, min_periods=1).mean()
            
            #base noise to throw on top of sensor readings
            base_noise = np.random.normal(loc=0, scale=noise_std, size=len(data))
            
            # larger spikes (artifacts) caused by random spikes in running
            artifacts = np.zeros(len(data))
            artifact_indices = np.random.rand(len(data)) < artifact_prob
            artifacts[artifact_indices] = np.random.normal(loc=0, scale=15.0, size=np.sum(artifact_indices))
            
            #combining noise with actual HR data
            data['HR_noisy'] = data['HR'] + base_noise + artifacts
            data['HR_noisy'] = np.clip(data['HR_noisy'], a_min=40, a_max=220)
           
            #estimate resting hr
            hr_rest = data[data['time'] < 60]['HR_noisy'].min()
            if np.isnan(hr_rest) or hr_rest < 40: hr_rest = 65 
            hrr_total = hr_max_theor - hr_rest
            
            stages = []
            hr_variances = [] 
            
            for speed, group in data[data['Speed'] > 5.0].groupby('Speed'):
                if len(group) > 25: 
                    # steady state hr in noisy window
                    steady_window = group['HR_noisy'].tail(15)
                    steady_hr = steady_window.mean()
                    
                    hr_variances.append(steady_window.var())
                    
                    perc_hrr = (steady_hr - hr_rest) / hrr_total
                    speed_m_min = speed * 16.67
                    vo2_demand = 3.5 + (0.2 * speed_m_min)
                    
                    if 0.45 <= perc_hrr <= 0.85:
                        stages.append([perc_hrr, vo2_demand])
            
            if len(stages) < 3: continue
            
            stages = np.array(stages)
            X = stages[:, 0].reshape(-1, 1) 
            y = stages[:, 1]
            
            huber = HuberRegressor()
            huber.fit(X, y)
            slope = huber.coef_[0]
            intercept = huber.intercept_
            
            est_vo2_max = (slope * 1.0) + intercept
            true_vo2_max = data['VO2_smooth'].max() / weight
            
            if 15 < est_vo2_max < 90:
                error = est_vo2_max - true_vo2_max
                results.append({
                    'ID_test': test_id,
                    'True_VO2_Max': true_vo2_max,
                    'Predicted_VO2_Max': est_vo2_max,
                    'Error': error,
                    'Pct_Error': abs(error) / true_vo2_max * 100
                })
                
            
                ukf_export_data[test_id] = {
                    'time': data['time'].values,
                    'u_speed': data['Speed'].values,          
                    'z_hr_clean': data['HR'].values,          
                    'z_hr_noisy': data['HR_noisy'].values,    
                    'x_true_vo2': data['VO2_smooth'].values / weight, 
                    'R_hr_noise': np.mean(hr_variances),      
                    'initial_hr_rest': hr_rest                
                }
                
        except: continue

    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("Error: No tests met the criteria.")
        return results_df, ukf_export_data

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.regplot(data=results_df, x='True_VO2_Max', y='Predicted_VO2_Max', ax=axes[0], 
                scatter_kws={'alpha':0.4, 'color':'teal'}, line_kws={'color':'red'})
    axes[0].plot([20, 80], [20, 80], 'k--', alpha=0.5) 
    axes[0].set_title('Predicted vs. Actual VO2 Max (Noisy PPG)')
    axes[0].set_xlabel('True VO2 Max (mL/kg/min)')
    axes[0].set_ylabel('Predicted VO2 Max (mL/kg/min)')

    sns.histplot(results_df['Pct_Error'], kde=True, ax=axes[1], color='purple')
    axes[1].set_title('Distribution of Percentage Errors')
    axes[1].set_xlabel('Percentage Error (%)')

    sns.scatterplot(data=results_df, x='True_VO2_Max', y='Error', ax=axes[2], alpha=0.5)
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_title('Estimation Bias (Error vs True Value)')
    axes[2].set_xlabel('True VO2 Max')
    axes[2].set_ylabel('Absolute Error (mL/kg/min)')

    plt.tight_layout()
    plt.show()

    print(f"Baseline (Smartwatch Only) Mean Absolute Error: {results_df['Error'].abs().mean():.2f} mL/kg/min")
    rmse = np.sqrt((results_df['Error']**2).mean())
    print(f"Baseline (Smartwatch Only) Root Mean Square Error: {rmse:.2f} mL/kg/min")
    
    return results_df, ukf_export_data

results_df, ukf_data = evaluate_vo2_max_clinical_method('./dataset_ppg/subject-info.csv', './dataset_ppg/test_measure.csv')