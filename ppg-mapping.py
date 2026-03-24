import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_vo2_max_clinical_method(subject_info_path, measure_path):
    #load data
    sub_info = pd.read_csv(subject_info_path)
    sub_info.columns = sub_info.columns.str.strip()
    test_measure = pd.read_csv(measure_path)
    
    results = []
    
    for test_id in sub_info['ID_test'].unique():
        try:
            # subject data
            meta = sub_info[sub_info['ID_test'] == test_id].iloc[0]
            age, weight = meta['Age'], meta['Weight']
            hr_max_theor = 220 - age
            
            #extract data and find resting HR
            data = test_measure[test_measure['ID_test'] == test_id].copy()
            data = data.dropna(subset=['HR', 'Speed', 'VO2'])
            
            #estimate resting HR from the first 30 seconds
            hr_rest = data[data['time'] < 60]['HR'].min()
            if np.isnan(hr_rest) or hr_rest < 40: hr_rest = 65 
            hrr_total = hr_max_theor - hr_rest
            
            stages = []
            # Group by unique speeds 
            for speed, group in data[data['Speed'] > 5.0].groupby('Speed'):
                if len(group) > 25: 
             
                    steady_hr = group['HR'].tail(10).mean()
                    
                    # Convert to %HRR (X-axis)
                    perc_hrr = (steady_hr - hr_rest) / hrr_total
                    
                    # ACSM Formula for VO2 Demand (Y-axis)
                    # VO2 = 3.5 + (0.2 * speed_m_min)
                    speed_m_min = speed * 16.67
                    vo2_demand = 3.5 + (0.2 * speed_m_min)
                    
                    #use the linear aerobic range (45% to 85% HRR)
                    if 0.45 <= perc_hrr <= 0.85:
                        stages.append([perc_hrr, vo2_demand])
            
            if len(stages) < 3: continue
            
            #linear regression on included points
            stages = np.array(stages)
            slope, intercept = np.polyfit(stages[:, 0], stages[:, 1], 1)
            
            # extrapolate
            est_vo2_max = (slope * 1.0) + intercept
            true_vo2_max = data['VO2'].max() / weight
            
            #filter outliers
            if 15 < est_vo2_max < 90:
                results.append({
                    'ID_test': test_id,
                    'True_VO2_Max': true_vo2_max,
                    'Predicted_VO2_Max': est_vo2_max,
                    'Error': est_vo2_max - true_vo2_max,
                    'Pct_Error': abs(est_vo2_max - true_vo2_max) / true_vo2_max * 100
                })
        except: continue

    results_df = pd.DataFrame(results)
    

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Predicted vs Actual
    sns.regplot(data=results_df, x='True_VO2_Max', y='Predicted_VO2_Max', ax=axes[0], 
                scatter_kws={'alpha':0.4, 'color':'teal'}, line_kws={'color':'red'})
    axes[0].plot([20, 80], [20, 80], 'k--', alpha=0.5) # Identity line
    axes[0].set_title('Predicted vs. Actual VO2 Max')
    axes[0].set_xlabel('True VO2 Max (mL/kg/min)')
    axes[0].set_ylabel('Predicted VO2 Max (mL/kg/min)')

    # Plot 2: Error Distribution
    sns.histplot(results_df['Pct_Error'], kde=True, ax=axes[1], color='purple')
    axes[1].set_title('Distribution of Percentage Errors')
    axes[1].set_xlabel('Percentage Error (%)')

    # Plot 3: Error vs True Value (Residuals)
    sns.scatterplot(data=results_df, x='True_VO2_Max', y='Error', ax=axes[2], alpha=0.5)
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_title('Estimation Bias (Error vs True Value)')
    axes[2].set_xlabel('True VO2 Max')
    axes[2].set_ylabel('Absolute Error (mL/kg/min)')

    plt.tight_layout()
    plt.show()

    print(f"Mean Absolute Error: {results_df['Error'].abs().mean():.2f} mL/kg/min")
    print(f"Mean Percentage Error: {results_df['Pct_Error'].mean():.2f}%")
    
    return results_df


results = evaluate_vo2_max_clinical_method('./dataset_ppg/subject-info.csv', './dataset_ppg/test_measure.csv')