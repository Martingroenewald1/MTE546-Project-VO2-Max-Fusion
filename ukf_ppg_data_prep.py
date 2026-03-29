import pandas as pd
import numpy as np

def process_single_test(test_id, age, weight, data, noise_std=6.0, artifact_prob=0.05):
    """
    Takes in raw breath data for ONE subject, simulates smartwatch noise, 
    and formats it into for UKF.
    """
    try:
        
        data = data.dropna(subset=['HR', 'Speed', 'VO2']).copy()
        if len(data) == 0: 
            return None
        
        hr_max_theor = 220 - age
        
       
        data['VO2_smooth'] = data['VO2'].rolling(window=15, min_periods=1).mean()
        
        #simulate smartwatch noise
        base_noise = np.random.normal(loc=0, scale=noise_std, size=len(data))
        artifacts = np.zeros(len(data))
        artifact_indices = np.random.rand(len(data)) < artifact_prob
        artifacts[artifact_indices] = np.random.normal(loc=0, scale=15.0, size=np.sum(artifact_indices))
        
        data['HR_noisy'] = data['HR'] + base_noise + artifacts
        data['HR_noisy'] = np.clip(data['HR_noisy'], a_min=40, a_max=220)
       
        # Filter Initialization Values (Resting HR & Sensor Noise)
        hr_rest = data[data['time'] < 60]['HR_noisy'].min()
        if np.isnan(hr_rest) or hr_rest < 40: 
            hr_rest = 65 
        
        hr_variances = [] 
        for speed, group in data[data['Speed'] > 5.0].groupby('Speed'):
            if len(group) > 25: 
                hr_variances.append(group['HR_noisy'].tail(15).var())
        
        # Fallback to theoretical variance if no steady states were found
        r_noise = np.mean(hr_variances) if hr_variances else (noise_std ** 2)
        
        # Format into a Pandas DataFrame block for this specific test
        test_df = pd.DataFrame({
            'ID_test': test_id,
            'time': data['time'].values,
            'u_speed': data['Speed'].values,          
            'z_hr_clean': data['HR'].values,          
            'z_hr_noisy': data['HR_noisy'].values,    
            'x_true_vo2': data['VO2_smooth'].values / weight, 
            'R_hr_noise': r_noise,      
            'initial_hr_rest': hr_rest,
            'hr_max': hr_max_theor
        })
        
        return test_df
            
    except Exception as e: 
        return None



if __name__ == "__main__":
    
    sub_info = pd.read_csv('./dataset_ppg/subject-info.csv')
    sub_info.columns = sub_info.columns.str.strip()
    test_measure = pd.read_csv('./dataset_ppg/test_measure.csv')
    
    all_processed_tests = []
    
    for test_id in sub_info['ID_test'].unique():
        
        meta = sub_info[sub_info['ID_test'] == test_id].iloc[0]
        age = meta['Age']
        weight = meta['Weight']
    
        subject_data = test_measure[test_measure['ID_test'] == test_id]
        
        # 3. Call our base function for just this one subject
        processed_df = process_single_test(
            test_id=test_id, 
            age=age, 
            weight=weight, 
            data=subject_data,
            noise_std=6.0, 
            artifact_prob=0.05
        )
    
        if processed_df is not None:
            all_processed_tests.append(processed_df)

    if all_processed_tests:
        final_df = pd.concat(all_processed_tests, ignore_index=True)
        output_file = 'ukf_ready_data.csv'
        final_df.to_csv(output_file, index=False)
        print(f"Success! Processed {len(all_processed_tests)} subjects and saved to '{output_file}'")
    else:
        print("Error: No data was processed.")