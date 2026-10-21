"""
Generate 5000 synthetic IVF patient records for ML training.
This script creates realistic IVF success data with 8 features that match
the existing model structure in the application.

Features:
- age: Patient age (22-45)
- bmi: Body Mass Index (18-35)
- amh: Anti-Müllerian Hormone level (0.2-6.0 ng/mL)
- fsh: Follicle Stimulating Hormone (3-18 mIU/mL)
- previous_ivf: Number of previous IVF cycles (0-5)
- stress: Stress level (1-5)
- sleep_hours: Average sleep hours (4-9)
- exercise_min: Daily exercise minutes (0-60)
"""

import pandas as pd
import random
import os

# Set random seed for reproducibility
random.seed(42)

def generate_ivf_records(num_records=5000):
    """Generate synthetic IVF patient records with realistic success patterns."""
    
    rows = []
    
    for i in range(num_records):
        # Basic demographics
        age = random.randint(22, 45)
        bmi = round(random.uniform(18, 35), 1)
        
        # Hormonal markers
        amh = round(random.uniform(0.2, 6.0), 2)
        fsh = round(random.uniform(3, 18), 2)
        
        # History
        previous_ivf = random.randint(0, 5)
        
        # Lifestyle factors
        stress = random.randint(1, 5)
        sleep_hours = round(random.uniform(4, 9), 1)
        exercise_min = random.randint(0, 60)
        
        # Realistic success rule based on medical criteria
        # Success factors:
        # - Age < 35 is optimal
        # - AMH > 2 indicates good ovarian reserve
        # - BMI in healthy range (18-25) is better
        # - FSH < 10 is optimal
        # - Lower stress improves success
        # - Good sleep (7-9 hours) helps
        # - Moderate exercise is beneficial
        
        # Calculate success probability factors
        success_score = 0
        
        # Age factor (most important)
        if age < 30:
            success_score += 3
        elif age < 35:
            success_score += 2
        elif age < 40:
            success_score += 0
        else:
            success_score -= 2
        
        # AMH factor
        if amh >= 3:
            success_score += 2
        elif amh >= 2:
            success_score += 1
        elif amh >= 1:
            success_score += 0
        else:
            success_score -= 1
        
        # BMI factor
        if 18 <= bmi <= 24:
            success_score += 1
        elif 25 <= bmi <= 29:
            success_score += 0
        else:
            success_score -= 1
        
        # FSH factor
        if fsh < 8:
            success_score += 2
        elif fsh < 10:
            success_score += 1
        elif fsh < 12:
            success_score += 0
        else:
            success_score -= 1
        
        # Previous IVF (fewer is better)
        if previous_ivf == 0:
            success_score += 1
        elif previous_ivf <= 2:
            success_score += 0
        else:
            success_score -= 1
        
        # Stress factor
        if stress <= 2:
            success_score += 1
        elif stress <= 3:
            success_score += 0
        else:
            success_score -= 1
        
        # Sleep factor
        if 7 <= sleep_hours <= 9:
            success_score += 1
        elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:
            success_score += 0
        else:
            success_score -= 1
        
        # Exercise factor
        if 20 <= exercise_min <= 45:
            success_score += 1
        elif exercise_min == 0:
            success_score -= 1
        # 0 for other values
        
        # Determine success based on score
        # Higher score = more likely success
        # With score >= 2, likely success (60% chance)
        # With score < 0, likely failure (40% chance)
        
        if success_score >= 3:
            success_prob = 0.85
        elif success_score >= 1:
            success_prob = 0.65
        elif success_score >= -1:
            success_prob = 0.45
        elif success_score >= -3:
            success_prob = 0.25
        else:
            success_prob = 0.10
        
        # Determine success (1) or failure (0)
        success = 1 if random.random() < success_prob else 0
        
        rows.append([
            age,
            bmi,
            amh,
            fsh,
            previous_ivf,
            stress,
            sleep_hours,
            exercise_min,
            success
        ])
    
    # Column names matching the existing model
    columns = [
        "age",
        "bmi", 
        "amh",
        "fsh",
        "previous_ivf",
        "stress",
        "sleep_hours",
        "exercise_min",
        "IVF_Success"
    ]
    
    df = pd.DataFrame(rows, columns=columns)
    return df


def main():
    """Main function to generate and save the dataset."""
    
    print("Generating 5000 synthetic IVF records...")
    
    # Generate the data
    df = generate_ivf_records(5000)
    
    # Display some statistics
    print(f"\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"\nSuccess rate: {df['IVF_Success'].mean()*100:.1f}%")
    print(f"\nFeature ranges:")
    for col in df.columns[:-1]:  # Exclude success column
        print(f"  {col}: {df[col].min():.1f} - {df[col].max():.1f}")
    
    # Save to CSV
    output_file = "ivf_5000_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Dataset saved to: {output_file}")
    
    # Also save to Excel for consistency with existing system
    excel_file = "ivf_5000_dataset.xlsx"
    
    # Create a modified sheet name for the Excel file (Excel sheet names have 31 char limit)
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="IVF_Success_Data", index=False)
    
    print(f"✅ Dataset also saved to: {excel_file}")
    
    return df


if __name__ == "__main__":
    main()
