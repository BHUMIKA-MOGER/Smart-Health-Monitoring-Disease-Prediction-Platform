import pandas as pd
import numpy as np

# --- 1. Define the Scoring Logic ---
def calculate_diabetes_score(df):
    """
    Calculates the Total Score and assigns the Condition based on the 
    user-defined scoring model, using the 9 symptom columns.
    """
    
    # 1. Define the columns for positive and negative scores
    
    # Positive symptoms (weighted * 2): High Sugar indicators (5 columns)
    positive_cols = [
        'Feeling very thirsty often', 
        'Urinating frequently', 
        'Always feeling hungry or tired', 
        'Having dry mouth or dry skin', 
        'Feels sleepy or slow most of the day'
    ]
    
    # Negative symptoms (weighted * -1): Low Sugar indicators (4 columns)
    negative_cols = [
        'Feeling sudden shakiness or dizziness', 
        'Sweating a lot suddenly', 
        'Feels cold or clammy sometimes', 
        'Fast heartbeat or irritability'
    ]
    
    symptom_cols = positive_cols + negative_cols

    # Check if the 9 required symptom columns exist in the DataFrame
    missing_cols = [col for col in symptom_cols if col not in df.columns]
    
    if missing_cols:
        # Provide more detail in the error message to help the user fix the Excel file
        raise ValueError(
            f"Error: One or more required symptom columns are missing in the Excel file. "
            f"Please ensure the following names exist and are spelled EXACTLY (check for extra spaces!): {', '.join(missing_cols)}"
        )

    # --- CRITICAL FIX for the '<= not supported' error ---
    # 2. Force the symptom columns to be numeric, handling any remaining strings/NaNs
    
    # 3. Calculate the Total Score
    # Score = (Positive Symptoms Sum) * 2 + (Negative Symptoms Sum) * (-1)
    
    positive_score = df[positive_cols].sum(axis=1) * 2
    negative_score = df[negative_cols].sum(axis=1) * -1
    
    df['Total Score'] = positive_score + negative_score

    # 4. Assign the Condition
    # Rules: <= -2 --> Low Sugar; -1 to +2 --> Normal; >= +3 --> High Sugar
    def assign_condition(score):
        # Now 'score' is guaranteed to be a number, so comparison works
        if score <= -2:
            return 'Low Sugar'
        elif score >= 3:
            return 'High Sugar'
        else:
            return 'Normal'

    df['Condition'] = df['Total Score'].apply(assign_condition)
    
    return df

# --- 2. Main Execution Block ---
if __name__ == '__main__':
    file_path = 'dia.xlsx'
    
    print(f"Loading data from {file_path}...")
    
    try:
        # Load the Excel file. Assumes the data is in the first sheet.
        df = pd.read_excel(file_path, sheet_name=0)
        
        # --- CRITICAL FIX 1: Clean Column Names ---
        # Strip leading/trailing whitespace from ALL column names to prevent mismatch errors
        df.columns = df.columns.str.strip() 
        
        # --- CRITICAL FIX 2: Handle 1(Yes)/0(No) strings in the data ---
        # Replace the full strings used in the Excel sheet with simple integers (1 or 0)
        df.replace({'1 (Yes)': 1, '0 (No)': 0, 'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}, inplace=True)

        # Force the symptom columns to be numeric, handling any remaining strings/NaNs
        symptom_cols = [
            'Feeling very thirsty often', 
            'Urinating frequently', 
            'Always feeling hungry or tired', 
            'Having dry mouth or dry skin', 
            'Feels sleepy or slow most of the day',
            'Feeling sudden shakiness or dizziness', 
            'Sweating a lot suddenly', 
            'Feels cold or clammy sometimes', 
            'Fast heartbeat or irritability'
        ]
        
        for col in symptom_cols:
            # Convert to numeric, coercing any remaining text or unexpected values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill any NaN values (including those created by 'coerce') with 0 before summing
        df[symptom_cols] = df[symptom_cols].fillna(0)
        
        # Apply the Model
        df_result = calculate_diabetes_score(df.copy())
        
        # Output Results
        print("\n--- Model Results ---")
        # Check if 'ID' column exists before trying to print it
        if 'ID' in df_result.columns:
            print(df_result[['ID', 'Total Score', 'Condition']])
        else:
             print(df_result[['Total Score', 'Condition']])
             
        # Save the results
        output_file_path = 'dia_results.xlsx'
        df_result.to_excel(output_file_path, index=False)
        print(f"\nResults saved successfully to {output_file_path}")
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Make sure it is in the same directory as the script.")
    except ValueError as e:
        print(f"Error processing data: {e}")
    except Exception as e:
        # Print the original error for better debugging
        print(f"An unexpected error occurred: {e}") 
