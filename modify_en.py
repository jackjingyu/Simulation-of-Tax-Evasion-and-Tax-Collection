# Import required libraries
import pandas as pd  # For data manipulation and analysis
import os  # For operating system dependent functionality

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Iterate through a range of values (0.2 to 0.9 in 0.1 increments)
for i in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    # Construct the file path by joining the script directory with the filename
    # The filename follows the pattern '0.0cost_simulation_resultsX.xlsx' where X is the current value of i
    file_path = os.path.join(script_dir, f'0.0cost_simulation_results{i}.xlsx')
    
    # Read the Excel file into a pandas DataFrame
    # Specify the sheet name as 'Sheet1'
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # Get a list of all column names in the DataFrame
    cols = df.columns.tolist()

    # Check if both 'meat' and 'rice' columns exist in the DataFrame
    if 'meat' in cols and 'rice' in cols:
        # Reorder columns: 'meat' and 'rice' first, followed by all other columns
        # This creates a new column order list
        cols = ['meat', 'rice'] + [col for col in cols if col not in ['meat', 'rice']]
        
        # Reindex the DataFrame with the new column order
        df = df[cols]
        
        # Save the modified DataFrame back to the original Excel file
        # index=False prevents pandas from writing row indices to the file
        df.to_excel(file_path, index=False)
        
        # Print the first few rows of the DataFrame to verify the changes
        print(df.head())
    else:
        # Print a message if either 'meat' or 'rice' column is missing
        print("'meat' or 'rice' column does not exist in the DataFrame")