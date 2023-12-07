import os

import pandas as pd

# Directory containing the CSV files
directory = '/home/sayem/Desktop/SODIndoorLoc/data/raw_data/Train'

# File to building ID mapping
building_ids = {
    'Training_CETC331.csv': 1,
    'Training_HCXY_All_Avg.csv': 2,
    'Training_SYL_All_Avg.csv': 3
}

# Process each CSV file
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        # Correcting BuildingID based on file name
        df['BuildingID'] = building_ids[filename]

        # Correcting FloorID based on file name
        if 'CETC331' in filename:
            # Ensuring FloorID is between 1 to 3 for CETC331
            df['FloorID'] = df['FloorID'].clip(lower=1, upper=3)
        else:
            # Setting FloorID as 4 for HCXY and SYL
            df['FloorID'] = 4

        # Ensure SceneID is between 1 to 3
        df['SceneID'] = df['SceneID'].clip(lower=1, upper=3)

        # Save the corrected DataFrame back to the same file
        df.to_csv(file_path, index=False)

print("Files processed and saved.")
