import os
import csv
import pandas as pd

# Redo the response_category column for every file in the directory
directory = '/home/cixay/Davis/ChatGPT/stereoset/data/cleaned/'
target_directory = directory + 'redo'

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(directory, filename))\

        # add a new column to the dataframe called correct_stereotype
        df['response_category'] = ''
        # for each column
        for index, row in df.iterrows():
            # if response is A or B or C then match it with the corresponding category label_A, Label_B, Label_C and fill in the response_category column
            if row['Response_label'] == 'A':
                df.at[index, 'response_category'] = row['label_A']
            elif row['Response_label'] == 'B':
                df.at[index, 'response_category'] = row['label_B']
            elif row['Response_label'] == 'C':
                df.at[index, 'response_category'] = row['label_C']
            else:
                df.at[index, 'response_category'] = "X"

        # Sort rows by values
        df = df.sort_values(by=['response_category'])

        # Write the cleaned dataframe back the folder ./cleaned
        df.to_csv(os.path.join(target_directory, filename), index=False)
