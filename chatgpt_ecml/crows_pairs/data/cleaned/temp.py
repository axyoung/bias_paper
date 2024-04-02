import os

# Get the current directory
current_dir = os.getcwd()

# List all files in the current directory
file_names = os.listdir(current_dir)

# Print the file names
for file_name in file_names:
    print(file_name)
