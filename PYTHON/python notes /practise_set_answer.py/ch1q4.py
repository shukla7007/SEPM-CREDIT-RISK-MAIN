import os

# Specify the directory you want to list
directory = '/'

try:
    # List all the files and directories in the specified path
    contents = os.listdir(directory)
    
    # Print each item in the directory
    for item in contents:
        print(item)
except FileNotFoundError:
    print("The specified directory was not found.")
except PermissionError:
    print("You do not have permission to access this directory.")
