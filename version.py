# Report with Python Version and Package Versions

# Run this script to check the versions on your machine. 
# The versions used in the Project 1 classes will be at the end of the chapter.

import sys
import joblib
import pandas
import sklearn
import flask

packages = [joblib, pandas, sklearn, flask]

print('\nVersions of Python Language and Packages Used in This Chapter:\n')

# Extract the full version string
version_string = sys.version

# Split the string by the first occurrence of space and get the first element
version_number = version_string.split()[0]

print("Python Language Version:", version_number)
for package in packages:
    print(f"{package.__name__} Version:", package.__version__)

