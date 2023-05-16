import argparse
import os
import re
import csv

# Parse command arguments
parser = argparse.ArgumentParser(description="Count variable occurrences in Python files")
parser.add_argument("--input-variables", type=str, required=True, help="File containing variables to analyze")
parser.add_argument("--dirs-to-search", type=str, required=True, help="List of directories separated by commas to search")
parser.add_argument("--output-csv", type=str, required=True, help="CSV containing the results")
args = parser.parse_args()

# Read input variables
with open(args.input_variables, "r") as f:
    variables = [line.strip() for line in f.readlines()]

# Initialize the dictionary to store variable counts
var_count = {var: 0 for var in variables}

# Get the list of directories to search
dirs_to_search = args.dirs_to_search.split(",")

# Define a function to recursively search for Python files
def search_python_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                yield os.path.join(root, file)

# Count variable occurrences in Python files
for directory in dirs_to_search:
    for python_file in search_python_files(directory):
        with open(python_file, "r") as f:
            content = f.read()
            for var in variables:
                count = len(re.findall(rf"\b{var}\b", content))
                var_count[var] += count

# Write the results to a CSV file
with open(args.output_csv, "w", encoding="UTF8") as f:
    writer = csv.writer(f)
    writer.writerow(["Variable", "Count"])
    for var, count in var_count.items():
        writer.writerow([var, count])
