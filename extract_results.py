# This is the script to extract the results from the saved logs
import os
import csv
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--save-path', default='./save/logs_FATIG_Deformer')
args = parser.parse_args()

logs_directory = args.save_path


# Function to read and log the results from a CSV file
def read_and_log_csv(folder_path):
    # Check if "results.csv" exists in the folder
    csv_file_path = os.path.join(folder_path, "result.csv")
    if not os.path.exists(csv_file_path):
        return  # Skip if "results.csv" does not exist in the folder

    # Read the content of the CSV file
    with open(csv_file_path, mode="r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Read the header row
        data = list(csv_reader)    # Read the data rows
    metric = []
    values = []
    for i, row in enumerate(data):
        metric.append(row[0])
        values.append(float(row[1]))
    return metric, values

values = []

for subfolder in os.listdir(logs_directory):
    subfolder_path = os.path.join(logs_directory, subfolder)
    if os.path.isdir(subfolder_path):
        metric, value = read_and_log_csv(subfolder_path)
        values.append(value)

header = metric
values = np.array(values)  # sub, 4
mean = np.mean(values, axis=0)  # 4
std = np.std(values, axis=0)  # 4
for i, m in enumerate(header):
    print('Mean {}:{} ({})'.format(m, mean[i], std[i]))
