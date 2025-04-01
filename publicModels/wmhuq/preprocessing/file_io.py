import csv

def load_csv(file_path):
    results = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            results.append(row)
    return results
