import csv

def load_single_row_csv(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        return next(reader) # return first row as list

def load_csvs(csvs):
    results = []
    for csv in csvs:
        results.append(load_single_row_csv)
    
    return results
