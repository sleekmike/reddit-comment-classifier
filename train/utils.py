import pandas as pd
import numpy as np
import random
import string
import json
from preprocessing import cleaner
import psycopg2
import csv

# ===============================================================> Download Database to CSV File <================================================ # 
def download_data_to_csv():
    # Try to establish a connection to the database
    try:
        # Connect to your postgres DB
        conn = psycopg2.connect(
            dbname="",
            user="",
            password="",
            host=""
        )
        # Open a cursor to perform database operations
        cur = conn.cursor()
        
        # Define the tables to be processed and their corresponding file names
        tables = {
            'reddit_usernames_comments': 'reddit_usernames_comments.csv',
            'reddit_usernames': 'reddit_usernames.csv'
        }
        # Process each table
        for table_name, file_name in tables.items():
            print(f"Processing table: {table_name}")   
            # Execute query to fetch all data from the table
            cur.execute(f"SELECT * FROM {table_name};")
            rows = cur.fetchall()
            # Get column headers
            col_names = [desc[0] for desc in cur.description]
            # Write to CSV file
            with open(file_name, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(col_names)  # Write the headers first
                writer.writerows(rows)  # Then write all the data
            print(f"Data from {table_name} written to {file_name}")
    except psycopg2.Error as e:
        print("Unable to connect or execute database operations.")
        print(e)
    
# Run the function to download data 
#download_data_to_csv()

def check_missing_labels(df):
    # Find rows where 'Label' is NaN
    missing_labels = df[df['Label'].isna()]

    # Print the row numbers and corresponding usernames
    print("Rows with missing labels:")
    for index, row in missing_labels.iterrows():
        print(f"Row: {index}, Username: {row['username']}")

    # Count the number of occurrences
    missing_count = missing_labels.shape[0]
    print(f"Total rows with missing labels: {missing_count}")

def clean_and_prepare_data(df):
    # Remove rows where either 'Label' or 'comments' is empty
    df = df.dropna(subset=['Label', 'comments'])
    # Function to generate a random alphanumeric string of length 4-6
    def generate_random_username():
        length = random.randint(4, 6)  # Choose a random length between 4 and 6
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for i in range(length))
    # Check for empty usernames and assign a random username if needed
    df.loc[df['username'].isna(), 'username'] = df[df['username'].isna()].apply(lambda x: generate_random_username(), axis=1)
    # Check the modifications
    print(df.head())
    # Save the cleaned dataset
    df.to_csv('cleaned_training_0-2000.csv', index=False)

# ========================================================= Cut Data ========================================================= #
def cut_data(file_path, new_file_path, num_entries):
    """
    Cut the first `num_entries` from the given CSV file and save to a new CSV file.
    """
    data = pd.read_csv(file_path)
    first_entries = data.head(num_entries)
    first_entries.to_csv(new_file_path, index=False)
    print(f"First {num_entries} entries saved to {new_file_path}")

def cut_data_range(file_path, new_file_path, start_row, end_row):
    """
    Cut a range of entries from `start_row` to `end_row` from the given CSV file and save to a new CSV file.
    """
    # Load data
    data = pd.read_csv(file_path)
    # Cut the data from start_row to end_row (end_row is exclusive)
    if end_row <= len(data):
        sliced_data = data.iloc[start_row:end_row]
        # Save the sliced data to a new CSV file
        sliced_data.to_csv(new_file_path, index=False)
        print(f"Entries from row {start_row} to {end_row-1} saved to {new_file_path}")
    else:
        print(f"Error: The end_row {end_row} exceeds the data length {len(data)}.")

# ========================================================= Convert CSV to JSON ========================================================= #
def csv_to_json(file_path, new_json_path, relevant_columns):
    """
    Convert the given CSV file to a JSON file with only relevant columns and an empty Label column.
    """
    data = pd.read_csv(file_path)
    data_relevant = data[relevant_columns].copy()
    data_relevant['Label'] = ''
    json_data = data_relevant.to_json(orient='records')
    with open(new_json_path, 'w') as json_file:
        json.dump(json.loads(json_data), json_file, indent=4)
    print(f"CSV file has been converted to JSON and saved as {new_json_path}")

# ========================================================= Read JSON Data ========================================================= #
def read_json_file(json_file_path, start, end):
    """
    Read and display a range of entries from a JSON file.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    #for i in range(start-1, end):
    for i in range(start, end):
        print(json.dumps(data[i], indent=4))

# ========================================================= Read CSV Data ========================================================= 
def read_csv_file(file_path, start, end):
    """
    Read and display a range of entries from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        if start < 0 or end > len(data):
            print("Specified range is out of bounds.")
        else:
            print(data.iloc[start:end])
    except Exception as e:
        print(f"An error occurred: {e}")

# ========================================================= Write Labeled JSON Data Back to CSV ========================================================= 
def write_labeled_json_to_csv(json_file_path, csv_output_path):
    """
    Write labeled JSON data back to a CSV file.
    """
    # Load the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    # Write DataFrame to CSV
    df.to_csv(csv_output_path, index=False)
    print(f"Labeled JSON data has been written to {csv_output_path}")