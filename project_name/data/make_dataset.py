import sqlite3
import re
import pandas as pd

def get_data_from_database(database_dir: str, table: str):
    """
    Function that extract data from database

    Args:
        database_dir (string): directory to the database containing the raw data
        table (string): The table to extract data from in the database

    Returns:
        table (DataFrame): DataFrame containing data from the selected table in the database
    """

    conn = sqlite3.connect(database_dir)
    query = f'SELECT * FROM {table}'

    table = pd.read_sql_query(query, conn)
    conn.close()

    return table

def text_preprocessing(text):
    """
    Takes a string representation of a list of words as input,
    removes any special characters from the words, and then removes any words that contain numbers.

    Args:
        text: A series of words.

    Returns:
        The function `text_preprocessing` returns a list of words without any special characters or
        newline command`\n`.
    """
    # Remove newline command
    text = ''.join(text.splitlines())
    # Remove Non-English words
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  
    return text

def processing():
    
    table = get_data_from_database("data/raw/wikibooks.sqlite", "en")
    table["body_text"] = table["body_text"].apply(text_preprocessing)
    table["body_text"].to_csv('data/processed/data.csv', index=False)
    
if __name__ == '__main__':
    # Get the data and process it
    processing()
    pass