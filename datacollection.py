import sqlite3
import pandas as pd

f1DB_path = '/Users/aarnavkoushik/Documents/GitHub/f1-timing-database/F1_timingdata_2014_2019.sqlite'

conn = sqlite3.connect(f1DB_path)

#creating dictionary
def load_sqlite_db_to_dfs(conn):
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query, conn)
    print(tables)
    dataframes = {}
    
    for table_name in tables['name']:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        dataframes[table_name] = df
        print(f"Loaded table {table_name} with {df.shape[0]} rows and {df.shape[1]} columns")
    
    return dataframes

# Loading
dataframes = load_sqlite_db_to_dfs(conn)
print(dataframes)
# testing
example_df = dataframes['laps']  # replace 'example_table_name' with an actual table name
print(example_df.head())

# Close the connection
conn.close()
