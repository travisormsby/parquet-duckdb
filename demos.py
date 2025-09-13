# %%
# Import modules
import csv
from difflib import Differ
from functools import partial
from glob import glob
import json
import os
from random import choice, randint, uniform

import duckdb
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq


# Create parameter variables
MB_PER_BYTE = 1_000_000
NUM_ROWS = 1_000_000
NUM_COLS = 200
DATA_DIR = "data"

# Derived variables
csv_file = os.path.join(DATA_DIR, "data.csv")
parquet_file = os.path.join(DATA_DIR, "data.parquet")
hive_dir = os.path.join(DATA_DIR, "by_year")
sorted_parquet_file = os.path.join(DATA_DIR, "sorted.parquet")
per_thread_dir = os.path.join(DATA_DIR, "per_thread")
profile_output = os.path.join(DATA_DIR, "duckdb_profile.json")
partitioned_glob = os.path.join(hive_dir, "*/*.parquet")
per_thread_glob = os.path.join(per_thread_dir, "*.parquet")

# Install and load the DuckDB spatial extension
duckdb.install_extension("spatial")
duckdb.load_extension("spatial")

# Set DuckDB profiling options
duckdb.sql("PRAGMA enable_profiling='json'")
duckdb.sql(f"PRAGMA profiling_output='{profile_output}'")


# Create helper functions
def create_rows(num_cols, num_rows):
    """
    Generate rows with random geometry, month, year, and random integer fields.

    Args:
        num_cols (int): Number of random integer columns to generate per row.
        num_rows (int): Number of rows to generate.

    Yields:
        list: A list representing a single row: [ID, geom, month, year, ...random ints...].
    """
    for i in range(num_rows):
        lon = uniform(-180, 180)
        lat = uniform(-90, 90)
        geom = f"POINT({lon:.5f} {lat:.5f})"
        month = choice(range(1, 13))
        year = choice(range(2020, 2025))
        row = [i, geom, month, year]
        row.extend(randint(0, 100) for _ in range(num_cols))
        yield row


def create_parquet_file_info(path):
    """
    Return a pandas DataFrame with the file path, size, number of rows, and number of columns
    for a given Parquet file or all Parquet files in a given directory.

    Args:
        path (str): Path to a Parquet file or a directory containing Parquet files.

    Returns:
        pandas.DataFrame: DataFrame with columns for file path, size, number of rows, and number of columns.
    """
    file_info_dfs = []
    if os.path.isdir(path):
        pattern = os.path.join(path, "**")
    else:
        pattern = path
    for f in glob(pattern, recursive=True):
        if os.path.isfile(f):
            pf = pq.ParquetFile(f)
            num_cols = len(pf.schema.names)
            num_rows = pf.metadata.num_rows
            df = create_file_info(f, num_cols, num_rows)
            file_info_dfs.append(df)
    return pd.concat(file_info_dfs)


def create_file_info(path, num_cols, num_rows):
    """
    Create a pandas DataFrame with file metadata.

    Args:
        path (str): The file path.
        num_cols (int): Number of columns in the file.
        num_rows (int): Number of rows in the file.

    Returns:
        pandas.DataFrame: A DataFrame with fields for the file path, file size, number of columns, and number of rows.
    """
    size = f"{os.path.getsize(path) / MB_PER_BYTE:.2f} MB"
    file_info = {
        "path": path,
        "size": size,
        "columns": num_cols,
        "rows": f"{num_rows:,}",
    }
    return pd.DataFrame([file_info])


def create_query(table, fields=("field_0",), category=None, where_clause=None):
    """
    Programmatically construct a SQL query string.

    Args:
        table (str): The table or file to query.
        fields (str or Sequence[str], optional): Field name or collection of field names to return.
            Defaults to ("field_0",).
        category (str, optional): Field name to group by. If provided, the query will sum the columns
            specified in `fields` and group by this category.
        where_clause (str, optional): WHERE clause of the query.

    Returns:
        str: A SQL query string suitable for passing to a query engine.
    """

    if isinstance(fields, str):
        select_fields = fields
    elif category:
        select_fields = f"{category}, " + ", ".join(f"sum({field})" for field in fields)
    else:
        select_fields = ", ".join(fields)

    query_lines = [f"SELECT {select_fields}", f"FROM '{table}'"]
    if where_clause:
        query_lines.append(f"WHERE {where_clause}")
    if category:
        query_lines.append(f"GROUP BY {category}")
        query_lines.append(f"ORDER BY {category}")

    return "\n".join(query_lines)


def report_query_time(query):
    """
    Execute a DuckDB SQL query with profiling enabled and return selected performance metrics.

    Args:
        query (str): The SQL query to execute.

    Returns:
        pandas.DataFrame: A DataFrame indexed by metric name, containing values for latency, CPU time,
            peak buffer memory, and cumulative rows scanned.
    """
    row_indices = ["latency", "cpu_time", "cumulative_rows_scanned"]

    df = pd.DataFrame(index=row_indices)
    duckdb.sql(query).fetchall()
    with open(profile_output) as f:
        profile = json.load(f)
        profile_data = {k: profile[k] for k in row_indices}
        df = pd.DataFrame.from_dict(profile_data, orient="index")
    return df


def name_queries(queries):
    """
    Generate short names for a pair of SQL queries by diffing their text.

    Args:
        queries (Sequence[str]): A sequence of two SQL query strings.

    Returns:
        list[str]: A list containing up to two short query name strings, extracted from the diff.
    """
    query_lines = (query.splitlines() for query in queries)
    query_diff = Differ().compare(*query_lines)
    query_names = [line[2:] for line in query_diff if line.startswith(("-", "+"))][:2]
    return query_names


def compare_reports(dfs, data_cols):
    """
    Combine multiple DataFrames of query performance metrics and create a Matplotlib chart comparing metrics.

    Args:
        dfs (tuple of pandas.DataFrame): A tuple containing DataFrames with matching indices.
        data_cols (tuple of str): Names to assign to the sets of metrics (e.g., query names).

    Returns:
        pandas.DataFrame: A DataFrame with columns for each input, a 'change' column showing percent difference,
            and the original indices as row labels.
    """
    compare_df = pd.concat(dfs, axis=1)
    compare_df.columns = data_cols
    return compare_df


def chart_report_comparison(df):
    latency_cpu = df.loc[["latency", "cpu_time"]]
    cumulative_rows = df.loc[["cumulative_rows_scanned"]]

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]}
    )

    # Clustered bar plot for latency and cpu_time
    latency_cpu.plot.bar(ax=ax1, rot=0)
    ax1.set_ylabel("Time (s)")

    # Horizontal bar plot for cumulative rows scanned
    cumulative_rows.plot.barh(ax=ax2)
    ax2.set_xlabel("Cumulative Rows Scanned")
    ax2.set_yticklabels([])
    ax2.legend().remove()
    ax2.invert_yaxis()


def show_query_info(*queries, display_table=False):
    """
    Display SQL query text and, if two queries are provided, compare their performance metrics.

    Args:
        *queries (str): One or more SQL query strings to display and/or compare.
        display_table (bool, optional): If True, print the DuckDB result table for the last query. Default is False.

    Returns:
        None
    """
    print(*queries, sep="\n--------\n")
    if display_table:
        print(duckdb.sql(queries[-1]))
    if len(queries) > 1:
        time_dfs = [report_query_time(query) for query in queries]
        report_df = compare_reports(time_dfs, name_queries(queries))
        chart_report_comparison(report_df)


# %%
# Create a large random spatial dataset in CSV format
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    header = ["ID", "wkt", "month", "year"] + [f"field_{i}" for i in range(NUM_COLS)]
    writer.writerow(header)
    data = create_rows(NUM_COLS, NUM_ROWS)
    writer.writerows(data)

csv_df = create_file_info(csv_file, len(header), NUM_ROWS)
display(csv_df)

# %%
# DuckDB can read CSV files into a relation
rel = duckdb.read_csv(csv_file)
rel.limit(5)

# %%
# DuckDB ST_GeomFromText function reads WKT as geometry
spatial_rel = rel.project("ST_GeomFromText(wkt) as geom, *")
spatial_rel.limit(5)


# %%
# DuckDB can export a relation with spatial types to GeoParquet
spatial_rel.to_parquet(parquet_file, overwrite=True)
parquet_df = create_parquet_file_info(parquet_file)
pd.concat([csv_df, parquet_df])


# %%
# DuckDB can read GeoParquet into a relation
pq_rel = duckdb.read_parquet(parquet_file)
pq_rel.limit(5)

# %%
# Parquet files can be hive partitioned by column value
pq_rel.to_parquet(hive_dir, overwrite=True, partition_by=["year"])

partitioned_files_df = create_parquet_file_info(hive_dir)
display(partitioned_files_df)

# %%
# Hive partitioned files don't contain the partitioned by column
partitioned_file = partitioned_files_df["path"].iloc[0]
partition_rel = duckdb.read_parquet(partitioned_file)
partition_rel.project("year")

# %%
# DuckDB can use globbing to query multiple files
glob_query = create_query(partitioned_glob, category="year")
show_query_info(glob_query, display_table=True)


# %%
# Querying partitioned parquet files can be faster
year2020 = partial(create_query, where_clause="year = 2020")
partitioned_query = year2020(partitioned_glob)
unpartitioned_query = year2020(parquet_file)
show_query_info(unpartitioned_query, partitioned_query)

# %%
# Partitioning benefits are less when the scheme isn't relevant to queries
january = partial(create_query, where_clause="month = 1")
unpartitioned_query_by_month = january(parquet_file)
partitioned_query_by_month = january(partitioned_glob)

show_query_info(unpartitioned_query_by_month, partitioned_query_by_month)


# %%
# Spatial queries on parquet files are faster
wkt_within = """ST_Within(
    ST_GeomFromText(wkt), 
    ST_GeomFromText(
        'POLYGON ((-90 45, -95 45, -95 40, -90 40, -90 45))'
    ) 
)"""

geom_within = wkt_within.replace("ST_GeomFromText(wkt)", "geom")

csv_spatial_query = create_query(csv_file, where_clause=wkt_within)
parquet_spatial_query = create_query(parquet_file, where_clause=geom_within)
show_query_info(csv_spatial_query, parquet_spatial_query)


# %%
# Non-spatial queries on parquet files are much faster

csv_query = create_query(csv_file)
parquet_query = create_query(parquet_file)
show_query_info(csv_query, parquet_query)


# %%
# Advantages to parquet are less when excluding fewer columns
csv_query_all = create_query(csv_file, fields="*")
parquet_query_all = create_query(parquet_file, fields="*")
show_query_info(csv_query_all, parquet_query_all)

# %%
# Parquet files are organized into row groups
data_file = pq.ParquetFile(parquet_file)
print(f"Metadata for {parquet_file}:")
data_file.metadata

# %%
# Each row group contains a subset of rows
row_group = 0
row_group_metadata = data_file.metadata.row_group(row_group)
print(f"Metadata for row group {row_group}:")
row_group_metadata

# %%
# Row groups have metadata for each column chunk
geom_col = 0
print(f"Metadata for column chunk {geom_col} of row group {row_group}:")
row_group_metadata.column(geom_col)

# %%
# Different data types have different encodings
month_col = 3
print(f"Metadata for column chunk {month_col} of row group {row_group}:")
row_group_metadata.column(month_col)

# %%
# DuckDB can write sorted data to disk
duckdb.sql(f"""
    COPY
        (
            SELECT *
            FROM '{parquet_file}'
            ORDER BY month
        )
        TO '{sorted_parquet_file}'
        (FORMAT parquet)
""")


# %%
# DuckDB does not specify the sorting_columns
sorted_data_file = pq.ParquetFile(sorted_parquet_file)
print(f"Metadata for {sorted_parquet_file}:")
sorted_data_file.metadata.row_group(row_group)

# %%
# But column chunk metadata has correct min and max
print(f"Metadata for column chunk {month_col} of row group {row_group}:")
sorted_data_file.metadata.row_group(row_group).column(month_col)

# %%
# Queries are more efficient because they only read some column chunks
quarter1 = partial(create_query, category="month", where_clause="month in [1, 2, 3]")
unsorted_query = quarter1(parquet_file)
sorted_query = quarter1(sorted_parquet_file)
show_query_info(unsorted_query, sorted_query)

# %%
# By default, DuckDB uses all the machine's threads
first_7_fields_by_month = partial(
    create_query,
    fields=[f"field_{i}" for i in range(7)],
    category="month",
    where_clause=geom_within,
)
max_threads_query = first_7_fields_by_month(partitioned_glob)
max_threads_df = report_query_time(max_threads_query)

duckdb.sql("""SELECT current_setting('threads')""")

# %%
# Specifying a lower number of threads slows down queries, but can be more efficient
duckdb.sql("PRAGMA threads=1")
print(duckdb.sql("""SELECT current_setting('threads')"""))

single_thread_df = report_query_time(max_threads_query)
thread_compare_df = compare_reports(
    (max_threads_df, single_thread_df), ("max threads", "1 thread")
)
print(max_threads_query)
chart_report_comparison(thread_compare_df)
duckdb.sql(f"PRAGMA threads={os.cpu_count()}")

# %%
# DuckDB can partition data by thread

print(duckdb.sql("SELECT current_setting('threads')"))
spatial_rel.to_parquet(per_thread_dir, per_thread_output=True, overwrite=True)
create_parquet_file_info(per_thread_dir)


# %%
# Matching the number of files to threads can make queries faster
first_7_fields_sum = partial(create_query, fields=[f"sum(field_{i})" for i in range(7)])
unoptimized_threads_query = first_7_fields_sum(partitioned_glob)
optimized_threads_query = first_7_fields_sum(per_thread_glob)
show_query_info(unoptimized_threads_query, optimized_threads_query)
