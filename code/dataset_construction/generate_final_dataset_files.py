import pandas as pd
import sqlite3
import os
import code.utils as utils

# --- Configuration ---
WRITE_FLAT_FILE = False
WRITE_STAR_SCHEMA_FILES = False
REBUILD_DB = False

if not (WRITE_FLAT_FILE or WRITE_STAR_SCHEMA_FILES or REBUILD_DB):
    raise ValueError("At least one of WRITE_FLAT_FILE, WRITE_STAR_SCHEMA_FILES, or REBUILD_DB must be True")

# --- Directory setup ---
INPUT_FILE = os.path.join(utils.get_data_dir(step=6), "single_table", "mental_health_apps.tsv")
OUTPUT_DIR = utils.get_out_dir()
SQL_FILE = os.path.join(OUTPUT_DIR, "mhap.db")
FLAT_FILE = os.path.join(OUTPUT_DIR, "mental_health_apps_wide_format.tsv")
STAR_DIR = os.path.join(OUTPUT_DIR, "star_schema")
os.makedirs(STAR_DIR, exist_ok=True)

def split_and_join_multilabels(df, column_name, suffix="", drop_original=False):
    df_split = df[column_name].str.get_dummies(sep=", ").astype(bool)
    df = df.join(df_split, rsuffix=suffix)
    if drop_original:
        df.drop(columns=[column_name], inplace=True)
    return df

def create_bridge_and_dim_tables(df, column_name):

    dimname = column_name[:-1]
    dim_id = f"{dimname}_id"

    bridge = (
        df[column_name]
        .str.split(", ")
        .apply(pd.Series, 1)
        .stack()
        .reset_index(drop=False)
    )

    bridge.drop(columns="level_1", inplace=True)
    bridge.columns = ["app_id", dimname]

    all_values = list(set(bridge[dimname]))
    all_values.sort()

    dim = pd.DataFrame(all_values)
    dim[dim_id] = range(1, len(all_values) + 1)
    dim.columns = [dimname, dim_id]
    dim.set_index(dim_id, inplace=True, drop=False)

    bridge = bridge.merge(dim, on=dimname, how="left")
    bridge.drop(columns=dimname, inplace=True)
    bridge.columns = ["app_id", dim_id]

    return bridge, dim


def build_sqlite_db(data, db_filename=SQL_FILE):

    sql_conn = sqlite3.connect(db_filename)

    # Create a cursor object
    cursor = sql_conn.cursor()

    create_features_table = """
    CREATE TABLE IF NOT EXISTS "features" (
    "feature" TEXT NOT NULL,
    "feature_id" INTEGER PRIMARY KEY
    );
    """

    create_indications_table = """
    CREATE TABLE IF NOT EXISTS "indications" (
    "indication" TEXT NOT NULL,
    "indication_id" INTEGER PRIMARY KEY
    );
    """

    create_demographics_table = """
    CREATE TABLE IF NOT EXISTS "demographics" (
    "demographic" TEXT NOT NULL,
    "demographic_id" INTEGER PRIMARY KEY
    );
    """

    create_categories_table = """
    CREATE TABLE IF NOT EXISTS "categories" (
    "category" TEXT NOT NULL,
    "category_id" INTEGER PRIMARY KEY
    );
    """

    create_apps_table = """
    CREATE TABLE IF NOT EXISTS "apps" (
    "app_id" INTEGER PRIMARY KEY,
    "name" TEXT NOT NULL,
    "rating" REAL,
    "company" TEXT,
    "url" TEXT,
    "description" TEXT NOT NULL,
    "downloads" INTEGER NOT NULL,
    "updated" DATE,
    "category_id" INTEGER REFERENCES categories(category_id)
    );
    """

    create_app_features_table = """
    CREATE TABLE IF NOT EXISTS "app_features" (
    "app_id" INTEGER REFERENCES apps(app_id),
    "feature_id" INTEGER REFERENCES features(feature_id)
    );
    """

    create_app_indications_table = """
    CREATE TABLE IF NOT EXISTS "app_indications" (
    "app_id" INTEGER REFERENCES apps(app_id),
    "indication_id" INTEGER REFERENCES indications(indication_id)
    );
    """

    create_app_demographics_table = """
    CREATE TABLE IF NOT EXISTS "app_demographics" (
    "app_id" INTEGER REFERENCES apps(app_id),
    "demographic_id" INTEGER REFERENCES demographics(demographic_id)
    );
    """

    cursor.execute(create_features_table)
    cursor.execute(create_indications_table)
    cursor.execute(create_demographics_table)
    cursor.execute(create_categories_table)
    cursor.execute(create_apps_table)
    cursor.execute(create_app_features_table)
    cursor.execute(create_app_indications_table)
    cursor.execute(create_app_demographics_table)

    # Commit the changes
    sql_conn.commit()

    for data_name, df in data.items():
        print(f"Inserting {data_name} data...")
        df.to_sql(data_name, sql_conn, if_exists="replace", index=False)

    sql_conn.close()


if __name__ == "__main__":

    apps = pd.read_csv(INPUT_FILE, sep="\t")

    apps.columns = apps.columns.str.lower()

    # Create a single-table version just flattening the multi-label values into boolean columns
    if WRITE_FLAT_FILE:
        print("Writing flat file...")
        apps_wide = apps.copy(deep=True)
        apps_wide = split_and_join_multilabels(apps_wide, "features", suffix="_feature")
        apps_wide = split_and_join_multilabels(apps_wide, "indications", suffix="_indication")
        apps_wide = split_and_join_multilabels(apps_wide, "demographics", suffix="_demographic")
        apps_wide.to_csv(FLAT_FILE, sep="\t", header=True, index=False)
    else:
        print("Not writing flat file")

    # --- Create star schema tables --- 
    apps.index = apps["app_id"]

    data = dict()

    data["app_features"], data["features"] = create_bridge_and_dim_tables(apps, "features")
    data["app_indications"], data["indications"] = create_bridge_and_dim_tables(apps, "indications")
    data["app_demographics"], data["demographics"] = create_bridge_and_dim_tables(apps, "demographics")

    # Create category dim table
    all_categories = list(set(apps["category"]))
    all_categories.sort()

    category_dim = pd.DataFrame(all_categories)
    category_dim["category_id"] = range(1, len(all_categories) + 1)
    category_dim.rename(columns={0: "category"}, inplace=True)
    category_dim.set_index("category_id", inplace=True, drop=False)
    data["categories"] = category_dim

    # Add reference to category_id in main apps table
    apps = apps.merge(category_dim, on="category", how="left")
    apps.drop(columns=["features", "demographics", "indications", "category"], inplace=True,)

    data["apps"] = apps

    if WRITE_STAR_SCHEMA_FILES:
        print("Writing star schema files...")
        for name, df in data.items():
            print(f"{name}: {df.shape}")
            OUTFILE = os.path.join(STAR_DIR, f"{name}.tsv")
            df.to_csv(OUTFILE, sep="\t", index=False)
    else:
        print("Not writing star schema files")

    if REBUILD_DB:
        print("Building new SQLite database...")
        build_sqlite_db(data, db_filename=SQL_FILE)
    else:
        print("Not building new SQLite database")
