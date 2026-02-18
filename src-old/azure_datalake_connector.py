import os
import json
import hashlib
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import DeltaTable  # Ensure this matches your Delta Lake client import
import fsspec

class AzureDataLakeConnector:
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str, storage_account_name: str):
        """
        Initialize the connector with a configuration object.
        
        Parameters:
            subscription_id (str): Azure subscription ID.
            resource_group (str): Azure resource group name.
            workspace_name (str): Azure Machine Learning workspace name.
            storage_account_name (str): Azure Storage account name.
        """
        
        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._workspace_name = workspace_name
        self._storage_account_name = storage_account_name

    def generate_cache_path(self, container_name: str, path_in_container: str, columns, filter_expression, base_dir: str = "./data/cache"):
        """
        Generate a unique cache file path based on query parameters.
        
        This ensures that each unique set of query parameters gets its own cache file.
        """
        os.makedirs(base_dir, exist_ok=True)

        # Convert filter_expression to a string if it's not None
        if filter_expression is not None:
            filter_str = str(filter_expression)
        else:
            filter_str = None

        params = {
            "container_name": container_name,
            "path_in_container": path_in_container,
            "columns": columns,
            "filter_expression": filter_str
        }
        params_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.md5(params_str.encode("utf-8")).hexdigest()
        file_name = f"cache_{hash_val}.parquet"
        return os.path.join(base_dir, file_name)


    def _unify_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scan every column in `df`. For each column of dtype 'object', test whether all
        non-null values are purely numeric. 
        • If yes → coerce it to a numeric dtype (int64 or float64).
        • If no  → cast everything to str (so we avoid mixed types).
        Return a copy of `df` with those columns converted in-place.
        """
        df = df.copy()
        for col in df.columns:
            ser = df[col]
            # Only care about 'object' columns (i.e. potential mixed‐type)
            if ser.dtype == "object":
                # Attempt to turn every value into a number; invalid parses become NaN
                coerced = pd.to_numeric(ser, errors="coerce")
                
                # Build masks of “originally non-null” vs “successfully converted”
                mask_notnull = ~ser.isna()            # which rows were non-null to begin with
                mask_isnumber = ~coerced.isna()       # which rows turned into a valid number
                
                # If every non-null value stayed non-null after coercion → it's fully numeric
                if mask_isnumber[mask_notnull].all():
                    # Replace the column with the numeric version.
                    # (If they were all integers, pandas will pick int64; 
                    #  if any decimals exist, it falls back to float64.)
                    df[col] = coerced
                else:
                    # Otherwise, force everything to a pure string.  That way
                    # PyArrow will store it as a UTF-8 string, not try to cast to int.
                    df[col] = ser.astype(str)
        return df
    

    def read_data(self,
                    container_name: str,
                    path_in_container: str,
                    columns: list = None,
                    batch_size: int = 100_000,
                    filter_expression=None,
                    use_cache: bool = False,
                    storage_format: str = "delta"):
        """
        Reads data from Azure Data Lake and returns it as a pandas DataFrame.    
        Parameters:
        -----------
        container_name : str
            Name of the container in Azure Data Lake.
        path_in_container : str
            Path within the container. This can be a Delta table or a folder containing CSV files.
        columns : list, optional    
            List of column names to select. If None, all columns are loaded.
        batch_size : int, default=100_000
            Number of rows per chunk when iterating over each file.
        filter_expression : str or callable, optional
            If given as a string, it will be passed to `DataFrame.query(...)`.
            If given as a callable, it must accept a DataFrame and return a boolean mask (or filtered DataFrame).
            Example usages:
            * `df.query("column1 == 'some_value'")`
            * `df.query(lambda df: df.column1 == 'some_value')`
        use_cache : bool, default=False
            If True, looks for a previously‐cached Parquet under the “cache path” (generated by
            `self.generate_cache_path(...)`). If found, loads from there instead of re‐reading all files.
        storage_format : str, default="delta"
            The format of the data to read. Can be "delta" for Delta Lake tables or "csv" for CSV files.
        Returns:
        --------
        pandas.DataFrame
            A single DataFrame containing the concatenated (and optionally filtered) rows from all files.
        """
        if storage_format.lower() == "delta":
            return self.read_delta_table(
                container_name=container_name,
                path_in_container=path_in_container,
                columns=columns,
                batch_size=batch_size,
                filter_expression=filter_expression,
                use_cache=use_cache
            )
        elif storage_format.lower() == "csv":
            return self.read_csv(
                container_name=container_name,
                path_in_container=path_in_container,
                columns=columns,
                batch_size=batch_size,
                filter_expression=filter_expression,
                use_cache=use_cache
            )
        else:
            raise ValueError(f"Unsupported storage format: {storage_format}")
        

    def read_csv(self,
             container_name: str,
             path_in_container: str,
             columns: list = None,
             batch_size: int = 100_000,
             filter_expression=None,
             use_cache: bool = False):
        """
        Reads all CSV files under a given ADLS path and returns the data as a single pandas DataFrame.
        
        Parameters:
        -----------
        container_name : str
            Name of the container in Azure Data Lake.
        path_in_container : str
            Folder (prefix) within the container. All “.csv” files under this prefix will be read.
        columns : list, optional
            List of column names to select. If None, all columns are loaded.
        batch_size : int, default=100_000
            Number of rows per chunk when iterating over each CSV.
        filter_expression : str or callable, optional
            If given as a string, it will be passed to `DataFrame.query(...)`. 
            If given as a callable, it must accept a DataFrame and return a boolean mask (or filtered DataFrame).
        use_cache : bool, default=False
            If True, looks for a previously‐cached Parquet under the “cache path” (generated by
            `self.generate_cache_path(...)`). If found, loads from there instead of re‐reading all CSVs.
        
        Returns:
        --------
        pandas.DataFrame
            A single DataFrame containing the concatenated (and optionally filtered) rows from all CSV files.
        """
        # 1) Determine cache path
        cache_path = self.generate_cache_path(container_name, path_in_container, columns, filter_expression)

        # 2) If caching is turned on and a cache file exists, load & return it immediately
        if use_cache and cache_path and os.path.exists(cache_path):
            logging.info(f"Loading data from cache at {cache_path} ...")
            table = pq.read_table(cache_path, columns=columns)
            batches = list(table.to_batches())
            return pd.concat([batch.to_pandas() for batch in batches], ignore_index=True)

        # 3) Build the “abfs://” URL prefix for this folder
        storage_account = self._storage_account_name
        abfs_folder_prefix = (
            f"abfs://{container_name}@{storage_account}.dfs.core.windows.net/{path_in_container}"
        ).rstrip("/")

        # 4) Construct fsspec filesystem with Azure credentials
        storage_options = {
            "account_name": storage_account,
            "tenant_id": os.getenv("AZURE_TENANT_ID"),
            "client_id": os.getenv("AZURE_CLIENT_ID"),
            "client_secret": os.getenv("AZURE_CLIENT_SECRET"),
        }
        fs = fsspec.filesystem("abfs", **storage_options)

        # 5) Find every “.csv” under that prefix (recursively)
        glob_pattern = abfs_folder_prefix + "/**/*.csv"
        csv_paths = fs.glob(glob_pattern, detail=False)

        if not csv_paths:
            # No CSVs found; return an empty DataFrame with the requested columns (if any)
            return pd.DataFrame(columns=columns) if columns else pd.DataFrame()

        # 6) For each CSV path, open via fs.open(...) and read it in chunks
        dataframes: list[pd.DataFrame] = []
        for single_csv in sorted(csv_paths):
            try:
                # Instead of passing 'single_csv' + 'storage_options' into pd.read_csv directly,
                # we open a file-object with fs.open(...) and hand that to pandas. That avoids
                # the “storage_options passed with file object or non-fsspec file path” error.
                with fs.open(single_csv, mode="rb") as fobj:
                    read_kwargs = {
                        "low_memory": False,  # Disable low memory mode to avoid dtype inference issues
                    }
                    if columns:
                        read_kwargs["usecols"] = columns

                    reader = pd.read_csv(
                        fobj,
                        chunksize          = batch_size,
                        **read_kwargs
                    )
                    
                    for chunk in reader:
                        # 6a) Apply filter if provided
                        if filter_expression is not None:
                            if isinstance(filter_expression, str):
                                chunk = chunk.query(filter_expression)
                            elif callable(filter_expression):
                                mask_or_df = filter_expression(chunk)
                                if isinstance(mask_or_df, pd.DataFrame):
                                    chunk = mask_or_df
                                else:
                                    chunk = chunk[mask_or_df]
                            else:
                                raise ValueError(
                                    "filter_expression must be a pandas query‐string or a callable that returns a mask/DF"
                                )

                        # 6b) If after filtering the chunk is empty, skip
                        if chunk.shape[0] == 0:
                            continue

                        dataframes.append(chunk)

            except Exception as e:
                logging.error(f"Failed to open {single_csv}: {e}")
                raise

        # 7) Concatenate all filtered chunks
        if not dataframes:
            # Everything filtered out; return empty but with the correct columns
            return pd.DataFrame(columns=columns) if columns else pd.DataFrame()

        full_df = pd.concat(dataframes, ignore_index=True)

        # 8) If caching is requested, write a Parquet snapshot to cache_path
        if use_cache and cache_path:
            try:
                logging.info(f"Caching combined CSV result to {cache_path} ...")
                full_df = self._unify_dtypes(full_df)
                arrow_table = pa.Table.from_pandas(full_df)
                pq.write_table(arrow_table, cache_path)
            except Exception as e:
                logging.warning(f"Failed to write cache to {cache_path}: {e}")

        return full_df


    def read_delta_table(self,
              container_name: str,
              path_in_container: str,
              columns: list = None,
              batch_size: int = 100_000,
              filter_expression=None,
              use_cache: bool = False):
        """
        Reads a Delta table and returns the data as a single pandas DataFrame.
        
        Parameters:
        container_name (str): Name of the container in Azure Data Lake.
        path_in_container (str): Path within the container.
        columns (list): List of columns to select (optional).
        batch_size (int): Number of rows per chunk.
        filter_expression: Filter expression to reduce the data read.
        use_cache (bool): If True, caching is enabled (recommended for development).
        
        Returns:
        A single pandas DataFrame containing all the data.
        """
        # Generate a unique cache path if caching is enabled and none was provided.
        cache_path = self.generate_cache_path(container_name, path_in_container, columns, filter_expression)
        
        # If caching is enabled and cache exists, load from cache.
        if use_cache and cache_path and os.path.exists(cache_path):
            logging.info(f"Loading data from cache at {cache_path} ...")
            table = pq.read_table(cache_path, columns=columns)
            batches = list(table.to_batches())
            return pd.concat([batch.to_pandas() for batch in batches], ignore_index=True)

        # Build the ABFSS path.
        storage_account = self._storage_account_name
        abfss_path = f"abfss://{container_name}@{storage_account}.dfs.core.windows.net/{path_in_container}"

        # Instantiate the DeltaTable with the required Azure credentials.
        dt = DeltaTable(
            abfss_path,
            storage_options={
                "AZURE_CLIENT_ID": os.getenv("AZURE_CLIENT_ID"),
                "AZURE_CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET"),
                "AZURE_TENANT_ID": os.getenv("AZURE_TENANT_ID"),
                "AZURE_STORAGE_ACCOUNT_NAME": storage_account
            }
        )

        # Convert the Delta table to a PyArrow dataset.
        arrow_dataset = dt.to_pyarrow_dataset()

        # Build a scanner with the optional filter.
        scanner = arrow_dataset.scanner(
            columns=columns,
            filter=filter_expression,
            batch_size=batch_size  # if your version supports this, otherwise remove
        )

        # Collect record batches.
        record_batches = list(scanner.to_batches())
        
        # If caching is enabled, cache the result.
        if use_cache and cache_path:
            logging.info(f"Caching filtered result to {cache_path} ...")
            table = pa.Table.from_batches(record_batches)
            pq.write_table(table, cache_path)
            table = pq.read_table(cache_path, columns=columns)
            record_batches = list(table.to_batches())
        
        # Concatenate all batches into a single DataFrame.
        print(cache_path)
        return pd.concat([record_batch.to_pandas() for record_batch in record_batches], ignore_index=True)



    def write_data(self, container_name: str, path_in_container: str, data, storage_format="parquet", **kwargs):
        """
        Writes data to the Azure Data Lake.
        
        Parameters:
          container_name (str): Container name.
          path_in_container (str): Destination path within the container.
          data: Data to write (pandas DataFrame or pyarrow Table).
          storage_format (str): File format for storage (default is "parquet").
          kwargs: Additional arguments for writing.
        
        Note:
          This is a basic implementation that writes data locally and uploads it.
          It may be extended to support other patterns.
        """
        storage_account = self.azure_config.storage_account_name
        abfss_path = f"abfss://{container_name}@{storage_account}.dfs.core.windows.net/{path_in_container}"

        # Convert data to a pyarrow Table if necessary.
        if hasattr(data, "to_parquet"):
            table = pa.Table.from_pandas(data)
        elif isinstance(data, pa.Table):
            table = data
        else:
            raise ValueError("Unsupported data type. Expected a pandas DataFrame or a pyarrow Table.")

        # Write table to a temporary local file.
        temp_file = "temp_data.parquet"
        pq.write_table(table, temp_file)

        # Upload the file using the Data Lake Service Client.
        account_url = f"https://{storage_account}.dfs.core.windows.net"
        # Use your secure authentication method here.
        from azure.storage.filedatalake import DataLakeServiceClient
        service_client = DataLakeServiceClient(
            account_url=account_url,
            credential=os.getenv("AZURE_CLIENT_SECRET")  # Replace with a secure method in production.
        )
        file_system_client = service_client.get_file_system_client(file_system=container_name)
        file_client = file_system_client.get_file_client(path_in_container)
        with open(temp_file, "rb") as f:
            file_client.upload_data(f, overwrite=True)
        os.remove(temp_file)
        logging.info(f"Data successfully written to {abfss_path}.")


if __name__ == "__main__":
    # Example usage
    connector = AzureDataLakeConnector(
        subscription_id="your_subscription_id",
        resource_group="your_resource_group",
        workspace_name="your_workspace_name",
        storage_account_name="your_storage_account_name"
    )

    # Read data
    df = connector.read_data(
        container_name="your_container",
        path_in_container="path/to/your/delta_table",
        columns=["column1", "column2"],
        filter_expression=None,
        use_cache=True
    )
    print(df.head())
