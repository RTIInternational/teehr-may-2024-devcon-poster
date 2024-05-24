# May 2024 CIROH DevCon Benchmark Poster Materials
This repo contains a collection of notebooks supporting the benchmark analysis poster presented at the 2024 CIROH Developer's Conference.

Each of the 12 notebooks was used to evaluate DuckDB, Dask, and Xarray-Zarr for calculating a set of performance metrics from a 3-year NWM retrospective test dataset.

An additional test included in the poster involving Spark can be found in a separate repo: https://github.com/RTIInternational/teehr-spark-iceberg

Notebooks:

* **01_teehr_local_ind_files**: DuckDB querying local un-joined parquet files.

* **02_teehr_local_joined**: DuckDB querying local joined parquet files.

* **03_teehr_local_db**: DuckDB querying local database (joined timeseries)

* **04_teehr_s3_ind_files**: DuckDB querying un-joined parquet files in S3.

* **05_teehr_s3_joined**: DuckDB querying joined parquet files in S3.

* **06_teehr_s3_db**: DuckDB querying database (joined timeseries) in S3.

* **07_dask_local_ind_files**: Using Dask DataFrames with local un-joined parquet files.

* **08_dask_local_joined**: Using Dask DataFrames with local joined parquet files.

* **09_dask_s3_joined**: Using Dask DataFrames with joined parquet files in S3.

* **10_dask_duckdb_local**: Attempting to combine Dask with DuckDB (unsuccessfully).

* **11_zarr_local_joined**: Using Xarray with local Zarr stores of joined timeseries.

* **12_zarr_s3_joined**: Using Xarray with local Zarr stores.