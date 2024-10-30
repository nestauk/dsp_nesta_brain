import io
import json
import os
import pickle  # nosec
import warnings

from fnmatch import fnmatch
from pathlib import Path
from typing import List

import boto3
import dotenv
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from botocore.client import BaseClient


dotenv.load_dotenv()

# Retrieve AWS file information from environment variables
BUCKET_NAME_RAW = os.getenv("BUCKET_NAME_RAW")
# Retrieve AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")


def s3_client(aws_access_key_id: str = AWS_ACCESS_KEY, aws_secret_access_key: str = AWS_SECRET_KEY) -> BaseClient:
    """Initialize and return an S3 client"""
    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def _get_bucket_filenames(bucket_name: str, dir_name: str = "") -> List[str]:
    """Get a list of all files in bucket directory.

    Taken from Nesta DS Utils.

    Args:
        bucket_name (str): S3 bucket name
        dir_name (str, optional): Directory or sub-directory within an S3 bucket.
            Defaults to '' (the top level bucket directory).

    Returns:
        List[str]: List of file names in bucket directory.
    """
    s3_resources = boto3.resource("s3")
    bucket = s3_resources.Bucket(bucket_name)
    return [object_summary.key for object_summary in bucket.objects.filter(Prefix=dir_name)]


def _match_one_file(all_files: list, file_name: str, dir_path: str) -> str:
    matched_files = [f for f in all_files if Path(f).stem == file_name]

    if not matched_files:
        raise FileNotFoundError(f"No file exactly matching '{file_name}' found in {dir_path}")
    elif len(matched_files) > 1:
        raise ValueError(
            f"Multiple files exactly matching '{file_name}' found in {dir_path}, requiring clarification."
        )

    return matched_files[0]


def _list_directories(s3_client: BaseClient, bucket: str, prefix: str) -> list:
    """List S3 directories under the specified prefix."""
    # List all objects with the specified prefix
    objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")

    # Extract the directory names from the CommonPrefixes list
    directories = [prefix["Prefix"] for prefix in objects.get("CommonPrefixes", [])]

    return directories


def _fileobj_to_df(fileobj: io.BytesIO, path_from: str, **kwargs) -> pd.DataFrame:
    """Convert bytes file object into dataframe.

    Args:
        fileobj (io.BytesIO): Bytes file object.
        path_from (str): Path of loaded data.

    Returns:
        pd.DataFrame: Data as dataframe.
    """
    if fnmatch(path_from, "*.csv"):
        return pd.read_csv(fileobj, **kwargs)
    elif fnmatch(path_from, "*.parquet"):
        return pd.read_parquet(fileobj, **kwargs)


def _fileobj_to_dict(fileobj: io.BytesIO, **kwargs) -> dict:
    """Convert bytes file object into dictionary.

    Args:
        fileobj (io.BytesIO): Bytes file object.

    Returns:
        dict: Data as dictionary.
    """
    return json.loads(fileobj.getvalue().decode(), **kwargs)


def _fileobj_to_list(fileobj: io.BytesIO, path_from: str, **kwargs) -> list:
    """Convert bytes file object into list.

    Args:
        fileobj (io.BytesIO): Bytes file object.
        path_from (str): Path of loaded data.

    Returns:
        list: Data as list.
    """
    if fnmatch(path_from, "*.csv"):
        list_data = pd.read_csv(fileobj, **kwargs)["0"].to_list()
    elif fnmatch(path_from, "*.txt"):
        list_data = fileobj.read().decode().splitlines()
    elif fnmatch(path_from, "*.json"):
        list_data = json.loads(fileobj.getvalue().decode())
    elif fnmatch(path_from, "*.jsonl"):
        list_data = [json.loads(line) for line in fileobj]

    return list_data


def _fileobj_to_str(fileobj: io.BytesIO) -> str:
    """Convert bytes file object into string.

    Args:
        fileobj (io.BytesIO): Bytes file object.
        path_from (str): Path of loaded data.

    Returns:
        str: Data as string.
    """
    return fileobj.getvalue().decode("utf-8")


def _fileobj_to_np_array(fileobj: io.BytesIO, path_from: str, **kwargs) -> np.ndarray:
    """Convert bytes file object into numpy array.

    Args:
        fileobj (io.BytesIO): Bytes file object.
        path_from (str): Path of loaded data.

    Returns:
        np.ndarray: Data as numpy array.
    """
    if fnmatch(path_from, "*.csv"):
        np_array_data = np.genfromtxt(fileobj, delimiter=",", **kwargs)

    return np_array_data


def _download_obj(
    s3_client: BaseClient,
    bucket: str,
    path_from: str,
    download_as: str = None,
    kwargs_boto: dict = None,
    kwargs_reading: dict = None,
) -> any:  # noqa: B006
    """Download data to memory from S3 location.

    Adapted from Nesta DS Utils

    Args:
        s3_client (BaseClient): The S3 client.
        bucket (str): Bucket's name.
        path_from (str): Path to data in S3.
        download_as (str, optional): Type of object downloading. Choose between
        ('dataframe', 'geodf', 'dict', 'list', 'str', 'np.array'). Not needed for 'pkl files'.
        kwargs_boto (dict, optional): Dictionary of kwargs for boto3 function 'download_fileobj'.
            Default is None, which results in an empty dict.
        kwargs_reading (dict, optional): Dictionary of kwargs for reading data.
            Default is None, which results in an empty dict.

    Returns:
        any: Downloaded data.
    """
    if kwargs_boto is None:
        kwargs_boto = {}
    if kwargs_reading is None:
        kwargs_reading = {}

    if not path_from.endswith(
        (
            ".csv",
            ".parquet",
            ".json",
            ".jsonl",
            ".txt",
            ".pkl",
            ".geojson",
            ".xlsx",
            ".xlsm",
        )
    ):
        raise NotImplementedError("This file type is not currently supported for download in memory.")
    fileobj = io.BytesIO()
    s3_client.download_fileobj(bucket, path_from, fileobj, **kwargs_boto)
    fileobj.seek(0)
    if not download_as:
        if path_from.endswith((".pkl",)):
            return pickle.load(fileobj, **kwargs_reading)  # nosec
        else:
            raise ValueError("'download_as' is required for this file type.")
    elif download_as == "dataframe":
        if path_from.endswith((".csv", ".parquet", ".xlsx", ".xlsm")):
            return _fileobj_to_df(fileobj, path_from, **kwargs_reading)
        else:
            raise NotImplementedError(
                "Download as dataframe currently supported only " "for 'csv','parquet','xlsx' and 'xlsm'."
            )
    elif download_as == "dict":
        if path_from.endswith((".json",)):
            return _fileobj_to_dict(fileobj, **kwargs_reading)
        elif path_from.endswith((".geojson",)):
            warnings.warn(
                "Please check geojson has a member with the name 'type', the value of the member must be one of the following:"
                "'Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon', 'GeometryCollection',"
                "'Feature' and 'FeatureCollection'. Else downloaded dictionary will not be valid geojson."
            )
            return _fileobj_to_dict(fileobj, **kwargs_reading)
        else:
            raise NotImplementedError("Download as dictionary currently supported only " "for 'json' and 'geojson'.")
    elif download_as == "list":
        if path_from.endswith((".csv", ".txt", ".json", ".jsonl")):
            return _fileobj_to_list(fileobj, path_from, **kwargs_reading)
        else:
            raise NotImplementedError("Download as list currently supported only " "for 'csv', 'txt' and 'json'.")
    elif download_as == "str":
        if path_from.endswith((".txt",)):
            return _fileobj_to_str(fileobj)
        else:
            raise NotImplementedError("Download as string currently supported only " "for 'txt'.")
    else:
        raise ValueError(
            "'download_as' not provided. Choose between ('dataframe', 'geodf', "
            "'dict', 'list', 'str', 'np.array'). Not needed for 'pkl files'.'"
        )


def _df_to_fileobj(df_data: pd.DataFrame, path_to: str, **kwargs) -> io.BytesIO:
    """Convert DataFrame into bytes file object.

    Args:
        df_data (pd.DataFrame): Dataframe to convert.
        path_to (str): Saving file name.

    Returns:
        io.BytesIO: Bytes file object.
    """
    buffer = io.BytesIO()
    if fnmatch(path_to, "*.csv"):
        df_data.to_csv(buffer, **kwargs)
    elif fnmatch(path_to, "*.parquet"):
        df_data.to_parquet(buffer, **kwargs)
    else:
        raise NotImplementedError("Uploading dataframe currently supported only for 'csv', 'parquet'.")
    buffer.seek(0)
    return buffer


def _dict_to_fileobj(dict_data: dict, path_to: str, **kwargs) -> io.BytesIO:
    """Convert dictionary into bytes file object.

    Args:
        dict_data (dict): Dictionary to convert.
        path_to (str): Saving file name.

    Returns:
        io.BytesIO: Bytes file object.
    """
    buffer = io.BytesIO()
    if fnmatch(path_to, "*.json"):
        buffer.write(json.dumps(dict_data, **kwargs).encode())
    elif fnmatch(path_to, "*.geojson"):
        if "type" in dict_data:
            if dict_data["type"] in [
                "Point",
                "MultiPoint",
                "LineString",
                "MultiLineString",
                "Polygon",
                "MultiPolygon",
                "GeometryCollection",
                "Feature",
                "FeatureCollection",
            ]:
                buffer.write(json.dumps(dict_data, **kwargs).encode())
            else:
                raise AttributeError(
                    "GeoJSONS must have a member with the name 'type', the value of the member must "
                    "be one of the following: 'Point', 'MultiPoint', 'LineString', 'MultiLineString',"
                    "'Polygon', 'MultiPolygon','GeometryCollection', 'Feature' or 'FeatureCollection'."
                )
    else:
        raise NotImplementedError("Uploading dictionary currently supported only for 'json' and 'geojson'.")
    buffer.seek(0)
    return buffer


def _list_to_fileobj(list_data: list, path_to: str, **kwargs) -> io.BytesIO:
    """Convert list into bytes file object.

    Args:
        list_data (list): List to convert.
        path_to (str): Saving file name.

    Returns:
        io.BytesIO: Bytes file object.
    """
    buffer = io.BytesIO()
    if fnmatch(path_to, "*.csv"):
        pd.DataFrame(list_data).to_csv(buffer, **kwargs)
    elif fnmatch(path_to, "*.txt"):
        for row in list_data:
            buffer.write(bytes(str(row) + "\n", "utf-8"))
    elif fnmatch(path_to, "*.json"):
        buffer.write(json.dumps(list_data, **kwargs).encode())
    else:
        raise NotImplementedError("Uploading list currently supported only for 'csv', 'txt' and 'json'.")
    buffer.seek(0)
    return buffer


def _str_to_fileobj(str_data: str, path_to: str, **kwargs) -> io.BytesIO:
    """Convert str into bytes file object.

    Args:
        str_data (str): String to convert.
        path_to (str): Saving file name.

    Returns:
        io.BytesIO: Bytes file object.
    """
    if fnmatch(path_to, "*.txt"):
        buffer = io.BytesIO(bytes(str_data.encode("utf-8")))
    else:
        raise NotImplementedError("Uploading string currently supported only for 'txt'.")
    buffer.seek(0)
    return buffer


def _np_array_to_fileobj(np_array_data: np.ndarray, path_to: str, **kwargs) -> io.BytesIO:
    """Convert numpy array into bytes file object.

    Args:
        np_array_data (np.ndarray): Numpy array to convert.
        path_to (str): Saving file name.

    Returns:
        io.BytesIO: Bytes file object.
    """
    buffer = io.BytesIO()
    if fnmatch(path_to, "*.csv"):
        np.savetxt(buffer, np_array_data, delimiter=",", **kwargs)
    elif fnmatch(path_to, "*.parquet"):
        pq.write_table(pa.table({"data": np_array_data}), buffer, **kwargs)
    else:
        raise NotImplementedError("Uploading numpy array currently supported only for 'csv' and 'parquet.")
    buffer.seek(0)
    return buffer


def _unsupp_data_to_fileobj(data: any, path_to: str, **kwargs) -> io.BytesIO:
    """Convert data into bytes file object using pickle file type.

    Args:
        data (any): Data to convert.
        path_to (str): Saving file name.

    Returns:
        io.BytesIO: Bytes file object.
    """
    buffer = io.BytesIO()
    if fnmatch(path_to, "*.pkl"):
        pickle.dump(data, buffer, **kwargs)
    else:
        raise NotImplementedError("This file type is not supported for this data. Use 'pkl' instead.")
    buffer.seek(0)
    return buffer


def upload_obj(
    obj: any,
    bucket: str,
    path_to: str,
    kwargs_boto: dict = None,
    kwargs_writing: dict = None,
) -> None:
    """Upload data from memory to S3 location.

    Args:
        obj (any): Data to upload.
        bucket (str): Bucket's name.
        path_to (str): Path location to save data.
        kwargs_boto (dict, optional): Dictionary of kwargs for boto3 function 'upload_fileobj'.
        kwargs_writing (dict, optional): Dictionary of kwargs for writing data.

    """
    if kwargs_boto is None:
        kwargs_boto = {}
    if kwargs_writing is None:
        kwargs_writing = {}

    if isinstance(obj, pd.DataFrame):
        obj = _df_to_fileobj(obj, path_to, **kwargs_writing)
    elif isinstance(obj, dict):
        obj = _dict_to_fileobj(obj, path_to, **kwargs_writing)
    elif isinstance(obj, list):
        obj = _list_to_fileobj(obj, path_to, **kwargs_writing)
    elif isinstance(obj, str):
        obj = _str_to_fileobj(obj, path_to, **kwargs_writing)
    elif isinstance(obj, np.ndarray):
        obj = _np_array_to_fileobj(obj, path_to, **kwargs_writing)
    else:
        obj = _unsupp_data_to_fileobj(obj, path_to, **kwargs_writing)
        warnings.warn("Data uploaded as pickle. Please consider other accessible file types among the supported ones.")

    s3 = s3_client()
    s3.upload_fileobj(obj, bucket, path_to, **kwargs_boto)
