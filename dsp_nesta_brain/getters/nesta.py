"""Getters for Nesta datasets

Usage:

Run these following two commands in the terminal to fetch the data from S3
```
python dsp_nesta_brain/getters/nesta.py --download
python dsp_nesta_brain/getters/nesta.py --unzip
```

Use the following function to view the metadata
```
metadata_df = load_metadata()
```

"""
import argparse
import shutil

from pathlib import Path

import pandas as pd

from dsp_nesta_brain import PROJECT_DIR
from dsp_nesta_brain import logger
from dsp_nesta_brain.utils import s3


LOCAL_PATH = PROJECT_DIR / "data/website_2024-10-29"
S3_PATH = "data/nesta_brain/"

s3_client = s3.s3_client()
S3_KEY = f"{S3_PATH}{LOCAL_PATH.name}.zip"


def upload_website_data(path: Path = LOCAL_PATH) -> None:
    """Compress and upload the website data to S3"""
    # Compress the website data

    # Check if the file already exists in S3
    try:
        s3_client.head_object(Bucket=s3.BUCKET_NAME_RAW, Key=S3_KEY)
        logger.info(
            f"File {S3_KEY} already exists in s3://{s3.BUCKET_NAME_RAW}. Skipping upload to avoid overwriting."
        )
        return
    except s3_client.exceptions.ClientError as e:
        # If a 404 error is raised, the file does not exist, so proceed with the upload
        if e.response["Error"]["Code"] == "404":
            logger.info(f"File {S3_KEY} does not exist in S3. Proceeding with upload.")
        else:
            # If it's any other error, log it and exit the function
            logger.error(f"Error checking if file exists in S3: {e}")
            return

    compressed_db_path = f"{path}.zip"
    shutil.make_archive(path, "zip", path)
    logger.info(f"Compressed website data to {compressed_db_path}")
    try:
        logger.info(f"Uploading {compressed_db_path} to S3 bucket {s3.BUCKET_NAME_RAW}")
        s3_client.upload_file(compressed_db_path, s3.BUCKET_NAME_RAW, S3_KEY)
        logger.info(f"Successfully uploaded {compressed_db_path} to s3://{s3.BUCKET_NAME_RAW}/{S3_KEY}")
    except Exception as e:
        logger.error(f"Error uploading database to S3: {e}")


def download_website_data(path: Path = LOCAL_PATH) -> None:
    """Download compressed website data from S3"""
    download_path = f"{path}.zip"
    try:
        logger.info(f"Downloading {S3_KEY} from S3 bucket {s3.BUCKET_NAME_RAW}")
        s3_client.download_file(s3.BUCKET_NAME_RAW, S3_KEY, download_path)
        logger.info(f"Successfully downloaded {S3_KEY} to {download_path}")
    except Exception as e:
        logger.error(f"Error downloading database from S3: {e}")


def unzip_data(path: Path = LOCAL_PATH) -> None:
    """Uncompress the website data"""
    try:
        compressed_db_path = f"{path}.zip"
        if Path(compressed_db_path).exists():
            path.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(compressed_db_path, path)
            logger.info(f"Uncompressed website data to {path}")
        else:
            logger.error(f"File not found at {compressed_db_path}. Run with --download first")
    except Exception as e:
        logger.error(f"Error uncompressing database: {e}")


def load_metadata() -> pd.DataFrame:
    """Load the metadata file into a DataFrame"""
    metadata_path = LOCAL_PATH / "metadata.jsonl"
    if metadata_path.exists():
        return pd.read_json(metadata_path, lines=True)
    else:
        logger.error(f"Metadata file not found at {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload or download Nesta website data")
    parser.add_argument("--upload", action="store_true", help="Upload data to S3")
    parser.add_argument("--download", action="store_true", help="Download data from S3")
    parser.add_argument("--unzip", action="store_true", help="Uncompress data")
    args = parser.parse_args()

    # Check which flag is used and call the appropriate function
    if args.upload:
        upload_website_data()
    elif args.download:
        download_website_data()
    elif args.unzip:
        unzip_data()
    else:
        logger.info("Please specify --upload, --download or --unzip")
