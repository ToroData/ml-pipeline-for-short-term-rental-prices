#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Perform basic cleaning on the input data artifact and save the cleaned data.

    This function drops outliers based on price, converts date columns to datetime objects,
    and applies a logarithmic transformation to reduce skewness in the 'minimum_nights' column.
    The cleaned dataset is then saved to a CSV file specified by the user.

    Parameters:
    - args: argparse.Namespace object containing arguments for the data cleaning process.
            Expected arguments are:
            - input_artifact: str, the path to the input data artifact.
            - output_artifact: str, the path where the cleaned data artifact will be saved.
            - output_type: str, the type of the output file.
            - output_description: str, a description of the output artifact.
            - min_price: float, the minimum price threshold for filtering data.
            - max_price: float, the maximum price threshold for filtering data.

    Returns:
    None

    The function does not return any value but writes the cleaned dataset to a file and logs the
    output artifact in the Weights & Biases run.

    The function assumes that Weights & Biases (wandb) and logging (logger) are properly set up
    before calling this function.
    """
    # Initialize a Weights & Biases run
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Logging for debugging
    logger.info("Download artifact")

    # Use the W&B artifact as the input data source
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Load the dataset
    logger.info("Loading dataframe")
    dataframe = pd.read_csv(artifact_local_path)
    
    # Drop missing 'name' and 'host_name'
    dataframe['name'].dropna(inplace=True)
    dataframe['host_name'].dropna(inplace=True)

    # Drop outliers
    logger.info("Dropping outliers")
    idx = dataframe['price'].between(args.min_price, args.max_price)
    dataframe = dataframe[idx].copy()
    
    # Fill missing 'reviews_per_month' with 0
    dataframe['reviews_per_month'].fillna(0, inplace=True)

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    dataframe['last_review'] = pd.to_datetime(dataframe['last_review'])

    # Log transform minimum_nights to reduce skewness
    logger.info("Reducing skewness of minimum_nights")
    dataframe['minimum_nights'] = np.where(
        dataframe['minimum_nights'] > 0, np.log(
            dataframe['minimum_nights']), 0)
    
    # Drop data for wrong geolocations
    logger.info("Correcting the coordinate range")
    idx = dataframe['longitude'].between(-74.25, -73.50) & dataframe['latitude'].between(40.5, 41.2)
    dataframe = dataframe[idx].copy()

    # Save the cleaned dataset to the specified output artifact path
    logger.info("Saving dataframe")
    dataframe.to_csv(args.output_artifact, index=False)

    # Log the output artifact in W&B
    logger.info("Uploading artifact to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The file path for the input data artifact that needs cleaning.",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The file path where the cleaned data artifact will be saved.",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type of the output file.",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A description of the output artifact,\
            such as its content and the cleaning operations performed.",
        required=True)

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price threshold used to filter the data.\
            Rows with prices below this value will be excluded.",
        required=True)

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price threshold used to filter the data.\
            Rows with prices above this value will be excluded.",
        required=True)

    args = parser.parse_args()

    go(args)
