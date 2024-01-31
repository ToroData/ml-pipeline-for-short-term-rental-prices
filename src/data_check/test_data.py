"""_summary_
File to test data from basic_cleaning step

Author: Ricard Santiago Raigada García
Original code: Udacity
Date: January, 2024
"""

import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data: pd.DataFrame) -> None:
    """
    Test to ensure the dataset contains the expected columns with the exact names and order.

    This test will pass if the DataFrame `data` has all of the expected columns in the correct
    order. It is essential that the dataset includes these columns and that they follow the
    specified sequence, as downstream data processing and analysis may depend on this structure.

    Parameters:
    - data: A pandas DataFrame containing the dataset to be tested.

    Raises:
    - AssertionError: If `data` does not have the exact columns in the expected order.

    Returns:
    None
    """
    expected_colums: list = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data: pd.DataFrame) -> None:
    """
    Test to ensure the 'neighbourhood_group' column contains only known neighborhood names.

    This test checks that the values in the 'neighbourhood_group' column of the DataFrame `data`
    match a predefined set of known neighborhood names. The test is unordered, meaning that the
    values must match, but the order in which they appear does not matter. This can be used to
    validate data integrity, especially after merging or joining datasets that could introduce
    new, unexpected categories.

    Parameters:
    - data: A pandas DataFrame containing the dataset to be tested.

    Raises:
    - AssertionError: If `data` contains neighbourhood groups not present in the known names list.

    Returns:
    None
    """
    known_names: list = [
        "Bronx",
        "Brooklyn",
        "Manhattan",
        "Queens",
        "Staten Island"]

    neigh: set = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, - \
                                    73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data: pd.DataFrame) -> None:
    """
    Test to ensure the number of rows in the dataset falls within an expected range.

    This test will pass if the number of rows is greater than 15,000 and less than 1,000,000,
    indicating the dataset is not too small or too large for expected analysis.

    Parameters:
    - data: A pandas DataFrame containing the dataset to be tested.

    Raises:
    - AssertionError: If the number of rows in `data` is not within the range (15,000, 1,000,000).

    Returns:
    None
    """
    assert 15000 < data.shape[0] < 1000000


def test_price_range(
        data: pd.DataFrame,
        min_price: float,
        max_price: float) -> None:
    """
    Test to check if there are any rows in the dataset where the 'price' falls within the specified range.

    This test is to ensure that there are listings within the expected price range. It will pass if
    at least one listing's price is between `min_price` and `max_price` inclusive.

    Parameters:
    - data: A pandas DataFrame containing the dataset to be tested.
    - min_price: A float representing the minimum price value for the test range.
    - max_price: A float representing the maximum price value for the test range.

    Raises:
    - AssertionError: If there is not at least one row in `data` where 'price' falls within
      the range (min_price, max_price).

    Returns:
    None
    """
    assert data['price'].between(min_price, max_price).any()
