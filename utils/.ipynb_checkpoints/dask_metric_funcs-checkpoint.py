"""Functions to calculate test metrics from dask dataframes"""
from typing import List

import dask
import pandas as pd
import numpy as np
from numba import jit, njit


def r_squared(group: pd.DataFrame) -> pd.Series:
    """Calculates r-squared value for a group"""
    if np.sum(group["primary_value"]) == 0:
        return np.nan

    pearson_correlation = (
        np.corrcoef(group["secondary_value"], group["primary_value"])
    )[0][1]
    r_squared_val = (np.power(pearson_correlation, 2))
    
    return r_squared_val


def kling_gupta_efficiency(group: pd.DataFrame) -> pd.Series:
    """Calculates kling gupta efficiency value for a group
    
       KGE: 1 - sqrt(pow(corr(secondary_value, primary_value) - 1, 2) + pow(stddev(secondary_value)/ stddev(primary_value) - 1, 2) + pow(avg(secondary_value) / avg(primary_value) - 1, 2))   
    
    """
    if np.sum(group["primary_value"]) == 0:
        return np.nan
        
    # Pearson correlation coefficient (same as kge)
    linear_correlation = np.corrcoef(
        group["secondary_value"], group["primary_value"]
    )[0, 1]

    # Variability_ratio
    variability_ratio = np.std(group["secondary_value"]) / np.std(group["primary_value"])

    # Relative mean (same as kge)
    relative_mean = (
        np.mean(group["secondary_value"])
        / np.mean(group["primary_value"])
    )

    # Scaled Euclidean distance
    euclidean_distance = np.sqrt(
        ((linear_correlation - 1.0)) ** 2.0 +
        ((variability_ratio - 1.0)) ** 2.0 +
        ((relative_mean - 1.0)) ** 2.0
        )

    kge = 1.0 - euclidean_distance    
    
    return kge

def relative_bias(group: pd.DataFrame) -> pd.Series:
    """Calculates relative bias value for a group"""

    if np.sum(group["primary_value"]) == 0:
        return np.nan
            
    relative_bias_val = np.sum(group["secondary_value"] - group["primary_value"])/np.sum(group["primary_value"])

    return relative_bias_val

def root_mean_squared_error(group: pd.DataFrame) -> pd.Series:
    """Calculates root mean squared error value for a group"""
    if len(group["primary_value"]) == 0:
        return np.nan
        
    # rmse_val = (np.sum((secondary_value - primary_value) ** 2)/primary_value.shape[0]) ** 0.5
    rmse_val = (((group["secondary_value"] - group["primary_value"]) ** 2).mean()) ** 0.5
    return rmse_val

# AGGREGATION FUNCTIONS -- Are much more efficient, but only operate on a single column at a time,
# making more complicated metrics much more difficult (even correlation).
# def rmse(ddf_tmp: dask.dataframe, groupby_fields: List) -> pd.Series:
#     """Computes root mean squared error of primary vs. secondary"""
#     ddf_tmp["absolute_diff_squared"] = ddf_tmp["absolute_difference"] ** 2

#     # NOTE: This is ~3 secs faster than the double-groupby line below
#     int_output = ddf_tmp.groupby(groupby_fields).agg({"absolute_diff_squared": ["sum", "count"]})
#     output = (int_output["absolute_diff_squared", "sum"] / int_output["absolute_diff_squared", "count"]) ** 0.5

#     # output = ddf.groupby(groupby_fields)["absolute_diff_squared"].sum() / ddf.groupby(groupby_fields)["absolute_diff_squared"].count()

#     return output.compute().rename("root_mean_squared_error")


# def relative_bias(ddf_tmp: dask.dataframe, groupby_fields: List) -> pd.Series:
#     """Computes relative bias of primary vs. secondary"""
#     ddf_tmp["difference"] = ddf_tmp["secondary_value"] - ddf_tmp["primary_value"]

#     int_output = ddf_tmp.groupby(groupby_fields).agg({"difference": ["sum"], "primary_value": ["sum"]})
#     output = int_output["difference", "sum"] / int_output["primary_value", "sum"]
    
#     return output.compute().rename("relative_bias")


# def r_squared(ddf_tmp: dask.dataframe, groupby_fields: List) -> pd.Series:
#     """Computes r-squared of primary vs. secondary"""
#     ddf_tmp["primary_secondary_product"] = ddf_tmp["primary_value"] * ddf_tmp["secondary_value"]
#     int_output = ddf_tmp.groupby(groupby_fields).agg({"secondary_value": ["mean", "std"], "primary_value": ["mean", "std", "count"], "primary_secondary_product": ["sum"]})
#     output = (int_output["primary_secondary_product", "sum"]/((int_output["primary_value", "count"] - 1) * int_output["primary_value", "std"] * int_output["secondary_value", "std"])) ** 2
    
#     # output = ddf_tmp.groupby(groupby_fields)[["secondary_value", "primary_value"]].corr() # ** 2  # RAISES ERROR
#     # output = ddf_tmp.groupby(groupby_fields).apply(lambda x: x.corr())  # RAISES ERROR

#     return output.compute().rename("r_squared")


# def kge(ddf_tmp: dask.dataframe, groupby_fields: List) -> pd.Series:
#     """Computes kling-gupta efficiency of primary vs. secondary"""
    
#     # Intermediate output
#     int_mean_std = output = ddf_tmp.groupby(groupby_fields).agg({'primary_value': ['mean', 'std'], 'secondary_value': ['mean', 'std']})
    
#     # CORRELATION TERM
#     ddf_tmp["primary_secondary_product"] = ddf_tmp["primary_value"] * ddf_tmp["secondary_value"]
#     int_output = ddf_tmp.groupby(groupby_fields).agg({"secondary_value": ["mean", "std"], "primary_value": ["mean", "std", "count"], "primary_secondary_product": ["sum"]})
#     corr = int_output["primary_secondary_product", "sum"]/((int_output["primary_value", "count"] - 1) * int_output["primary_value", "std"] * int_output["secondary_value", "std"])   

#     # KGE
#     output = ((corr - 1) ** 2 + (int_output["secondary_value", "std"]/int_output["primary_value", "std"] - 1) ** 2 + (int_output["secondary_value", "mean"]/int_output["primary_value", "mean"] - 1) ** 2) ** 0.5

#     return output.compute().rename("kling_gupta_efficiency")