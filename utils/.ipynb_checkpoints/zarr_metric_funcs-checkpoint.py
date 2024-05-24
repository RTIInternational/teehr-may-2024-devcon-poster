import numpy as np
import pandas as pd
import xarray as xr


def get_zarr_metrics(primary_da: xr.DataArray, secondary_da: xr.DataArray, configuration: str) -> pd.DataFrame:
    """Compute performance metrics on dask arrays"""
    rmse = _root_mean_squared_error(primary_da, secondary_da)
    bias = _relative_bias(primary_da, secondary_da)
    r_squared = _r_squared(primary_da, secondary_da)
    kge = _kling_gupta_efficiency(primary_da, secondary_da)

    df = pd.DataFrame({
        "primary_location_id": primary_da.primary_location_id.values,
        "configuration": [configuration] * primary_da.shape[1],
        "kling_gupta_efficiency": kge.compute().values,
        "root_mean_squared_error": rmse.compute().values,
        "relative_bias": bias.compute().values,
        "r_squared": r_squared.compute().values
    })

    return df
    

def _root_mean_squared_error(primary_da: xr.DataArray, secondary_da: xr.DataArray) -> xr.DataArray:
    """Compute RMSE on dask arrays"""
    # rmse_vals = (((secondary_da - primary_da) ** 2).sum(axis=0)/primary_da.shape[0]) ** 0.5
    rmse_vals = (((secondary_da - primary_da) ** 2).mean(axis=0, skipna=True)) ** 0.5

    return rmse_vals
    

def _relative_bias(primary_da: xr.DataArray, secondary_da: xr.DataArray) -> xr.DataArray:
    """Compute relative bias on data arrays"""
    bias_vals = (secondary_da - primary_da).sum(axis=0, skipna=True) / primary_da.sum(axis=0, skipna=True)

    return bias_vals


def _r_squared(primary_da: xr.DataArray, secondary_da: xr.DataArray) -> xr.DataArray:
    """Calculates r-squared value for a data array"""
    # if np.sum(primary_da, axis=1).values == 0:
    #     return np.nan

    pearson_correlation = xr.corr(primary_da, secondary_da, dim="value_time")
    r_squared_val = pearson_correlation ** 2

    return r_squared_val


def _kling_gupta_efficiency(primary_da: xr.DataArray, secondary_da: xr.DataArray) -> xr.DataArray:
    """Calculates kling gupta efficiency value for a group
    
       KGE: 1 - sqrt(pow(corr(secondary_value, primary_value) - 1, 2) + pow(stddev(secondary_value)/ stddev(primary_value) - 1, 2) + pow(avg(secondary_value) / avg(primary_value) - 1, 2))   
    
    """
        
    # Pearson correlation coefficient (same as kge)
    pearson_correlation = xr.corr(primary_da, secondary_da, dim="value_time")

    # Variability_ratio
    variability_ratio = secondary_da.std(dim="value_time") / primary_da.std(dim="value_time")

    # Relative mean (same as kge)
    relative_mean = secondary_da.mean(dim="value_time") / primary_da.mean(dim="value_time")

    # Scaled Euclidean distance
    euclidean_distance = np.sqrt(
        ((pearson_correlation - 1.0)) ** 2.0 +
        ((variability_ratio - 1.0)) ** 2.0 +
        ((relative_mean - 1.0)) ** 2.0
        )

    kge = 1.0 - euclidean_distance    

    return kge

# =================================================================================================

# @njit



# def relative_bias(primary_value: np.array, secondary_value: np.array) -> float:
#     """Calculates relative bias value for a group"""

#     if np.sum(primary_value) == 0:
#         return np.nan
            
#     relative_bias_val = np.sum(secondary_value - primary_value)/np.sum(primary_value)

#     return relative_bias_val


# def kling_gupta_efficiency(primary_value: np.array, secondary_value: np.array) -> float:
#     """Calculates kling gupta efficiency value for a group
    
#        KGE: 1 - sqrt(pow(corr(secondary_value, primary_value) - 1, 2) + pow(stddev(secondary_value)/ stddev(primary_value) - 1, 2) + pow(avg(secondary_value) / avg(primary_value) - 1, 2))   
    
#     """
#     if np.sum(primary_value) == 0:
#         return np.nan
        
#     # Pearson correlation coefficient (same as kge)
#     linear_correlation = np.corrcoef(
#         secondary_value, primary_value
#     )[0, 1]

#     # Variability_ratio
#     variability_ratio = np.std(secondary_value) / np.std(primary_value)

#     # Relative mean (same as kge)
#     relative_mean = (
#         np.mean(secondary_value)
#         / np.mean(primary_value)
#     )

#     # Scaled Euclidean distance
#     euclidean_distance = np.sqrt(
#         ((linear_correlation - 1.0)) ** 2.0 +
#         ((variability_ratio - 1.0)) ** 2.0 +
#         ((relative_mean - 1.0)) ** 2.0
#         )

#     kge = 1.0 - euclidean_distance    
    
#     return kge
    

# def root_mean_squared_error(primary_value: np.array, secondary_value: np.array) -> float:
#     """Calculates root mean squared error value for a group"""
#     if primary_value.shape[0] == 0:
#         return np.nan
        
#     # rmse_val = (np.sum((secondary_value - primary_value) ** 2)/primary_value.shape[0]) ** 0.5
#     rmse_val = (((secondary_value - primary_value) ** 2).mean()) ** 0.5
#     return rmse_val