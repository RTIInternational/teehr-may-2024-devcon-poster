import numpy as np



# @njit
def r_squared(primary_value: np.array, secondary_value: np.array) -> float:
    """Calculates r-squared value for a group"""
    if np.sum(primary_value) == 0:
        return np.nan

    pearson_correlation = (
        np.corrcoef(secondary_value, primary_value)
    )[0][1]
    r_squared_val = (np.power(pearson_correlation, 2))
    
    return r_squared_val

def relative_bias(primary_value: np.array, secondary_value: np.array) -> float:
    """Calculates relative bias value for a group"""

    if np.sum(primary_value) == 0:
        return np.nan
            
    relative_bias_val = np.sum(secondary_value - primary_value)/np.sum(primary_value)

    return relative_bias_val

def kling_gupta_efficiency(primary_value: np.array, secondary_value: np.array) -> float:
    """Calculates kling gupta efficiency value for a group
    
       KGE: 1 - sqrt(pow(corr(secondary_value, primary_value) - 1, 2) + pow(stddev(secondary_value)/ stddev(primary_value) - 1, 2) + pow(avg(secondary_value) / avg(primary_value) - 1, 2))   
    
    """
    if np.sum(primary_value) == 0:
        return np.nan
        
    # Pearson correlation coefficient (same as kge)
    linear_correlation = np.corrcoef(
        secondary_value, primary_value
    )[0, 1]

    # Variability_ratio
    variability_ratio = np.std(secondary_value) / np.std(primary_value)

    # Relative mean (same as kge)
    relative_mean = (
        np.mean(secondary_value)
        / np.mean(primary_value)
    )

    # Scaled Euclidean distance
    euclidean_distance = np.sqrt(
        ((linear_correlation - 1.0)) ** 2.0 +
        ((variability_ratio - 1.0)) ** 2.0 +
        ((relative_mean - 1.0)) ** 2.0
        )

    kge = 1.0 - euclidean_distance    
    
    return kge

def root_mean_squared_error(primary_value: np.array, secondary_value: np.array) -> float:
    """Calculates root mean squared error value for a group"""
    if primary_value.shape[0] == 0:
        return np.nan
        
    # rmse_val = (np.sum((secondary_value - primary_value) ** 2)/primary_value.shape[0]) ** 0.5
    rmse_val = (((secondary_value - primary_value) ** 2).mean()) ** 0.5
    return rmse_val