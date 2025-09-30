import os
import numpy as np
import pandas as pd
import geopandas as gpd
import sklearn.datasets as datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler


####################################################################################################


def load_preprocessed_climate(filepath):
    """
    Loads a preprocessed climate dataset. Specifically, we split the original data
    set into a historical period (from years 1900-1999) and a recent period (2013-2023),
    and take climate change to be the percent difference between historical and 
    recent values in each of 24 features. Finally, percent change values are 
    normalized with standard scaling.

    Args:
        filepath (str): This should be the filepath to the common folder in which the 
            all of the climate data is contained. I.e. 'data/climate' or some variation 
            on this depending upon the current working directory. 

    Returns:
        data (np.ndarray): Size (344 x 24) numpy array with preprocessed data for each 
            of 344 climate division locations across 24 measurement features. 

        data_labels (np.ndarray): Array of labels associated with data points. In this case, 
            the dataset does not have any labels and will return None. 
        
        feature_labels (List[str]): List of feature names associated 
            with each of the columns in the scaled data array.

        scaler (Scaler): Sklearn MinMaxScaler, which is handy for returning data to 
            its original values. 
    """
    # This is intended to be run from the main directory.
    shape_file_path = os.path.join(filepath, 'climate_divisions/climate_divisions.shp')
    dtype_dict = {'CLIMDIV': 'str'}
    gdf = gpd.read_file(shape_file_path, dtype = dtype_dict)
    gdf['CLIMDIV'] = gdf['CLIMDIV'].apply(lambda x: f'{int(x):04d}')

    data_file_path = os.path.join(filepath, 'climate.csv')
    climate_data = pd.read_csv(data_file_path, dtype={'ID': str, 'Year': str})
    climate_data.set_index(['ID', 'Year'], inplace=True)

    historical_years = [str(i) for i in range(1900,2000)]
    recent_years = [str(i) for i in range(2013,2024)]

    historical = climate_data.loc[pd.IndexSlice[:, historical_years], :]
    recent = climate_data.loc[pd.IndexSlice[:, recent_years], :]

    historical_avg = historical.groupby(level='ID').mean()
    recent_avg = recent.groupby(level='ID').mean()

    climate_change = (recent_avg - historical_avg)/historical_avg
    climate_change = climate_change.loc[gdf.CLIMDIV,:]

    '''
    # Using seasons instead of months:
    groupings = {'pcpn_winter': ['pcpn_dec', 'pcpn_jan', 'pcpn_feb'],
                'pcpn_spring': ['pcpn_mar', 'pcpn_apr', 'pcpn_may'],
                'pcpn_summer': ['pcpn_june', 'pcpn_july', 'pcpn_aug'],
                'pcpn_fall': ['pcpn_sept', 'pcpn_oct', 'pcpn_nov'],
                'temp_winter': ['temp_dec', 'temp_jan', 'temp_feb'],
                'temp_spring': ['temp_mar', 'temp_apr', 'temp_may'],
                'temp_summer': ['temp_june', 'temp_july', 'temp_aug'],
                'temp_fall': ['temp_sept', 'temp_oct', 'temp_nov']}

    seasonal_historical = pd.DataFrame()
    seasonal_recent = pd.DataFrame()
    seasonal_climate_change = pd.DataFrame()

    # Calculate the average for each group of months
    for group_name, columns in groupings.items():
        seasonal_historical[group_name] = historical_avg[columns].mean(axis=1)
        seasonal_recent[group_name] = recent_avg[columns].mean(axis=1)
        seasonal_climate_change[group_name] = climate_change[columns].mean(axis=1)
    '''
        
    # Normalize the data
    data = climate_change.to_numpy()
    data_labels = None
    feature_labels = list(climate_change.columns)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, data_labels, feature_labels, scaler


####################################################################################################


def load_preprocessed_digits():
    """
    Loads a preprocessed sklearn digits dataset. Values are normalized to the range 
    with standard scaling.

    Args:


    Returns:
        data (np.ndarray): (samples x features) numpy array of preprocessed data. 

        data_labels (np.ndarray): Array of labels associated with data points.
        
        feature_labels (List[str]): List of feature names associated 
            with each of the columns in the scaled data array.

        scaler (Scaler): Sklearn MinMaxScaler, which is handy for returning data to 
            its original values. 
    """
    data, data_labels = datasets.load_digits(return_X_y=True)

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    feature_labels = [str(i) for i in range(scaled_data.shape[1])]

    return scaled_data, data_labels, feature_labels, scaler


####################################################################################################


def load_preprocessed_mnist():
    """
    Loads a preprocessed MNIST digits dataset. Values are normalized to the range 
    with standard scaling.

    Args:


    Returns:
        data (np.ndarray): (samples x features) numpy array of preprocessed data. 

        data_labels (np.ndarray): Array of labels associated with data points.
        
        feature_labels (List[str]): List of feature names associated 
            with each of the columns in the scaled data array.

        scaler (Scaler): Sklearn MinMaxScaler, which is handy for returning data to 
            its original values. 
    """
    data, data_labels = datasets.fetch_openml(
        name = 'mnist_784',
        return_X_y=True,
        as_frame=False
    )
    data_labels = data_labels.astype(int)

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    feature_labels = [str(i) for i in range(scaled_data.shape[1])]

    return scaled_data, data_labels, feature_labels, scaler


####################################################################################################


def load_preprocessed_fashion():
    """
    Loads a preprocessed Fashion MNIST dataset. Values are normalized to the range [0,1].

    Args:


    Returns:
        data (np.ndarray): (samples x features) numpy array of preprocessed data. 

        data_labels (np.ndarray): Array of labels associated with data points.
        
        feature_labels (List[str]): List of feature names associated 
            with each of the columns in the scaled data array.

        scaler (Scaler): Sklearn MinMaxScaler, which is handy for returning data to 
            its original values. 
    """
    data, data_labels = datasets.fetch_openml(
        name = 'Fashion-MNIST',
        return_X_y=True,
        as_frame=False
    )
    data_labels = data_labels.astype(int)

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    feature_labels = [str(i) for i in range(scaled_data.shape[1])]

    return scaled_data, data_labels, feature_labels, scaler


####################################################################################################


def load_preprocessed_covtype():
    """
    Loads a preprocessed Forest Cover type dataset. Values are normalized with standard scaling.

    Args:


    Returns:
        data (np.ndarray): (samples x features) numpy array of preprocessed data. 

        data_labels (np.ndarray): Array of labels associated with data points.
        
        feature_labels (List[str]): List of feature names associated 
            with each of the columns in the scaled data array.

        scaler (Scaler): Sklearn MinMaxScaler, which is handy for returning data to 
            its original values. 
    """
    D = datasets.fetch_covtype()
    data = D['data']
    data_labels = D['target']
    feature_labels = D['feature_names']

    # Only taking non-categorical features.
    data = data[:,:10]
    feature_labels = feature_labels[:10]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    #feature_labels = [str(i) for i in range(scaled_data.shape[1])]

    return scaled_data, data_labels, feature_labels, scaler


####################################################################################################


def load_preprocessed_anuran(filepath):
    """
    Loads a preprocessed anuran dataset. Values are normalized via standard scaling.

    Args:
        filepath (str): This should be the filepath to the common folder in which the 
            all of the digits data is contained. I.e. 'data/anuran' or some variation 
            on this depending upon the current working directory. 

    Returns:
        data (np.ndarray): (samples x features) numpy array of preprocessed data. 

        data_labels (np.ndarray): Array of labels associated with data points. In this case, 
            the dataset does not have any labels and will return None. 
        
        feature_labels (List[str]): List of feature names associated 
            with each of the columns in the scaled data array.

        scaler (Scaler): Sklearn MinMaxScaler, which is handy for returning data to 
            its original values. 
    """
    data_filepath = os.path.join(filepath, 'Frogs_MFCCs.csv')
    anuran = pd.read_csv(data_filepath)
    anuran = anuran.iloc[:, :-4]
    anuran = anuran.to_numpy()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(anuran)

    feature_labels = [str(i) for i in range(scaled_data.shape[1])]

    return scaled_data, None, feature_labels, scaler



