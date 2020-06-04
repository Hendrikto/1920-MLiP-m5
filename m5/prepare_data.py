import numpy as np
import pandas as pd

import config


def generate_data(sales_data, calendar, dtype='float32'):
    n_days = int(sales_data.columns[-1].split('_')[1])
    n_features = len(config.feature_index)
    n_products = len(sales_data)

    data = np.empty((n_products * n_days, n_features), dtype=dtype)

    data[:, config.feature_index['day']] = np.repeat(
        np.arange(1, n_days + 1, dtype='int16'),
        n_products,
    )
    data[:, config.feature_index['sales']] = np.concatenate([
        sales_data[f'd_{i}'] for i in range(1, n_days + 1)
    ])
    for column_name, column in sales_data.loc[:, 'item_id':'state_id'].items():
        data[:, config.feature_index[column_name]] = np.tile(column.cat.codes, n_days)
    for column_name, column in calendar.loc[:n_days - 1, 'wday':'snap_WI'].items():
        if column.dtype == 'bool':
            values = column
        else:
            values = column.cat.codes
        data[:, config.feature_index[column_name]] = np.repeat(values, n_products)

    return data


def generate_data_frame(sales_data, calendar):
    n_days = int(sales_data.columns[-1].split('_')[1])
    n_products = len(sales_data)

    data = {}

    data['day'] = np.repeat(np.arange(1, n_days + 1, dtype='int16'), n_products)
    data['sales'] = pd.concat(
        (sales_data[f'd_{i}'] for i in range(1, n_days + 1)),
        copy=False,
        ignore_index=True,
    )
    for column_name, column in sales_data.loc[:, 'item_id':'state_id'].items():
        data[column_name] = pd.concat((column,) * n_days, copy=False, ignore_index=True)
    for column_name, column in calendar.loc[:n_days - 1, 'wday':'snap_WI'].items():
        data[column_name] = column.repeat(n_products).reset_index(drop=True)

    return pd.DataFrame(data)


def generate_prediction_data(sales_data, calendar, days):
    """
    Generate prediction data.

    :param sales_data: dataframe containing sales information
    :param calendar: dataframe with calendar information
    :param days: 0-based range of days to predict

    :returns: array with generated prediction data
    """
    n_days = len(days)
    n_features = len(config.feature_names) - 1
    n_products = len(sales_data)

    data = np.empty((n_days * n_products, n_features), dtype='int16')
    data[:, config.feature_index['day']] = np.repeat(days, n_products) + 1  # sales_data is 1-based
    for column_name, column in calendar.loc[days].items():
        if column.dtype == 'bool':
            values = column
        else:
            values = column.cat.codes
        data[:, config.feature_index[column_name]] = np.repeat(values, n_products)
    for column_name, column in sales_data.loc[:, 'item_id':'state_id'].items():
        data[:, config.feature_index[column_name]] = np.tile(column.cat.codes, n_days)

    return data
