import numpy as np
import pandas as pd

import config


def cat2int(data_frame):
    for column_name, column in data_frame.items():
        t = column.dtype
        if t.name == 'category':
            column = column.cat.codes
        elif t.name == 'bool':
            column = column.astype('int8')
        data_frame[column_name] = column


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


def generate_data_frame(sales_data, calendar, prices):
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
    for column_name, column in calendar.loc[:n_days - 1, 'wm_yr_wk':'snap_WI'].items():
        data[column_name] = column.repeat(n_products).reset_index(drop=True)
    data['wm_yr_wk'] = calendar.wm_yr_wk[:n_days].repeat(n_products).reset_index(drop=True)

    data = pd.DataFrame(data)

    data = data.merge(prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
    data.sell_price = data.sell_price.astype('float32')
    data.drop(columns=('wm_yr_wk'), inplace=True)

    return data


def generate_prediction_data(sales_data, calendar, prices, days):
    """
    Generate prediction data.

    :param sales_data: dataframe containing sales information
    :param calendar: dataframe with calendar information
    :param days: 0-based range of days to predict

    :returns: dataframe with generated prediction data
    """
    n_days = len(days)
    n_products = len(sales_data)

    data = {}

    data['day'] = np.repeat(np.arange(days.start, days.stop, dtype='int16'), n_products)
    for column_name, column in sales_data.loc[:, 'item_id':'state_id'].items():
        data[column_name] = pd.concat((column,) * n_days, copy=False, ignore_index=True)
    for column_name, column in calendar.loc[days].items():
        data[column_name] = column.repeat(n_products).reset_index(drop=True)
    data['wm_yr_wk'] = calendar.wm_yr_wk[:n_days].repeat(n_products).reset_index(drop=True)

    data = pd.DataFrame(data)

    data = data.merge(prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
    data.sell_price = data.sell_price.astype('float32')
    data.drop(columns=('wm_yr_wk'), inplace=True)

    return data
