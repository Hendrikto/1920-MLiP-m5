import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def calculate_rolling_mean(data, n_products, window_size=28):
    column = f'rmean{window_size}_sales'
    data[column] = float('nan')
    for i in tqdm(range(n_products), desc=f'window size = {window_size}'):
        data[column][window_size * n_products + i::n_products] = (
            data
            .sales  # select column
            [i::n_products]  # select rows
            .rolling(window_size, min_periods=1)
            .mean()
            [:-window_size]  # discard future data
            .values  # discard index
        )


def cat2int(data_frame):
    for column_name, column in data_frame.items():
        t = column.dtype
        if t.name == 'category':
            column = column.cat.codes
        elif t.name == 'bool':
            column = column.astype('int8')
        data_frame[column_name] = column


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
