import numpy as np

feature_names_categorical = (
    'cat_id',
    'dept_id',
    'event_name_1',
    'event_type_1',
    'event_name_2',
    'event_type_2',
    'item_id',
    'month',
    'snap_CA',
    'snap_TX',
    'snap_WI',
    'state_id',
    'store_id',
    'wday',
    'year',
)
feature_names_numerical = (
    'day',
)
feature_names = (
    *feature_names_numerical,
    *feature_names_categorical,
    'sales',
)
feature_index = dict(zip(feature_names, range(len(feature_names))))


def generate_data(sales_data, calendar, dtype='float32'):
    n_days = int(sales_data.columns[-1].split('_')[1])
    n_features = len(feature_index)
    n_products = len(sales_data)

    data = np.empty((n_products * n_days, n_features), dtype=dtype)

    data[:, feature_index['day']] = np.repeat(
        np.arange(1, n_days + 1, dtype='int16'),
        n_products,
    )
    data[:, feature_index['sales']] = np.concatenate([
        sales_data[f'd_{i}'] for i in range(1, n_days + 1)
    ])
    for column_name, column in sales_data.loc[:, 'item_id':'state_id'].items():
        data[:, feature_index[column_name]] = np.tile(column.cat.codes, n_days)
    for column_name, column in calendar.loc[:n_days - 1, 'wday':'snap_WI'].items():
        if column.dtype == 'bool':
            values = column
        else:
            values = column.cat.codes
        data[:, feature_index[column_name]] = np.repeat(values, n_products)

    return data
