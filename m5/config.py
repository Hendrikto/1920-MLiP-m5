from pathlib import Path

# data types
calendar_dtypes = {
    'event_name_1': 'string',
    'event_name_2': 'string',
    'event_type_1': 'string',
    'event_type_2': 'string',
    'month': 'category',
    'snap_CA': 'bool',
    'snap_TX': 'bool',
    'snap_WI': 'bool',
    'wday': 'category',
    'year': 'category',
}
sales_data_dtypes = {
    'id': 'string',
    **{column: 'category' for column in (
        'item_id',
        'dept_id',
        'cat_id',
        'store_id',
        'state_id',
    )},
    **{f'd_{i}': 'int16' for i in range(1, 1942)},
}

# features
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

# paths
input_path = Path('/kaggle/input')
m5_path = input_path / 'm5-forecasting-accuracy'
