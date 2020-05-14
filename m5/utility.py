from zipfile import (
    ZIP_DEFLATED,
    ZipFile,
)

import numpy as np
import pandas as pd

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


def read_arrays(path, *args):
    with ZipFile(path) as zip_file, \
            zip_file.open('data.npz') as npz_file:
        return dict(np.load(npz_file))


def read_calendar(path):
    calendar = pd.read_csv(path, dtype=calendar_dtypes, usecols=calendar_dtypes)

    event_names = set(
        pd.unique(calendar[['event_name_1', 'event_name_2']].values.ravel()),
    ) - {pd.NA}
    calendar.event_name_1 = pd.Categorical(calendar.event_name_1, categories=event_names)
    calendar.event_name_2 = pd.Categorical(calendar.event_name_2, categories=event_names)

    event_types = set(
        pd.unique(calendar[['event_type_1', 'event_type_2']].values.ravel()),
    ) - {pd.NA}
    calendar.event_type_1 = pd.Categorical(calendar.event_type_1, categories=event_types)
    calendar.event_type_2 = pd.Categorical(calendar.event_type_2, categories=event_types)

    return calendar


def read_sales_data(path):
    return pd.read_csv(path, dtype=sales_data_dtypes)


def save_arrays(path, **kwargs):
    with ZipFile(path, 'w', compression=ZIP_DEFLATED) as zip_file, \
            zip_file.open('data.npz', 'w') as npz_file:
        np.savez_compressed(npz_file, **kwargs)
