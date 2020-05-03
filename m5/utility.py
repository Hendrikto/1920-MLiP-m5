import pandas as pd

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


def read_sales_data(path):
    return pd.read_csv(path, dtype=sales_data_dtypes)
