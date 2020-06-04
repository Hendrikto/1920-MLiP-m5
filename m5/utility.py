import csv
from zipfile import (
    ZIP_DEFLATED,
    ZipFile,
)

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import config


def read_arrays(path, *args):
    with ZipFile(path) as zip_file, \
            zip_file.open('data.npz') as npz_file:
        return dict(np.load(npz_file))


def read_calendar(path):
    calendar = pd.read_csv(path, dtype=config.calendar_dtypes, usecols=config.calendar_dtypes)

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
    return pd.read_csv(path, dtype=config.sales_data_dtypes)


def save_arrays(path, **kwargs):
    with ZipFile(path, 'w', compression=ZIP_DEFLATED) as zip_file, \
            zip_file.open('data.npz', 'w') as npz_file:
        np.savez_compressed(npz_file, **kwargs)


def save_predictions(
    predictions_evaluation,
    predictions_validation,
    product_ids,
    path='submission.csv',
):
    n_days = predictions_validation.shape[1]
    with open(path, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(('id', *(f'F{i + 1}' for i in range(n_days))))
        for product_id, values_evaluation, values_validation in tqdm(zip(
            product_ids,
            predictions_evaluation,
            predictions_validation,
        ), total=len(product_ids)):
            base_id = '_'.join(product_id.split('_')[:-1])
            writer.writerow((base_id + '_evaluation', *values_evaluation))
            writer.writerow((base_id + '_validation', *values_validation))
