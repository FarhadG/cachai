import glob
import json
import pandas as pd

from cachai.utils import constants as C


def build_metadata_df(config_glob_path):
    metadata = []
    config_files_paths = glob.glob(config_glob_path, recursive=True)
    for config_path in config_files_paths:
        dir_path = '/'.join(config_path.split('/')[:-1])
        logger_paths = glob.glob(f'{dir_path}/**/*.csv', recursive=True)

        advisor_logger_path = [logger_path for logger_path in logger_paths if C.ADVISOR_LOGGER in logger_path]
        experiment_logger_path = [logger_path for logger_path in logger_paths if C.EXPERIMENT_LOGGER in logger_path]

        if len(advisor_logger_path) != 1 or len(experiment_logger_path) != 1:
            raise ValueError(f'Expected one file with loggers')

        with open(config_path) as f:
            config = json.load(f)

        experiment_name = config[C.EXPERIMENT_NAME]
        metadata.append([
            experiment_name,
            advisor_logger_path[0],
            experiment_logger_path[0]
        ])
    return pd.DataFrame(metadata, columns=[C.EXPERIMENT_NAME, C.ADVISOR_LOGGER, C.EXPERIMENT_LOGGER])


def build_df(metadata_df, logger_name, callback=None):
    data = []
    for _, row in metadata_df.iterrows():
        model_logger = row[logger_name]
        if model_logger is None:
            continue
        df = pd.read_csv(model_logger)
        df[C.EXPERIMENT_NAME] = row[C.EXPERIMENT_NAME]
        if callback is not None:
            callback(df, row)
        data.append(df)
    return pd.concat(data) if len(data) > 0 else None
