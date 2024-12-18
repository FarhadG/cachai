import os
import logging
from pydantic import BaseModel

from src.utils.file_system import create_dir


def create_logger(
    name,
    output_dir,
    schema,
    include_headers=True,
    overwrite=True
):
    create_dir(output_dir)

    formatter = logging.Formatter('%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    output_path = os.path.join(output_dir, f'{name}.csv')
    field_names = schema.model_fields.keys()

    if overwrite and os.path.exists(output_path):
        os.remove(output_path)

    handler = logging.FileHandler(output_path)
    handler.setFormatter(formatter)

    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    def log(msg):
        data = schema.model_validate(msg).model_dump()
        csv_line = ','.join(str(data[field]) for field in field_names)
        logger.info(csv_line)

    logger.log = log

    if include_headers:
        logger.info(','.join(field_names))

    return logger
