import os

import cachai.utils.models as M
import cachai.utils.constants as C


def create_dir(dir_path, verbose=False):
    # TODO: make printing based on global verbose flag
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        if verbose:
            print(f'Folder {dir_path} created')
    else:
        if verbose:
            print(f'Folder {dir_path} already exists')


def write_config(config, output_dir):
    create_dir(output_dir)
    config_path = f'{output_dir}/config.json'
    with open(config_path, 'w') as f:
        f.write(config.model_dump_json(indent=4))


def standardize_path(path):
    return path.lower().replace('\\', '/').rstrip('/').replace(' ', '_')
