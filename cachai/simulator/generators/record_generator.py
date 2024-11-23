import random
import numpy as np
from typing import Callable


def generate_record_payload(
    count=1,
    payload_length=10
):
    return [{
        f'attribute_{i}': ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=payload_length))
    } for i in range(1, count+1)]


def generate_records(
    records_count,
    key_length=10,
    generate_record_payload=generate_record_payload
):
    records_dict = {}
    for _ in range(records_count):
        # ensure the item is unique
        while True:
            record = ''.join(random.choices(
                'abcdefghijklmnopqrstuvwxyz0123456789', k=key_length))
            if record not in records_dict:
                break
        records_dict[record] = generate_record_payload()
    record_keys = np.array(list(records_dict.keys()))
    return record_keys, records_dict
