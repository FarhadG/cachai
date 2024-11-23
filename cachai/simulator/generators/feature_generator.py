import pandas as pd
import numpy as np

from cachai.utils import constants as C


def generate_y_true(df):
    result = []
    sorted_df = df.sort_values(C.TIMESTAMP)
    unique_records = sorted_df[C.RECORD].unique()
    last_timestamp = sorted_df.iloc[-1][C.TIMESTAMP]

    for record in unique_records:
        record_df = sorted_df[sorted_df[C.RECORD] == record]
        updated_record_df = record_df[record_df[C.OPERATION] == C.DATA_CHANGED]
        lowest_update_index = 0

        for i in range(0, len(record_df)):
            current_record_operation_time = record_df.iloc[i].timestamp

            if len(updated_record_df) > 0:
                next_updated_record_time = updated_record_df.iloc[lowest_update_index].timestamp
                while next_updated_record_time < current_record_operation_time:
                    if lowest_update_index >= len(updated_record_df)-1:
                        next_updated_record_time = last_timestamp
                        break
                    lowest_update_index += 1
                    next_updated_record_time = updated_record_df.iloc[lowest_update_index].timestamp
            else:
                next_updated_record_time = last_timestamp

            y_true = (next_updated_record_time - current_record_operation_time).total_seconds()
            # OFFSET for debugging to be able to shift the gold TTLs by a certain amount
            # if y_true != 0: y_true = max(y_true+1, 0)
            result.append([current_record_operation_time, record, y_true])
    return pd.DataFrame(result, columns=[C.TIMESTAMP, C.RECORD, C.Y_TRUE]).drop_duplicates()


def generate_y_true_json(df):
    result = {
        'stats': {
            'overall_mean': 0,
            'record_ttl_mean': {},
        }
    }
    records = df['record'].unique()
    last_timestamp = df.iloc[-1][C.TIMESTAMP]
    sorted_df = df.sort_values(C.TIMESTAMP)

    for record in records:
        record_df = sorted_df[sorted_df[C.RECORD] == record]
        updated_record_df = record_df[record_df[C.OPERATION] == C.DATA_CHANGED]

        time_stamp_diff = {}
        lowest_update_index = 0

        for i in range(0, len(record_df)):
            current_record_operation_time = record_df.iloc[i].timestamp

            if len(updated_record_df) > 0:
                next_updated_record_time = updated_record_df.iloc[lowest_update_index].timestamp
                while next_updated_record_time < current_record_operation_time:
                    if lowest_update_index >= len(updated_record_df)-1:
                        next_updated_record_time = last_timestamp
                        break
                    lowest_update_index += 1
                    next_updated_record_time = updated_record_df.iloc[lowest_update_index].timestamp
            else:
                next_updated_record_time = last_timestamp

            time_diff = (next_updated_record_time - current_record_operation_time).total_seconds()
            # OFFSET
            # if time_diff != 0: time_diff = max(time_diff+1, 0)
            time_stamp_diff[str(current_record_operation_time)] = time_diff

        result[record] = time_stamp_diff
        result['stats']['record_ttl_mean'][record] = np.mean(list(time_stamp_diff.values()))
    result['stats']['overall_mean'] = np.mean(list(result['stats']['record_ttl_mean'].values()))
    return result


def generate_feature(
    input,
    relationship_type='linear',
    weight=1.0,
    noise=0.0,
):
    if isinstance(input, float):
        input = np.array([input])
    elif isinstance(input, list):
        input = np.array(input)

    input_len = len(input)

    if relationship_type == 'linear':
        output = weight*input
    elif relationship_type == 'quadratic':
        output = weight*(input)**2
    elif relationship_type == 'logarithmic':
        output = weight*np.log(input+0.1)
    elif relationship_type == 'random':
        output = np.random.random(input_len)
    else:
        raise ValueError('Unsupported relationship type')

    # add noise for anything above 0
    if noise > 0:
        output += np.random.normal(0, noise, input_len)

    return np.round(output, 2)
