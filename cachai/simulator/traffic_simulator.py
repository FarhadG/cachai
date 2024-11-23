import numpy as np
import pandas as pd

import cachai.utils.models as M
import cachai.utils.constants as C
import cachai.simulator.generators.record_generator as RecordGenerator
import cachai.simulator.generators.record_sampler as RecordSampler
import cachai.simulator.generators.traffic_generator as TrafficGenerator
import cachai.simulator.generators.feature_generator as FeatureGenerator


class AdvancedSimulator():
    DF_COLUMNS = [C.TIMESTAMP, C.OPERATION, C.RECORD, C.PAYLOAD]
    SUPPORTED_OPERATIONS = [C.CREATE, C.READ, C.UPDATE, C.DELETE, C.DATA_CHANGED]

    def __init__(self, config):
        self.record_keys = []
        self.records_dict = {}
        self.load_phase_df = None
        self.run_phase_df = None
        self.features_df = None
        self.y_true_json = None
        self.records_count = config.records_count
        self.operations_count = config.operations_count

    def generate_records(self):
        return RecordGenerator.generate_records(
            self.records_count, generate_record_payload=self.generate_record_payload
        )

    def generate_record_payload(self):
        return RecordGenerator.generate_record_payload()

    def generate_y_true(self, df):
        return FeatureGenerator.generate_y_true(df)

    def generate_y_true_json(self, df):
        return FeatureGenerator.generate_y_true_json(df)

    def generate_traffic(self, start='01-01-2024 00:00:00', end='01-01-2024 00:00:00', freq='s'):
        return TrafficGenerator.generate_traffic(start, end, freq=freq, count=self.operations_count)

    def sample_operations(self):
        return TrafficGenerator.sample_operations()

    def sample_records(self):
        return RecordSampler.zipf()

    def reset_progress(self):
        self._progress = 0.0

    def update_progress(self, increment=1.0):
        self._progress += (increment/self.operations_count)

    def get_progress(self):
        return round(self._progress, 2)

    def generate(self):
        self.reset_progress()
        # record generation
        self.record_keys, self.records_dict = self.generate_records()

        # load phase
        load_phase = []
        for record, payload in self.records_dict.items():
            load_phase.append([None, C.CREATE, record, payload])
        self.load_phase_df = pd.DataFrame(load_phase, columns=AdvancedSimulator.DF_COLUMNS)

        # run phase
        run_phase = []
        sampled_operations = self.sample_operations()
        for sampled_operation in sampled_operations:
            record = sampled_operation.sample_record(self)[0]
            payload = self.records_dict[record]
            if sampled_operation.type == C.UPDATE:
                payload = self.generate_record_payload()
                self.records_dict[record] = payload
            run_phase.append([None, sampled_operation.type, record, payload])
            self.update_progress()
        self.run_phase_df = pd.DataFrame(run_phase, columns=AdvancedSimulator.DF_COLUMNS)
        self.run_phase_df[C.TIMESTAMP] = pd.to_datetime(self.generate_traffic())

        # gold TTLs
        y_true = self.generate_y_true(self.run_phase_df)
        self.run_phase_df = self.run_phase_df.merge(
            y_true[y_true.columns],
            on=[C.TIMESTAMP, C.RECORD],
            how='left',
            validate='many_to_one',
        )
        # TODO: may not be needed
        self.y_true_json = self.generate_y_true_json(self.run_phase_df)

        # features
        y_true_values = np.array(self.run_phase_df[C.Y_TRUE])
        X = np.column_stack([
            FeatureGenerator.generate_feature(y_true_values, 'linear'),
            # FeatureGenerator.generate_feature(
            #     y_true_values, 'linear', min_value=100, max_value=1_500, weight=-1
            # ),
            # FeatureGenerator.generate_feature(
            #     y_true_values, 'quadratic', min_value=100, max_value=1_500
            # ),
            # FeatureGenerator.generate_feature(
            #     y_true_values, 'quadratic', min_value=100, max_value=1_500, weight=-1
            # ),
            # FeatureGenerator.generate_feature(
            #     y_true_values, 'logarithmic', min_value=100, max_value=1_500
            # ),
            # FeatureGenerator.generate_feature(
            #     y_true_values, 'logarithmic', min_value=100, max_value=1_500, weight=-1
            # ),
            # FeatureGenerator.generate_feature(
            #     y_true_values, 'random', min_value=10, max_value=20
            # ),
        ])
        self.run_phase_df[C.X] = list(np.round(X, 2))

        self.features_df = pd.DataFrame(np.array(self.run_phase_df[C.X].tolist()))
        self.features_df[C.Y_TRUE] = self.run_phase_df[C.Y_TRUE]


class CustomAdvancedSimulator(AdvancedSimulator):

    def sample_operations(self):
        return TrafficGenerator.sample_operations(
            count=self.operations_count,
            probs=[0.7, 0.3],
            operations=[
                TrafficGenerator.Operation(C.READ, lambda self: RecordSampler.zipf(self.record_keys, alpha=1.5)),
                TrafficGenerator.Operation(C.DATA_CHANGED, lambda self: RecordSampler.periodic_zipf(
                    self.record_keys,
                    alpha=2.0,
                    periods_count=2,
                    progress=self.get_progress()
                ))
            ])

    def generate_traffic(self):
        return TrafficGenerator.generate_traffic(
            count=self.operations_count,
            start='01-01-2024 00:00:00',
            end='01-01-2024 01:00:00',
            freq='s'
        )
