import pandas as pd
import os
from ctgan import CTGAN

class Generator:
    def __init__(self, model_type, out_dir='./datasets', in_dir='', datecolumn=None):
        self.model_type = model_type
        self.out_dir, self.in_dir = out_dir, in_dir
        self.datecolumn = datecolumn
        self.model = None
        self.source_data = None

    def generate_rounds(self, n_rounds=10, rate=1):
        strip_path = self.in_dir.rstrip('.csv')
        self.source_data = pd.read_csv(self.in_dir)
        if self.datecolumn:
            self.source_data[self.datecolumn] = pd.to_datetime(self.source_data[self.datecolumn])
            self.source_data = self.source_data.sort_values(by=self.datecolumn)

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        chunk_size = self.source_data.shape[0] // n_rounds

        if self.model_type == 'ctgan':
            self.model = CTGAN()

        for num_round in range(n_rounds):
            self.generate_save(num_round, chunk_size, strip_path, rate)

    def generate_save(self, num_round, chunk_size, strip_path, rate):
        first = num_round * chunk_size
        last = first + chunk_size

        chunk = self.source_data[first:last]
        disc_cols = chunk.select_dtypes(include=['object', 'datetime64']).columns
        self.model.fit(chunk, disc_cols)
        new_data = self.model.sample(chunk_size * rate)
        new_data = pd.concat([chunk, new_data], axis=0)
        new_data = new_data.sort_values(by=self.datecolumn)

        chunk_path = f'./{self.out_dir}/{strip_path}_round{num_round}.csv'
        new_data.to_csv(chunk_path)

