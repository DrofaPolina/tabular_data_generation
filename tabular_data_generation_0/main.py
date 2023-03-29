from tabular_data_generation_0 import Generator

Generator = Generator('ctgan', './datasets', 'small_dataset.csv', 'issue_d')
Generator.generate_rounds(n_rounds=10, rate=1)