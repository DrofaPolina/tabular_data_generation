from ctgan_data_generation_BB import Generator


Generator = Generator('ctgan', './datasets', 'small_dataset.csv', 'issue_d')
rounds = Generator.generate_rounds(n_rounds=1, rate=1)

