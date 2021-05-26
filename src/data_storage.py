from typing import Optional, List

import numpy as np
import pandas as pd
import csv

class EA_Experiment:
    folder = "csv"

    def __init__(self, file_name):
        all_path = self.folder + "/" + file_name + ".csv"
        self.file_name = all_path
        self.df = pd.DataFrame()

    def init_save(self):
        with open(self.file_name, 'w') as f:
            self.df.to_csv(f, '\t')

    def add_experiment(self, n: int, t: Optional[int], r: int, ds: List[int], id: int, iter: int):
        ln = len(ds)
        new_df = pd.DataFrame({"n": np.repeat(n, ln),
                               "t": np.repeat(t, ln),
                               "r": np.repeat(r, ln),
                               "d": ds,
                               "i": np.arange(ln),
                               "id": np.repeat(id, ln),
                               "iter": np.repeat(iter, ln),
                               })
        self.df.append(new_df)

    def save_to_file(self, n: int, t: Optional[int], r: int, ds: List[int], id: int, iter: int):
        ln = len(ds)
        new_df = pd.DataFrame({"n": np.repeat(n, ln),
                               "t": np.repeat(t, ln),
                               "r": np.repeat(r, ln),
                               "d": ds,
                               "i": np.arange(ln),
                               "id": np.repeat(id, ln),
                               "iter": np.repeat(iter, ln),
                               })
        with open(self.file_name, 'w') as f:
            new_df.to_csv(f)

    def save_files(self):
        with open(self.file_name, 'a') as f:
            self.df.to_csv(f, header=False)

    def get_averaged_ds(self, n: int, t: Optional[int], r: int):
        self.df[(self.df.n == n) & (self.df.t == t) & (self.df.r == r) & (self.df.r == r), 'i']

