import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

class ReadWrite:
    """
    Inside the data dir their are multiple possible sub_dirs.
    To ensure we don't run into errors we allow sub_dir as an input
    """
    def __init__(self):
        self.base_path = '../data'
        
    def read_file(self, file, sub_dir=""):
        table = pq.read_table(self.base_path + sub_dir + file)
        df = table.to_pandas()
        return df

    def write_file(self, file, data, sub_dir=""):
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, 'dataframe.parquet', compression='snappy')