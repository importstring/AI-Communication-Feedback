import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os

class ReadWrite:
    """
    Inside the data dir their are multiple possible sub_dirs.
    To ensure we don't run into errors we allow sub_dir as an input
    """
    def __init__(self):
        self.base_path = '../data'
        
    def read_parquet(self, file, sub_dir=""):
        path = os.path.join(self.base_path, sub_dir, file)
        table = pq.read_table(path)
        df = table.to_pandas()
        return df

    def write_parquet(self, data="", file="", sub_dir=""):
        # Create directory if it doesn't exist
        output_dir = os.path.join(self.base_path, sub_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create full path
        filepath = os.path.join(output_dir, file)
        
        # Convert to DataFrame if not already
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table, filepath, compression='snappy'
        )
        return filepath