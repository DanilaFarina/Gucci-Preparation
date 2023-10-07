# %
import pandas as pd
import numpy as np
from pathlib import Path
import sweetviz as sv
import utils

# %%
master_dir = Path.cwd()
data_dir = master_dir/'data'
# %
df = pd.read_csv(data_dir/'articles.csv')
# %%
# #! exploratory
my_report = sv.analyze(df)
my_report.show_html(filepath=master_dir/'reports/SWEETVIZ_REPORT.html')

#! topline clean
df = df.pipe(
    utils.clean_headers
).pipe(
    utils.clean_text, col='text_col'
)
