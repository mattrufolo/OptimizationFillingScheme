#%%

from tools_box.analysis import bb_tool_box as btb
import numpy as np

bool_B1  = np.zeros(3564)
bool_B2  = np.zeros(3564)

[df_B1,df_B2] = btb.bbschedule(bool_B1,bool_B2,20)

assert df_B1.shape[0] == 0
assert df_B2.shape[0] == 0
# %%
