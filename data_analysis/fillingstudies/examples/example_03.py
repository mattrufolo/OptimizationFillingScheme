#%%
from tools_box.synthesis import data_synthesis as ds
from tools_box.analysis import tool_box_to_bool as tbtb
import numpy as np
from matplotlib import pyplot as plt

n_sample = 300
bunches = 48
[B1,B2] = tbtb.filling_scheme_from_lpc_url("examples/fill_8148.json",8148)


if B1[0] == 0:
    beam = 1
    BEAM = B1
else: 
    beam = 2
    BEAM = B2

zeros = np.where(BEAM == np.zeros(3564))
ones =  np.where(BEAM == np.ones(3564))
counter = 1
B_emptyspaces = []
B_fullspaces = []
B_spaces = []
# the computaion of the interesting empty spaces, the ones that are bigger than 31 slots
for i in np.arange(len(zeros[0])-1):
    if zeros[0][i+1] == zeros[0][i]+ 1:
        counter+=1
    else:
        if counter<31 and len(B_emptyspaces)>0:
            len_empties = counter
            
        B_emptyspaces = np.append(B_emptyspaces,[counter])
        counter = 1
if i == len(zeros[0])-2:
    B_emptyspaces = np.append(B_emptyspaces,[counter])

counter = 1
# computation of ones
for i in np.arange(len(ones[0])-1):
    if ones[0][i+1] == ones[0][i]+ 1:
        counter+=1
    else:
        if counter>12:
            len_bunch = counter
            
        B_fullspaces = np.append(B_fullspaces,[ counter])
        counter = 1
if i == len(ones[0])-2:
    B_fullspaces = np.append(B_fullspaces,[counter])

for i in np.arange(len(B_fullspaces)):
    B_spaces = np.append(B_spaces,[B_emptyspaces[i],B_fullspaces[i]])
B_spaces = np.append(B_spaces,[B_emptyspaces[i+1]])

slots_to_shift = [B_spaces[0]-1,sum(B_spaces[:2])+1]
for i in np.arange(len(B_spaces)):
    if B_spaces[i] == len_empties and  B_spaces[i-2] !=len_empties:
        c = sum(B_spaces[:i-1])-1
    if B_spaces[i] == len_empties and B_spaces[i+2] !=len_empties: #and B1_spaces[i+2]!=121:
        d = sum(B_spaces[:i+2])+1
        slots_to_shift = np.append(slots_to_shift,[[c,d]])
    # in case there are INDIV
    if B_spaces[i] == 1:
        c = sum(B_spaces[:i])-1
        d = sum(B_spaces[:i+1])+1
        slots_to_shift = np.append(slots_to_shift,[c,d])
    # in case there are trains with a single batch
    if i%2 == 1 and B_spaces[i-1]>=31 and B_spaces[i+1]>=31 and B_spaces[i] == len_bunch:
        c = sum(B_spaces[:i])-1
        d = sum(B_spaces[:i+1])+1
        slots_to_shift = np.append(slots_to_shift,[c,d])
# it keepes the information of the slots that can be shifted for both the trains that face the same empty space
slots_to_shift = [int(ii) for ii in slots_to_shift]

#twelve bunches of B2 are set at 0!!
slots_init = 0

df = ds.MC_shift(slots_to_shift,slots_init,beam,n_sample,bunches,slots_to_shift)

events_ALICE = [tbtb.events_in_IPN(B1,B2,'IP2')[1]]
events_ALICE = np.append(events_ALICE, df['events_ALICE'])
events_LHCb = [tbtb.events_in_IPN(B1,B2,'IP8')[1]]
events_LHCb = np.append(events_LHCb, df['events_LHCb'])
plt.scatter(events_ALICE[1:],events_LHCb[1:], c = 'green', alpha = 0.2, label ='MC_simulation')
plt.scatter(events_ALICE[0],events_LHCb[0], c = 'blue', label ='initial_data')
plt.xlabel('n° of collisions in ALICE')
plt.ylabel('n° of collisions in LHCb')
plt.title('MC simulation')
plt.legend()
plt.savefig("images/MC_simulation.png", dpi = 200)
# %%

# %%
