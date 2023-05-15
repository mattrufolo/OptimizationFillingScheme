
#%%
from tools_box.analysis import bb_tool_box as btb
from tools_box.analysis import tool_box_to_bool as tbtb
import numpy as np
from matplotlib import pyplot as plt

[B1,B2] = tbtb.filling_scheme_from_lpc_url("/home/mrufolo/data_analysis/fillingstudies/examples/fill_8148.json",8148)
[df_B1,df_B2] = btb.bbschedule(B1,B2,20)

for i in [1,2]:
    if i == 1:
        bbs = df_B1
    else:
        bbs = df_B2
    fig1 = plt.figure(100 + i, figsize = (6.4*1.5, 1.6*4.8))
    ax1 = fig1.add_subplot(3,1,1)
    ax2 = fig1.add_subplot(3,1,2, sharex=ax1)
    ax3 = fig1.add_subplot(3,1,3, sharex=ax1)
    
    #print(bbs['# LR in IP1'])

    ax1.plot(bbs['# of LR in ATLAS/CMS'],'.',color = 'b')
    ax2.plot(bbs['# of LR in ALICE'],'.',color = 'r')
    ax3.plot(bbs['# of LR in LHCB'],'.',color = 'g')

    ax1.set_ylabel('N. LR in ATLAS/CMS')
    ax2.set_ylabel('N. LR in ALICE')
    ax3.set_ylabel('N. LR in LHCb')

    ax3.set_xlim(0,3564)
    ax3.set_xlabel('25 ns slot')

    ax1.grid(True,linestyle = ':')
    ax2.grid(True,linestyle = ':')
    ax3.grid(True,linestyle = ':')

    ax1.set_ylim(bottom = 0)
    ax2.set_ylim(bottom = 0)
    ax3.set_ylim(bottom = 0)
    
    fig1.subplots_adjust(left = .06, right = .96, top = .92)
    fig1.suptitle(f'B{i}_bb_schedule')
    fig1.savefig(f"images/B{i}_bb_summary.png", dpi = 200)

plt.show()


 # %%
