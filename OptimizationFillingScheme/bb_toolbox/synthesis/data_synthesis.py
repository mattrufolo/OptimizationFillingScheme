import numpy as np
from numpy import linalg as LA
import pandas as pd
import copy
import json
from matplotlib import pyplot as plt
from bb_toolbox.analysis import tool_box_to_bool as tbtb
from bb_toolbox.analysis import bb_tool_box as bbt
import random




def prefill_algorithm_NOINDIV_ALICE(bunches,empty,len_batch, twelve = 12, inj_twelve = 12, inj_space = 31):
    '''
    Having as input the number of bunches for each PS_batch, the number of empties for the empty gap
    between PS batches, and how many PS_batches there are inside an SPS_batch, this script gives as output
    the two filling scheme of the two beams with the desired constraints given in input, with the number of collisions 
    in ATLAS/CMS/ALICE maximized.
    Hypothesys:
        - in this script we are considering that the "twelve" bunches for one of the two beams 
        are positioned in 0 and for the other in inj_twelve given as input
        - we are also considering that the ABK is optimized as best posssible in order to contatin perfectly the
        longest SPS batch, and it is completely filled
    Args:
        bunches: number of bunches for every PS_batch
        empty: number of empty slots for every empty gap between PS_batches
        len_batch: maximum of number of PS_batches for every SPS_batch
        twelve: how many bunches, for the first "twelve" bunches, set as 12
        inj_twelve: where to inject the "twelve" bunches of the detached beam, set as 12
        inj_space: number of empty slots fot every empty gap between SPS_batches
    Return: 
        np.array(collisions_max); a vector with the number of collisions using the desired filling scheme
        np.array(B1_max): a boolean vector that represent the slots filled by bunches for the desired beam1
        np.array(B2_max): a boolean vector that represent the slots filled by bunches for the desired beam2
    '''
    
    vec = [[]]
    vec[0] = np.zeros(bunches+inj_space)
    vec[0][0:bunches] = np.ones(bunches)
    collisions =np.array([0,0,0])
    collisions_max1 = np.array([0,0,0])
    collisions_new = np.array([0,0,0])
    B1_max1 = np.zeros(3564)
    B2_max1 = np.zeros(3564)
    B1_max2 = np.zeros(3564)
    B2_max2 = np.zeros(3564)
    B1_new2 = np.zeros(3564)
    B2_new2 = np.zeros(3564)
    B1_new = -np.ones(3564)
    B2_new = -np.ones(3564)
    B1 = -np.ones(3564)
    B2 = -np.ones(3564)
    # vector that represent the batches trains, in counts the inj space only at the end!!
    for j in np.arange(len_batch-1)+2:
        v= np.zeros((j)*(bunches+empty)+(inj_space-empty))
        v[(j)*(bunches+empty):(j)*(bunches+empty)+(inj_space-empty)] = np.zeros((inj_space-empty))
        for i in np.arange(j):
            v[(i)*(bunches+empty):(i)*(bunches+empty)+bunches] = np.ones(bunches)
        vec.append(v)
    
    
    # fill the initial part with the twelve bunches and the final part, considering the AB and ABK 
    B1[0:inj_twelve] = np.zeros(inj_twelve)
    B1[inj_twelve:inj_twelve+twelve] = np.ones(twelve)
    B1[inj_twelve+twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_space)
    B2[0:twelve] = np.ones(twelve)    
    B2[twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_twelve+inj_space)
    B1_new[0:inj_twelve] = np.zeros(inj_twelve)
    B1_new[inj_twelve:inj_twelve+twelve] = np.ones(twelve)
    B1_new[inj_twelve+twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_space)
    B2_new[0:twelve] = np.ones(twelve)    
    B2_new[twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_twelve+inj_space)
    slot = 3564-121-((len_batch-1)*(bunches+empty)+bunches)
    B1[slot:3564-121+inj_space] = vec[len_batch-1]
    B2[slot:3564-121+inj_space] = vec[len_batch-1]
    slot = 3564-121-2*((len_batch-1)*(bunches+empty)+bunches)-31
    #fill the most possible the filling scheme with the biggest trains respecting the maximum given as input by thr user
    i = 1
    while slot > 2673-121+31:
        B1[slot:slot+((len_batch-1)*(bunches+empty)+bunches)+inj_space] = vec[len_batch-1]
        B2[slot:slot+((len_batch-1)*(bunches+empty)+bunches)+inj_space] = vec[len_batch-1]
        slot_New = slot
        i += 1
        slot = 3564-120-(i+1)*((len_batch-1)*(bunches+empty)+bunches)-i*31-1
    
    #copy the last quarter in the first one
    #PS: Be careful in copying only the part of the last quarter that can fill after the 12 bunches in the first quarter
    if slot_New-2673>inj_twelve+twelve+inj_space:
        slot_New_pos = slot_New
    else:
        slot_New_pos = slot_New + (len_batch-1)*(bunches+empty)+(inj_space-empty)
    B1[slot_New_pos-2673:891-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
    B2[slot_New_pos-2673:891-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
    


    
    # fill the space in a clever way, either trying to fill the remaining empty space with the maximum train possible
    # but consider that losing some collisions, shifting these longest trains, you could insert a train tha contain one batch more, 
    # and this would lead to a better configuration possible for ATLAS/CMS

    # either here than (*) I use the flag because there are some specific approaches to lead to the best solution, and once done this
    # you wanna exit, because the other approaches would lead to worst solutions!!
    flag = 0
    for j in (len_batch-1)-np.arange(len_batch-1):
        frac = ((slot_New)-(2673)+(121)-(2*inj_space))/((j-1)*(bunches+empty)+bunches)
        # check if you can fill the empty space between quarters with a SPS batch of the maximum length possible
        if frac >1 and flag == 0:
            # check that shifting the longest trains of some slots, you could insert in the empty space between quarters a SPS batch
            # with a PS batch more, once entered in this shift, this situation is already preferred because you increase ATLAS/CMS...

            # POSSIBLE IMPROVEMENT: now you see if shifting of a quantity, you could increase the SPS batch by one, and then stop. But..
            # you could check also if shifting by >quantity and then position that SPS batch in different positions, considering also the SPS batch after
            # the twelve that you will insert later, if ALICE increases.
            max_number_shift = int((slot_New_pos-2673-(inj_twelve+twelve+inj_space))/3)
            check = (slot_New)-(2673)+(121)-(2*inj_space)+(1+np.arange(max_number_shift))>=(j*(bunches+empty)+bunches)

            if check.any():
                slot_shift = 1+np.arange(max_number_shift)[np.where(check)[0][0]]

                if (B1[slot_New-2673-3*slot_shift]==-1):
                    # here you are considering that in the different quarters there are different trains, so if the last train
                    # is shifted by a quantity "tot" so the train in the previous quarters is shifted by two times "tot" because he is shifted
                    # but is affected by the shift of the train ahead
                    init_slots_shift = [slot_New-2673-3*slot_shift,(891)-slot_shift*2,(1782)-slot_shift]
                    quarter_slot_shifted = (2673)-slot_shift-121+inj_space
                    B1_new[init_slots_shift[0]:891-3*slot_shift-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
                    B2_new[init_slots_shift[0]:891-3*slot_shift-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
                    B1_new[2673-121+inj_space:3564-121+31] = B1[2673-121+inj_space:3564-121+31]
                    B2_new[2673-121+inj_space:3564-121+31] = B2[2673-121+inj_space:3564-121+31]
                    B1_new[quarter_slot_shifted:quarter_slot_shifted+(j+1)*(bunches+empty)+(inj_space-empty)] = vec[j]
                    B2_new[quarter_slot_shifted:quarter_slot_shifted+(j+1)*(bunches+empty)+(inj_space-empty)] = vec[j]

                    for k in np.arange(2)+1:
                        B1_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k])+891-121+31] = B1_new[quarter_slot_shifted:3564-121+31]
                        B2_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k])+891-121+31] = B2_new[quarter_slot_shifted:3564-121+31]
                        
                    # (*)
                    # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                    # which one is the best possible in terms of collisions in ALICE and save it
                    flag_init = 0
                    # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                    # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                    # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                    # is "half" of the spaces considered before
                    for l in j-np.arange(j):
                        initial_frac = (init_slots_shift[0]-inj_twelve-twelve-(2*inj_space))/((l-1)*(bunches+empty)+bunches)
                        if initial_frac >=1 and flag_init == 0:
                            flag_init = 1
                            B1_check = copy.copy(B1_new)
                            B2_check = copy.copy(B2_new)
                            for k in np.arange(init_slots_shift[0]-inj_space-(inj_twelve+twelve+inj_space))+inj_twelve+twelve+inj_space:
                                indec = k
                                end_indec = indec+(l-1)*(bunches+empty)+bunches
                                if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(init_slots_shift[0]-inj_space):
                                    B1_new[indec:end_indec+inj_space] = vec[l-1]
                                    B2_new[indec:end_indec+inj_space] = vec[l-1]
                                    B1_new[np.where(B1_new==-1)] = 0
                                    B2_new[np.where(B2_new==-1)] = 0
                                    collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                                    print(collisions_new)
                                    if collisions_new[1]>collisions_max1[1]:
                                        collisions_max1 = collisions_new
                                        vec_B1 = copy.copy(B1_new)
                                        vec_B2 = copy.copy(B2_new)
                                    B1_new = copy.copy(B1_check)
                                    B2_new = copy.copy(B2_check)
                            B1_max1 = vec_B1
                            B2_max1 = vec_B2  
                    if flag_init == 0:
                        B1_new[np.where(B1_new==-1)] = 0
                        B2_new[np.where(B2_new==-1)] = 0
                        collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                        collisions_max1 = collisions_new
                        B1_max1 = copy.copy(B1_new)
                        B2_max1 = copy.copy(B2_new)        
                    # in this configuration you have maximized ATLAS/CMS more than all the other possible solutions, and also with that number 
                    # of collisions of ATLAS/CMS you have been checked
                    flag = 2

            
            if flag != 2:
                # now you couldn't increase the number of PS batch inside the SPS batch, so we maximize the the number of collisions in ALICE trying to
                # insert the SPS batch in the best position possible!
                gap_frac = (slot_New+1-2673+121-2*inj_space)/((j-1)*(bunches+empty)+bunches)
                flag_gap = 1
                if gap_frac>=1 and flag_gap ==1:
                    flag_gap = 1
                    B1_new2[:] = copy.copy(B1[:])
                    B2_new2[:] = copy.copy(B2[:])
                    for k in np.arange(slot_New+1-inj_space-(2673-121+inj_space))+2673-121+inj_space:
                        indec = k
                        end_indec = indec+j*(bunches+empty)+(inj_space-empty)
                        if (B1_new2[indec] == -1) and (B1_new2[end_indec] == -1):
                            B1_new2[indec:indec+j*(bunches+empty)+(inj_space-empty)] = vec[j-1]
                            B2_new2[indec:indec+j*(bunches+empty)+(inj_space-empty)] = vec[j-1]
                        # we are maximizing ALICE, so we are copying and pasting the SPS batches perfectly on quarters of all the ring
                        for l in [891,1782]:
                            B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
                            B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]

                        # (*)
                        # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                        # which one is the best possible in terms of collisions in ALICE and save it
                        flag_init = 0
                        # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                        # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                        # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                        # is "half" of the spaces considered before
                        for l in j-np.arange(j):
                            initial_frac = (slot_New_pos-2673-inj_twelve-twelve-(2*inj_space))/((l-1)*(bunches+empty)+bunches)
                            if initial_frac >=1 and flag_init == 0:
                                flag_init = 1
                                B1_check = copy.copy(B1_new2)
                                B2_check = copy.copy(B2_new2)
                                for k in np.arange(slot_New_pos-2673-inj_space-(inj_twelve+twelve+inj_space))+inj_twelve+twelve+inj_space:
                                    indec = k
                                    end_indec = indec+(l-1)*(bunches+empty)+bunches
                                    if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-inj_space):
                                        B1_new2[indec:end_indec+inj_space] = vec[l-1]
                                        B2_new2[indec:end_indec+inj_space] = vec[l-1]
                                        B1_new2[np.where(B1_new2==-1)] = 0
                                        B2_new2[np.where(B2_new2==-1)] = 0
                                        collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                                        if collisions[1]>collisions_max1[1]:
                                            collisions_max1 = collisions
                                            vec_B1 = copy.copy(B1_new2)
                                            vec_B2 = copy.copy(B2_new2)
                                        B1_new2 = copy.copy(B1_check)
                                        B2_new2 = copy.copy(B2_check)
                                B1_max1 = vec_B1
                                B2_max1 = vec_B2  
                        if flag_init == 0:
                            B1_new2[np.where(B1_new2==-1)] = 0
                            B2_new2[np.where(B2_new2==-1)] = 0
                            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                            collisions_max1 = collisions
                            B1_max1 = copy.copy(B1_new2)
                            B2_max1 = copy.copy(B2_new2) 
                # in this for we are going from the longest SPS batch to the smaller ones, so once you checked that one batch can be inserted,
                # surely the following steps will be smaller so you would lose in number of collisions in ATLAS/CMS, so you use the flag in order to exit from the cycle
                flag = 1

    # this is the case where you couldn't insert any SPS batch between quarters
    if flag == 0:
        B1_new2[:] = B1[:]
        B2_new2[:] = B2[:]
        for l in [891,1782]:
            B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
            B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]
        
        # (*)
        # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
        # which one is the best possible in terms of collisions in ALICE and save it
        flag_init = 0
        # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
        # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
        # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
        # is "half" of the spaces considered before
        for l in j-np.arange(j):
            initial_frac = (slot_New_pos-2673-inj_twelve-twelve-(2*inj_space))/((l-1)*(bunches+empty)+bunches)
            if initial_frac >=1 and flag_init == 0:
                flag_init = 1
                B1_check = copy.copy(B1_new2)
                B2_check = copy.copy(B2_new2)
                for k in np.arange(slot_New_pos-2673-inj_space-(inj_twelve+twelve+inj_space))+inj_twelve+twelve+inj_space:
                    indec = k
                    end_indec = indec+(l-1)*(bunches+empty)+bunches
                    if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-inj_space):
                        B1_new2[indec:end_indec+inj_space] = vec[l-1]
                        B2_new2[indec:end_indec+inj_space] = vec[l-1]
                        B1_new2[np.where(B1_new2==-1)] = 0
                        B2_new2[np.where(B2_new2==-1)] = 0
                        collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                        if collisions[1]>collisions_max1[1]:
                            collisions_max1 = collisions
                            vec_B1 = copy.copy(B1_new2)
                            vec_B2 = copy.copy(B2_new2)
                        B1_new2 = copy.copy(B1_check)
                        B2_new2 = copy.copy(B2_check)
                B1_max1 = vec_B1
                B2_max1 = vec_B2  
        if flag_init == 0:
            B1_new2[np.where(B1_new2==-1)] = 0
            B2_new2[np.where(B2_new2==-1)] = 0
            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
            collisions_max1 = collisions
            B1_max1 = copy.copy(B1_new2)
            B2_max1 = copy.copy(B2_new2) 

    # save the best solution possible considering the twelve bunches detached from 0 for both the beams, and see wich configuration is the best possible
    B1_max2 = copy.copy(B1_max1)
    B2_max2 = copy.copy(B2_max1)
    B2_max2[0:inj_twelve] = np.zeros(inj_twelve)
    B2_max2[inj_twelve:inj_twelve+twelve] = np.ones(twelve)
    B2_max2[inj_twelve+twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_space)
    B1_max2[0:twelve] = np.ones(twelve)    
    B1_max2[twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_twelve+inj_space)
    collisions_max2 = tbtb.Head_on(B1_max2,B2_max2,np.array([0,891,2670]))

    if collisions_max1[1]>collisions_max2[1]:
        B1_max = B1_max1
        B2_max = B2_max1
        collisions_max = collisions_max1
        print("prefered B1") 
    else:
        B1_max = B1_max2
        B2_max = B2_max2
        collisions_max = collisions_max2
        print("prefered B2") 
    print(collisions_max1)
    print(collisions_max2)
    return collisions_max,B1_max,B2_max

def prefill_algorithm_NOINDIV_LHCB(bunches,empty,len_batch, twelve = 12, inj_twelve = 12, inj_space = 31):
    '''
    Having as input the number of bunches for each PS_batch, the number of empties for the empty gap
    between PS batches, and how many PS_batches there are inside an SPS_batch, this script gives as output
    the two filling scheme of the two beams with the desired constraints given in input, with the number of collisions 
    in ATLAS/CMS/LHCb maximized.
    Hypothesys:
        - in this script we are considering that the "twelve" bunches for one of the two beams 
        are positioned in 0 and for the other in inj_twelve given as input
        - we are also considering that the ABK is optimized as best posssible in order to contatin perfectly the
        longest SPS batch, and it is completely filled
    Args:
        bunches: number of bunches for every PS_batch
        empty: number of empty slots for every empty gap between PS_batches
        len_batch: number of PS_batches for every SPS_batch
        twelve: how many bunches, for the first "twelve" bunches, set as 12
        inj_twelve: where to inject the "twelve" bunches of the detached beam, set as 12
        inj_space: number of empty slots fot every empty gap between SPS_batches
    Return: 
        np.array(collisions_max); a vector with the number of collisions using the desired filling scheme
        np.array(B1_max): a boolean vector that represent the slots filled by bunches for the desired beam1
        np.array(B2_max): a boolean vector that represent the slots filled by bunches for the desired beam2
    '''
    vec = [[]]
    vec = [[]]
    vec[0] = np.zeros(bunches+inj_space)
    vec[0][0:bunches] = np.ones(bunches)
    collisions =np.array([0,0,0])
    collisions_max1 = np.array([0,0,0])
    collisions_new = np.array([0,0,0])
    B1_max1 = np.zeros(3564)
    B2_max1 = np.zeros(3564)
    B1_max2 = np.zeros(3564)
    B2_max2 = np.zeros(3564)
    B1_new2 = np.zeros(3564)
    B2_new2 = np.zeros(3564)
    B1_new = -np.ones(3564)
    B2_new = -np.ones(3564)
    B1 = -np.ones(3564)
    B2 = -np.ones(3564)
    # vector that represent the batches trains, in counts the inj space only at the end!!
    for j in np.arange(len_batch-1)+2:
        v= np.zeros((j)*(bunches+empty)+(inj_space-empty))
        v[(j)*(bunches+empty):(j)*(bunches+empty)+(inj_space-empty)] = np.zeros((inj_space-empty))
        for i in np.arange(j):
            v[(i)*(bunches+empty):(i)*(bunches+empty)+bunches] = np.ones(bunches)
        vec.append(v)
    
    # fill the initial part with the twelve bunches and the final part, considering the AB and ABK 
    B1[0:inj_twelve] = np.zeros(inj_twelve)
    B1[inj_twelve:inj_twelve+twelve] = np.ones(twelve)
    B1[inj_twelve+twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_space)
    B2[0:twelve] = np.ones(twelve)    
    B2[twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_twelve+inj_space)
    B1_new[0:inj_twelve] = np.zeros(inj_twelve)
    B1_new[inj_twelve:inj_twelve+twelve] = np.ones(twelve)
    B1_new[inj_twelve+twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_space)
    B2_new[0:twelve] = np.ones(twelve)    
    B2_new[twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_twelve+inj_space)
    slot = 3564-121-((len_batch-1)*(bunches+empty)+bunches)
    B1[slot:3564-121+inj_space] = vec[len_batch-1]
    B2[slot:3564-121+inj_space] = vec[len_batch-1]
    slot = 3564-121-2*((len_batch-1)*(bunches+empty)+bunches)-31
    #fill the most possible the filling scheme with the biggest trains respecting the maximum given as input by thr user
    i = 1
    while slot > 2673-121+31:
        B1[slot:slot+((len_batch-1)*(bunches+empty)+bunches)+inj_space] = vec[len_batch-1]
        B2[slot:slot+((len_batch-1)*(bunches+empty)+bunches)+inj_space] = vec[len_batch-1]
        slot_New = slot
        i += 1
        slot = 3564-120-(i+1)*((len_batch-1)*(bunches+empty)+bunches)-i*31-1
    
    #copy the last quarter in the first one (shifting of 3 times of 3 slots (in order to optimize LHCb))
    #PS: Be careful in copying only the part of the last quarter that can fill after the 12 bunches in the first quarter
    if slot_New-2673-9>inj_twelve+twelve+inj_space:
        slot_New_pos = slot_New
    else:
        slot_New_pos = slot_New + j*(bunches+empty)+(inj_space-empty)
    B1[slot_New_pos-2673-9:891-9-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
    B2[slot_New_pos-2673-9:891-9-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
    


    # fill the space in a clever way, either trying to fill the remaining empty space with the maximum train possible
    # but consider that losing some collisions in LHCb, shifting these longest trains, you could insert a train tha contain one batch more, 
    # and this would lead to a better configuration possible for ATLAS/CMS

    # either here than (*) I use the flag because there are some specific approaches to lead to the best solution, and once done this
    # you wanna exit, because the other approaches would lead to worst solutions!!
    flag = 0
    for j in (len_batch-1)-np.arange(len_batch-1):
        frac = ((slot_New)-(2673-3)+(121)-(2*inj_space))/((j-1)*(bunches+empty)+bunches)
        # check if you can fill the empty space between quarters with a SPS batch of the maximum length possible
        if frac >1 and flag == 0:
            # check that shifting the longest trains of some slots, you could insert in the empty space between quarters a SPS batch
            # with a PS batch more, once entered in this shift, this situation is already preferred because you increase ATLAS/CMS...
            # we are doing the check using 2673-3 because the trains in the 3 quarter are shifted automatically by three slots on left, so 
            # there is more space 

            # POSSIBLE IMPROVEMENT: now you see if shifting of a quantity, you could increase the SPS batch by one, and then stop. But..
            # you could check also if shifting by >quantity and then position that SPS batch in different positions, considering also the SPS batch after
            # the twelve that you will insert later, if LHCb increases.
            max_number_shift = int((slot_New_pos-2673-9-(inj_twelve+twelve+inj_space))/3)
            check = (slot_New)-(2673-3)+(121)-(2*inj_space)+(1+np.arange(max_number_shift))>=(j*(bunches+empty)+bunches)

            if check.any():
                slot_shift = 1+np.arange(max_number_shift)[np.where(check)[0][0]]
                if (B1[slot_New-2673-9-3*slot_shift]==-1):
                    # here you are considering that in the different quarters there are different trains, so if the last train
                    # is shifted by a quantity "tot" so the train in the previous quarters is shifted by two times "tot" because he is shifted
                    # but is affected by the shift of the train ahead
                    # and you could see this shift of 3 slots that we talked before also here, always in order to maximize LHCb
                    init_slots_shift = [(slot_New-2673-9)-3*slot_shift,(885-3)-slot_shift*2,(1779-3)-slot_shift]
                    quarter_slot_shifted = (2673-3)-slot_shift-121+inj_space
                    B1_new[init_slots_shift[0]:891-9-3*slot_shift-121+inj_space] = B1[slot_New_pos:3564-121+inj_space]
                    B2_new[init_slots_shift[0]:891-9-3*slot_shift-121+inj_space] = B2[slot_New_pos:3564-121+inj_space]
                    B1_new[2673-121+inj_space:3564-121+31] = B1[2673-121+inj_space:3564-121+31]
                    B2_new[2673-121+inj_space:3564-121+31] = B2[2673-121+inj_space:3564-121+31]
                    B1_new[quarter_slot_shifted:quarter_slot_shifted+(j+1)*(bunches+empty)+(inj_space-empty)] = vec[j]
                    B2_new[quarter_slot_shifted:quarter_slot_shifted+(j+1)*(bunches+empty)+(inj_space-empty)] = vec[j]
                    for k in np.arange(2)+1:
                        # this 3 here, is ispired at the spaces which is shifted  the neirby quarter, before we subtracted 3 slots because the quarter on left
                        # was shifting 3 times more, but the trains that we were considering weren't shifted by these 3 slots so now I have to sum them again
                        B1_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k]+3)+891-121+31] = B1_new[quarter_slot_shifted:3564-121+31]
                        B2_new[init_slots_shift[k]-slot_shift-121+inj_space:(init_slots_shift[k]+3)+891-121+31] = B2_new[quarter_slot_shifted:3564-121+31]
                    

                    # (*)
                    # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                    # which one is the best possible in terms of collisions in ALICE and save it
                    flag_init = 0
                    # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                    # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                    # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                    # is "half" of the spaces considered before
                    for l in j-np.arange(j):
                        initial_frac = (init_slots_shift[0]-inj_twelve-twelve-(2*inj_space))/((l-1)*(bunches+empty)+bunches)
                        if initial_frac >=1 and flag_init == 0:
                            flag_init = 1
                            B1_check = copy.copy(B1_new)
                            B2_check = copy.copy(B2_new)
                            for k in np.arange(init_slots_shift[0]-inj_space-(inj_twelve+twelve+inj_space))+inj_twelve+twelve+inj_space:
                                indec = k
                                end_indec = indec+(l-1)*(bunches+empty)+bunches
                                if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(init_slots_shift[0]-inj_space):
                                    B1_new[indec:end_indec+inj_space] = vec[l-1]
                                    B2_new[indec:end_indec+inj_space] = vec[l-1]
                                    B1_new[np.where(B1_new==-1)] = 0
                                    B2_new[np.where(B2_new==-1)] = 0
                                    collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                                    print(collisions_new)
                                    if collisions_new[2]>collisions_max1[2]:
                                        collisions_max1 = collisions_new
                                        vec_B1 = copy.copy(B1_new)
                                        vec_B2 = copy.copy(B2_new)
                                    B1_new = copy.copy(B1_check)
                                    B2_new = copy.copy(B2_check)
                            B1_max1 = vec_B1
                            B2_max1 = vec_B2  
                    if flag_init == 0:
                        B1_new[np.where(B1_new==-1)] = 0
                        B2_new[np.where(B2_new==-1)] = 0
                        collisions_new = tbtb.Head_on(B1_new,B2_new,np.array([0,891,2670]))
                        collisions_max1 = collisions_new
                        B1_max1 = copy.copy(B1_new)
                        B2_max1 = copy.copy(B2_new)        
                    # in this configuration you have maximized ATLAS/CMS more than all the other possible solutions, and also with that number 
                    # of collisions of ATLAS/CMS you have been checked
                    flag = 2

            
            if flag != 2:
                # now you couldn't increase the number of PS batch inside the SPS batch, so we maximize the the number of collisions in LHCb trying to
                # insert the SPS batch in the best position possible!
                gap_frac = (slot_New+1-2673+121-2*inj_space)/((j-1)*(bunches+empty)+bunches)
                flag_gap = 1
                if gap_frac>=1 and flag_gap ==1:
                    flag_gap = 1
                    B1_new2[:] = copy.copy(B1[:])
                    B2_new2[:] = copy.copy(B2[:])
                    for k in np.arange(slot_New+1-inj_space-(2673-121+inj_space))+2673-121+inj_space:
                        indec = k
                        end_indec = indec+j*(bunches+empty)+(inj_space-empty)
                        if (B1_new2[indec] == -1) and (B1_new2[end_indec] == -1):
                            B1_new2[indec:indec+j*(bunches+empty)+(inj_space-empty)] = vec[j-1]
                            B2_new2[indec:indec+j*(bunches+empty)+(inj_space-empty)] = vec[j-1]
                        # we are maximizing LHCb, so we are copying and pasting the SPS batches not perfectly on quarters of all the ring, so 
                        # we are shifting all by three slots, and we are ading up all the shifts
                        for l in [885,1779]:
                            B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
                            B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]
                        
                        # (*)
                        # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
                        # which one is the best possible in terms of collisions in ALICE and save it
                        flag_init = 0
                        # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
                        # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
                        # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
                        # is "half" of the spaces considered before
                        for l in j-np.arange(j):
                            initial_frac = (slot_New_pos-2673-9-inj_twelve-twelve-(2*inj_space))/((l-1)*(bunches+empty)+bunches)
                            if initial_frac >=1 and flag_init == 0:
                                flag_init = 1
                                B1_check = copy.copy(B1_new2)
                                B2_check = copy.copy(B2_new2)
                                for k in np.arange(slot_New_pos-2673-9-inj_space-(inj_twelve+twelve+inj_space))+inj_twelve+twelve+inj_space:
                                    indec = k
                                    end_indec = indec+(l-1)*(bunches+empty)+bunches
                                    if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-9-inj_space):
                                        B1_new2[indec:end_indec+inj_space] = vec[l-1]
                                        B2_new2[indec:end_indec+inj_space] = vec[l-1]
                                        B1_new2[np.where(B1_new2==-1)] = 0
                                        B2_new2[np.where(B2_new2==-1)] = 0
                                        collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                                        if collisions[2]>collisions_max1[2]:
                                            collisions_max1 = collisions
                                            vec_B1 = copy.copy(B1_new2)
                                            vec_B2 = copy.copy(B2_new2)
                                        B1_new2 = copy.copy(B1_check)
                                        B2_new2 = copy.copy(B2_check)
                                B1_max1 = vec_B1
                                B2_max1 = vec_B2  
                        if flag_init == 0:
                            B1_new2[np.where(B1_new2==-1)] = 0
                            B2_new2[np.where(B2_new2==-1)] = 0
                            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                            collisions_max1 = collisions
                            B1_max1 = copy.copy(B1_new2)
                            B2_max1 = copy.copy(B2_new2) 
                # in this for we are going from the longest SPS batch to the smaller ones, so once you checked that one batch can be inserted,
                # surely the following steps will be smaller so you would lose in number of collisions in ATLAS/CMS, so you use the flag in order to exit from the cycle
                flag = 1
    
    # this is the case where you couldn't insert any SPS batch between quarters
    if flag == 0:
        B1_new2[:] = B1[:]
        B2_new2[:] = B2[:]
        for l in [885,1779]:
            B1_new2[l-121+inj_space:l+891-121+31] = B1_new2[2673-121+inj_space:3564-121+31]
            B2_new2[l-121+inj_space:l+891-121+31] = B2_new2[2673-121+inj_space:3564-121+31]
        
        # (*)
        # here we are filling the initial empty space, after the first twelve bunches, checking all the position available to see 
        # which one is the best possible in terms of collisions in ALICE and save it
        flag_init = 0
        # in the following cycle we are considering as maximum SPS batch the one inserted between quarters, would be useless do the cycle
        # for all length available because this is limited by the twelve bunches that there aren't in the other quarters and also becasue before 
        # we fill the empty space between quarters, here before 0 there is the ABORT GAP where must be neither one bunch, so the space that we can fill
        # is "half" of the spaces considered before
        for l in j-np.arange(j):
            initial_frac = (slot_New_pos-2673-9-inj_twelve-twelve-(2*inj_space))/((l-1)*(bunches+empty)+bunches)
            if initial_frac >=1 and flag_init == 0:
                flag_init = 1
                B1_check = copy.copy(B1_new2)
                B2_check = copy.copy(B2_new2)
                for k in np.arange(slot_New_pos-2673-9-inj_space-(inj_twelve+twelve+inj_space))+inj_twelve+twelve+inj_space:
                    indec = k
                    end_indec = indec+(l-1)*(bunches+empty)+bunches
                    if (B1_check[indec] == -1) and (B1_check[end_indec] == -1) and end_indec<=(slot_New_pos-2673-9-inj_space):
                        B1_new2[indec:end_indec+inj_space] = vec[l-1]
                        B2_new2[indec:end_indec+inj_space] = vec[l-1]
                        B1_new2[np.where(B1_new2==-1)] = 0
                        B2_new2[np.where(B2_new2==-1)] = 0
                        collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
                        if collisions[1]>collisions_max1[1]:
                            collisions_max1 = collisions
                            vec_B1 = copy.copy(B1_new2)
                            vec_B2 = copy.copy(B2_new2)
                        B1_new2 = copy.copy(B1_check)
                        B2_new2 = copy.copy(B2_check)
                B1_max1 = vec_B1
                B2_max1 = vec_B2  
        if flag_init == 0:
            B1_new2[np.where(B1_new2==-1)] = 0
            B2_new2[np.where(B2_new2==-1)] = 0
            collisions = tbtb.Head_on(B1_new2,B2_new2,np.array([0,891,2670]))
            collisions_max1 = collisions
            B1_max1 = copy.copy(B1_new2)
            B2_max1 = copy.copy(B2_new2) 

    # save the best solution possible considering the twelve bunches detached from 0 for both the beams, and see wich configuration is the best possible
    B1_max2 = copy.copy(B1_max1)
    B2_max2 = copy.copy(B2_max1)
    B2_max2[0:inj_twelve] = np.zeros(inj_twelve)
    B2_max2[inj_twelve:inj_twelve+twelve] = np.ones(twelve)
    B2_max2[inj_twelve+twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_space)
    B1_max2[0:twelve] = np.ones(twelve)    
    B1_max2[twelve:inj_twelve+twelve+inj_space] = np.zeros(inj_twelve+inj_space)
    collisions_max2 = tbtb.Head_on(B1_max2,B2_max2,np.array([0,891,2670]))

    if collisions_max1[2]>collisions_max2[2]:
        B1_max = B1_max1
        B2_max = B2_max1
        collisions_max = collisions_max1 
        print("prefered B1") 
    else:
        B1_max = B1_max2
        B2_max = B2_max2
        collisions_max = collisions_max2
        print("prefered B2") 
    print(collisions_max)
    return collisions_max,B1_max,B2_max


#@profile
def MC_shift(slots_to_shift, slots_init, beam, n_sample, bunches, slots_to_shift_check, empties = 7, twelve = 0, ABK = 0):
    '''
    a MC method to shift the trains in order to get a configuration with more collisions in LHCb or ALICE, the shift will done of 1 slot 
    singularly at every train, and the script will save all the configurations found
    Hypothesis:
        - considering that the wo beams are so similar, in "slots_to_shift" are represented the (first slot-1) and the (last slot +1) for every SPS_batch, only 
        for one of thetwo beams, which one is represented in "beam". The information of the position of the "twelve" bunches fot the other beam is represented in "slots_init"
        - for sake of semplicity we will call B1 as the beam that is represented by slots_to_shift
    Args:
        slots_to_shift: positions of the (first element -1) and (last element+1) of every SPS_batch for B1
        slots_init: initial position of the twelve bunches of the other beam (B2)
        beam: give the number of the beam connected with slots_to_shift (can be deleted)
        bunches: number of bunches for every PS batch (can be computed I think)
        slots_to_shift_check: the beam with whom compare the distance of MC simulation
        empties: number of empty slots for every empty gap between PS_batches, set as 7
        twelve: where  the "twelve" bunches of the detached beam, set as 0 (is this slots_to_shift-11?) control see if works
        ABK: free space before the Abort Gap that has the beam in case is not perfectly positioned in the Abort Gap Keeper
    Return: 
        pd.DataFrame: that contains the slots_to_shift, distance wrt slots_to_shift_check, slots_init, beam, events_in_ALICE, events_in_LHCb
        of all the MC simulation
    '''

    #vector that contains the number of remaining slots in empty gap that are more than 800ns 
    B1_more800ns = []
    index_INDIV = []

    B1 = np.zeros(3564)
    B1_more800ns = np.zeros(len(slots_to_shift))
    B1_more800ns[0] = -(slots_to_shift[0]-11)
    B1_more800ns[-1] = ABK

    #computation of the two beams and the B1_more800ns
    for i in np.arange(int((len(slots_to_shift)-2)/2))+1:
        more800_ns = slots_to_shift[2*i]-slots_to_shift[2*i-1]-32+3
        B1_more800ns[2*i-1] = more800_ns
        B1_more800ns[2*i] = -more800_ns
        slot = slots_to_shift[2*i-2]+1
        flag = 0
        while flag == 0:
            if slots_to_shift[2*i-1]-slots_to_shift[2*i-2] == 3:
                index_INDIV = np.append(index_INDIV, 2*i-2)
                B1[slot] = 1
                flag = 1
            else:
                if 2*i-2 ==0:
                    B1[slot:slot+12] = np.ones(12)
                    flag = 1
                else:
                    B1[slot:slot+bunches] = np.ones(bunches)
                    slot += bunches+empties
                    if (slot>=slots_to_shift[2*i-1]-1+empties):
                        flag = 1
    flag = 0
    slot = slots_to_shift[-2]+1
    while flag == 0:
        B1[slot:slot+bunches] = np.ones(bunches)
        slot += bunches+empties
        if (slot>=slots_to_shift[-1]-1+empties):
            flag = 1
    B2 = copy.copy(B1)
    init_B2 = np.zeros(slots_to_shift[2]+1)
    init_B2[slots_init:slots_init+12] = np.ones(12)
    B2[:slots_to_shift[2]+1] = init_B2


    # MC method, you compute all the shifts that are possible, in order to have an improvement on the number of collisions
    vector = np.zeros(100*n_sample)
    # list that contains all the interesting information of the MonteCarlo
    list_informations= [ii for ii in vector]
    informations = np.append(slots_to_shift,[tbtb.events_in_IPN(B1,B2,'IP8')[1],tbtb.events_in_IPN(B1,B2,'IP2')[1],slots_init,beam])
    list_informations[0] = [ii for ii in informations]

    
    #take the informations of how many possible samples of the MonteCarlo can be considered
    count = 1
    if len(B1_more800ns>0):
        
        #repeat 100 times the same MC simulation with the same number of possible shifts
        for index in np.arange(100):
            # copy in order to not overwrite the initial data
            B1_new = copy.copy(B1)
            B2_new = copy.copy(B2)
            B1_more800ns_new = copy.copy(B1_more800ns)
            slots_to_shift_new = copy.copy(slots_to_shift)
            # - (how many slots it is detached the first of beam1)
            init_shift_B1 = -slots_to_shift[0]
            for iter in np.arange(n_sample):
                # random choice of the position where have the shift (could be optimized this, sth to do)
                pos_shift = np.random.randint(len(B1_more800ns_new))
                
            

                # flag that in cases you cannot move all the INDIVs it surpass this step, and didn't move anything
                exit_flag_INDIV = 0
                if B1_more800ns_new[pos_shift]!=0:
                    count += 1
                    #shift value is the sign of the possible movement of the SPS batch
                    shift_value = int(np.sign(B1_more800ns_new[pos_shift]))
                    # if the pos shift is odd it means that the shift of SPS batch will be on right
                    if pos_shift%2 == 1:
                        if pos_shift-1 in index_INDIV:
                            # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                            # Pick all the INDIV except the pos_shift of the first INDIV
                            vec_INDIV = index_INDIV[~(pos_shift == index_INDIV+1)]+1
                            vec_INDIV = [int(i) for i in vec_INDIV]
                            #check if the movements are available
                            if np.array([B1_more800ns_new[i] !=0 for i in vec_INDIV]).all():
                                # do the shift for every indiv with except of the pos_shift that we'll do later
                                # and change all the information of the slots_to_shift_new
                                for i in vec_INDIV:
                                    pos_shift_2 = i
                                    
                                    B1_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]] = np.roll(B1_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]],shift_value)
                                    B2_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]] = np.roll(B2_new[slots_to_shift_new[pos_shift_2-1]:slots_to_shift_new[pos_shift_2]],shift_value)
                                    slots_to_shift_new[pos_shift_2-1] += shift_value
                                    B1_more800ns_new[pos_shift_2] -= shift_value
                                    B1_more800ns_new[pos_shift_2-1] -= shift_value
                                    if pos_shift_2!=len(B1_more800ns_new)-1:
                                        B1_more800ns_new[pos_shift_2+shift_value] += shift_value
                                    if pos_shift_2 != 1:
                                        B1_more800ns_new[pos_shift_2-2] += shift_value
                                    slots_to_shift_new[pos_shift_2] += shift_value
                                    
                                    
                            else:
                                #if you cannot change the INDIV
                                exit_flag_INDIV = 1

                        #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs
                        if exit_flag_INDIV == 0:

                            #shift of B1
                            B1_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]] = np.roll(B1_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]],shift_value)
                            if pos_shift != 1:
                                # shift of B2 if we are not moving the first 12 bunches
                                B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]] = np.roll(B2_new[slots_to_shift_new[pos_shift-1]:slots_to_shift_new[pos_shift]],shift_value)
                                B1_more800ns_new[pos_shift-2] += shift_value
                            else:
                                #change of parameters 
                                init_shift_B1 -= shift_value
                                #change of slots_init for the B2 chosen randomly among the positions available after the shift of B1
                                if B1_more800ns_new[1]>=12:
                                    c = np.random.randint(2)
                                    slots_init = np.random.randint([0,slots_to_shift_new[1]+shift_value+1],[slots_to_shift_new[0]-10+shift_value,slots_to_shift_new[2]-10])[c]
                                else:
                                    slots_init = np.random.randint([0],[slots_to_shift_new[0]-10+shift_value])[0]
                                
                                init_B2 = np.zeros(slots_to_shift_new[2]+1)
                                init_B2[slots_init:slots_init+12] = np.ones(12)
                                B2_new[:slots_to_shift_new[2]+1] = init_B2

                            #change of all the parameters
                            slots_to_shift_new[pos_shift-1] += shift_value
                            B1_more800ns_new[pos_shift] -= shift_value
                            B1_more800ns_new[pos_shift-1] -= shift_value
                            if pos_shift!=len(B1_more800ns_new)-1:
                                B1_more800ns_new[pos_shift+shift_value] += shift_value
                            slots_to_shift_new[pos_shift] += shift_value
                    # shift of SPS batch will be on right, we will do the same passages but considering the opposite direction
                    else:
                        if pos_shift in index_INDIV:
                            # I'm moving all the INDIVs at the same time in order to not affect the number of collisions
                            # Pick all the INDIV except the pos_shift of the first INDIV
                            vec_INDIV = index_INDIV[~(pos_shift == index_INDIV)]
                            vec_INDIV = [int(i) for i in vec_INDIV]
                            #check if the movements are available
                            if np.array([B1_more800ns_new[i] !=0 for i in vec_INDIV]).all():
                                # do the shift for every indiv with except of the pos_shift that we'll do later
                                # and change all the information of the slots_to_shift_new
                                for i in vec_INDIV:
                                    pos_shift_2 = i
                                    
                                    B1_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]] = np.roll(B1_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]],shift_value)
                                    B2_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]] = np.roll(B2_new[slots_to_shift_new[pos_shift_2]:slots_to_shift_new[pos_shift_2+1]],shift_value)
                                    
                                    slots_to_shift_new[pos_shift_2+1] += shift_value
                                    B1_more800ns_new[pos_shift_2] -= shift_value
                                    B1_more800ns_new[pos_shift_2+1] -= shift_value
                                    if pos_shift_2!=len(B1_more800ns_new)-2:
                                        B1_more800ns_new[pos_shift_2+2] += shift_value 
                                    if pos_shift_2 !=0:   
                                        B1_more800ns_new[pos_shift_2+shift_value] += shift_value
                                    slots_to_shift_new[pos_shift_2] += shift_value
                                  
                            else:
                                #if you cannot change the INDIV
                                exit_flag_INDIV = 1
                        #if we haven't pick the INDIV in pos_shift and also if we picked it, we could move all the other INDIVs        
                        if exit_flag_INDIV == 0:
                            #shift of B1
                            B1_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]] = np.roll(B1_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]],shift_value)
                            if pos_shift !=0:
                                # shift of B2 if we are not moving the first 12 bunches
                                B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]] = np.roll(B2_new[slots_to_shift_new[pos_shift]:slots_to_shift_new[pos_shift+1]],shift_value)
                                B1_more800ns_new[pos_shift+shift_value] += shift_value
                            else:
                                #change of parameters 
                                init_shift_B1 -= shift_value
                                #change of slots_init for the B2 chosen randomly among the positions available after the shift of B1
                                if B1_more800ns_new[1]>=12:
                                    c = np.random.randint(2)
                                    slots_init = np.random.randint([0,slots_to_shift_new[1]+shift_value+1],[slots_to_shift_new[0]-10+shift_value,slots_to_shift_new[2]-10])[c]
                                else:
                                    
                                    slots_init = np.random.randint([0],[slots_to_shift_new[0]-10+shift_value])[0]
                                init_B2 = np.zeros(slots_to_shift_new[2]+1)
                                init_B2[slots_init:slots_init+12] = np.ones(12)
                                B2_new[:slots_to_shift_new[2]+1] = init_B2
                            #change of all the parameters
                            slots_to_shift_new[pos_shift+1] += shift_value
                            B1_more800ns_new[pos_shift] -= shift_value
                            B1_more800ns_new[pos_shift+1] -= shift_value
                            if pos_shift!=len(B1_more800ns_new)-2:
                                B1_more800ns_new[pos_shift+2] += shift_value    
                            slots_to_shift_new[pos_shift] += shift_value

                    #once done the shift save the information that you need in order to restart the MonteCarlo
                    events_LHCb_new = tbtb.events_in_IPN(B1_new,B2_new,'IP8')[1]
                    events_ALICE_new = tbtb.events_in_IPN(B1_new,B2_new,'IP2')[1]
                    informations_new = np.append(slots_to_shift_new,[events_LHCb_new,events_ALICE_new,slots_init,beam])
                    list_informations[count-1] = [ii for ii in informations_new]
            
        # put all the informations, sectioned, that we want in the DataFrame 
        del list_informations[count:]
        df_fill_schemes2 = pd.DataFrame(index = np.arange(count))
        df_fill_schemes2['informations'] = list_informations
        df_fill_schemes = pd.DataFrame(np.unique(df_fill_schemes2), columns=df_fill_schemes2.columns)
        df_help = pd.DataFrame(df_fill_schemes2["informations"].values.tolist())
        df_fill_schemes = pd.DataFrame(index = np.arange(len(df_help.index)))
        vec_string_slot = df_help[df_help.columns[0:len(slots_to_shift)]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        help_vec = [list(ii.split(" ")) for ii in vec_string_slot]
        df_fill_schemes['slots_to_shift'] = [[int(eval(ii)) for ii in i] for i in help_vec]
        df_fill_schemes['distance'] = [LA.norm(ii-slots_to_shift_check[0]) for ii in df_fill_schemes['slots_to_shift']]
        df_fill_schemes['events_LHCb'] = df_help.iloc[:,-4]
        df_fill_schemes['events_ALICE'] = df_help.iloc[:,-3]
        df_fill_schemes['slots_init'] = [int(ii) for ii in df_help.iloc[:,-2]]
        df_fill_schemes['beam'] = df_help.iloc[:,-1]

    return df_fill_schemes



def from_LPC_json_to_parquet(file_name, parquet_name = 'input'):
    '''
    Having as input the name of the json file downloaded from LPC, this function returns a parquet file, with the name given as input, that contains:
        slots_to_shift: positions of the (first element -1) and (last element+1) of every SPS_batch for the deteached beam
        slots_init: initial position of the twelve bunches of the other beam 
        beam: give the number of the beam connected with slots_to_shift 
        events_LHCb: number of collisions in LHCb for the filling scheme given as input
        events_ALICE: number of collisions in ALICE for the filling scheme given as input
    Args:
        file_name: name of the json file downloaded from LPC
        parquet_name: name of the parquet file that contains the information, set as 'input' considering to use that for the MonteCarlo simulation
    '''
    json_file = open(file_name)

    data_json = json.load(json_file)
    B1 = data_json['beam1']
    B2 = data_json['beam2']

    # in order to understand which one of the two beams are detached from 0, in case both beams are detached we pick B1, because there are no prefence
    # considering that after we used this parquet for a MonteCalrlo completely casual
    no_beam_detached = 0
    if B1[0] == 0:
        beam = 1
        BEAM = B1
    elif B2[0] == 0: 
        beam = 2
        BEAM = B2
    else:
        beam = 1
        BEAM = B1
        no_beam_detached = 1
    
    zeros = np.where(BEAM == np.zeros(3564))
    ones =  np.where(BEAM == np.ones(3564))
    counter = 1
    B_emptyspaces = []
    B_fullspaces = []
    B_spaces = []

    # computaion all the length of the empty spaces
    for i in np.arange(len(zeros[0])-1):
        if zeros[0][i+1] == zeros[0][i]+ 1:
            counter+=1
        else:
            if counter<31 and len(B_emptyspaces)>0:
                #saving the length of the empty gap between two PS batch
                len_empties = counter
                
            B_emptyspaces = np.append(B_emptyspaces,[counter])
            counter = 1
    if i == len(zeros[0])-2:
        B_emptyspaces = np.append(B_emptyspaces,[counter])

    counter = 1
    # computation  all the length of consecutive ones
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

    # merging the empty spaces and full spaces in a coherent ways
    for i in np.arange(len(B_fullspaces)):
        if no_beam_detached == 1:
            B_spaces = np.append(B_spaces,[B_fullspaces[i],B_emptyspaces[i]])
        else:
            B_spaces = np.append(B_spaces,[B_emptyspaces[i],B_fullspaces[i]])
        if i == len(B_fullspaces)-1:    
            B_spaces = np.append(B_spaces,[B_emptyspaces[i+1]])

    # computation of positions of the (first element -1) and (last element+1) of every SPS_batch from B_spaces
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

    # computation of the position of the first twelve for the beam in case at least one is detached and if the one chosen is B1
    if beam == 1 and no_beam_detached == 0:
        flag = 0
        other_beam = [1,2][np.where([1,2]!=beam*np.ones(2))[0][0]]
        zeros = np.where(data_json[f'beam{other_beam}'] == np.zeros(3564))
        if zeros[0][0] == 0:
            for i in np.arange(len(zeros[0])-1):
                if zeros[0][i+1] == zeros[0][i]+ 1 and flag == 0:
                    counter+=1
                else:
                    slots_init = counter
                    flag = 1
        else: 
            slots_init = 0
    elif beam == 2 or no_beam_detached == 1:
        slots_init = 0

    # creation of the parquet file, passing from a dictionary
    json_dict = {'0':{
        'slots_to_shift':slots_to_shift,
        'events_LHCb':tbtb.events_in_IPN(B1,B2,'IP8')[1],
        'events_ALICE':tbtb.events_in_IPN(B1,B2,'IP2')[1],
        'slots_init': slots_init,
        'beam' :beam
        }}
    df = pd.DataFrame.from_dict(json_dict,orient = 'index')
    df.to_parquet(f'{parquet_name}.parquet')



