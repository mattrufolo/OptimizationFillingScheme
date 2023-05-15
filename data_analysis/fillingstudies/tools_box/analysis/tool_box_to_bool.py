import numpy as np
import pandas as pd
import glob
import json

def pandas_index_localize(input_df : pd.DataFrame, inplace=True) -> pd.DataFrame:
    '''~
    Big change here: a function who acts on an input is VERY dangerous.
    This function takes as iput a DF with an index made of unix timestamps, and returns a DF
    with pandas Timestamps index.
    Args:
        input_df (pd.DataFrame) : input DF to be modified
    Return: 
        output_df (pd.DataFrame) : DF with new index
    '''
    if not inplace:
        output_df = input_df.copy()
        output_df.index = output_df.index.map(lambda x: pd.Timestamp(x).tz_localize('UTC'))
        return output_df
    else:
        input_df.index = input_df.index.map(lambda x: pd.Timestamp(x).tz_localize('UTC'))
        return input_df

def filled_bunches(bunches_intensity,threshold) -> np.array:
    '''~
    Create a boolean matrix, that represent where are the slots full
    for every timestamp
    Args:
        bunches_intensity: a pd.Series, that represent the intensity
        of every bunch for every timestamp
        treshold: treshold that the bunch intensity has to be larger
        to consider it different form 0
    Return: 
        np.array of boolean np.array that represent the slots
    '''
    filling_scheme = np.array(bunches_intensity)>threshold
    return filling_scheme



def indeces_b_mode(series_b_modes,b_mode_interested)-> np.array:
    '''~
    Using the series_b_modes that you have at the different ordered time, 
    you compute the indeces of the b_mode_interested in the fill
    Args:
        series_b_modes: pd.Series that represent all the b_mode using
        in the dataframe
        b_mode_interested: a string with the name of the b_mode interested
    Return: 
        a np.array that represent the indeces where the fill
        is in the b_mode_interested
    '''
    #used [0][0], because the first one is to obtain the vector and the second
    #one is for the first index
    return(np.where(series_b_modes == b_mode_interested)[0])

def events_in_IPN(filling_scheme_b1,filling_scheme_b2, IPN) -> np.array:
    '''~
    Using the boolean slots of beam1 and beam2, at the timestamp in which
    they have a maximum number of elements, you compute the number of 
    events that happen in IP numbered by N_IPC (seen from B1)
    Args:
        filling_scheme_b1,filling_scheme_b2: np.array of boolean np.array 
        that represent the slots filled for the beam1 and the beam2
        IPN: a string that tell you the IP in which you are interested in
    Return: 
         a boolean np.array that tells the slots where there will be events,
         and also a number that tells you how many are these slots
    '''
    Dict_IPN = {'IP1':0,'IP5':0,'IP8':2670,'IP2':891}
    collisions = (np.roll(filling_scheme_b1,Dict_IPN[IPN])*filling_scheme_b2)

    return np.roll(collisions,-Dict_IPN[IPN]),collisions.sum()

def Head_on(filling_scheme_b1,filling_scheme_b2, N_IP) -> np.array:
    '''~
    Using the boolean slots of beam1 and beam2, at the timestamp in which
    they have a maximum number of elements, you compute the number of 
    events that happen in IP numbered by N_IPC
    Args:
        filling_scheme_b1,filling_scheme_b2: np.array of boolean np.array 
        that represent the slots filled for the beam1 and the beam2
        IPN: a string that tell you the IP in which you are interested in
    Return: 
         a boolean np.array that tells the slots where there will be events,
         and also a number that tells you how many are these slots
    '''
    #Dict_IPN = {'IP1':0,'IP5':0,'IP2':2670,'IP8':891}
    n_collisions = np.zeros(len(N_IP))
    for i in np.arange(len(N_IP)):
        collisions = (np.roll(filling_scheme_b1,N_IP[i])*filling_scheme_b2)
        n_collisions[i] = collisions.sum()
    
    return n_collisions
    
def dataframe_from_parquet(path) -> pd.DataFrame:
    '''
    This function return the pd.DataFrame from a parquet file, achieved following the path,
    that should have at least the information about the BMODE, and the bunch_intensity of the two beams
    Args:
        path: is a string with all the path necessary to reach the parquet file with the information of the FILL
    Returns:
        pd.Dataframe: that contain the information about the BMODE and the bunch intensity of the two beams
    '''
    aux = dd.read_parquet(list(glob.glob(path + '/*/*')), columns = ['LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY','LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY','HX:BMODE'])
    df = aux.compute().sort_index()
    df = pandas_index_localize(df)
    return df

def filling_scheme_from_dataframe(df, TimeStamp = None, bmode = 'RAMP'):# -> tuple[np.array,np.array]:
    '''
    This function return the two filling schemes (in that input TimeStamp if specified
    or in that input bmode if specified differently from 'RAMP') from a dataframe that should have at least
    the information about the BMODE, and the bunch_intensity of the two beams
    Args:
        df: a dataframe that contain all the information need to compute the two bollean arrays for the beams
        TimeStamp: is a optional input and it gives the time in which the user want the information about the LR collision
        bmode: is an input, that become useless in case TimeStamap is given, it indicate the bmode in which the user is 
        interested for to receive the information about the LR collision (the function will use the first instant of that bmode)
    Returns:
        np.array(B1): a boolean vector that represent the slots filled by bunches for beam1
        np.array(B2): a boolean vector that represent the slots filled by bunches for beam2
    '''
    if TimeStamp == None:
        index_bmode = indeces_b_mode(df['HX:BMODE'],bmode)
        my_df = df[index_bmode]
        B1 = my_df['LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY']
        B2 = my_df['LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY']
        B1 = filled_bunches(B1,1e10)
        B2 = filled_bunches(B2,1e10)
    else:
        my_df = df.iloc[TimeStamp]
        B1 = (my_df.iloc[:,0].apply(filled_bunches,threshold = 1e10))
        B2 = (my_df.iloc[:,1].apply(filled_bunches,threshold = 1e10))
    return B1,B2

def filling_scheme_from_lpc_url(filename,FILLN):#-> tuple[np.array,np.array]:
    '''
    This function return two filling schemes from a .json file that is downloaded from the url link of LPC
    Args:
        filename: string that contain the name of the file with its extection(json) downloaded from the url of LPC
        FILLN : number of the fill
    Returns:
        np.array(B1): a boolean vector that represent the slots filled by bunches for beam1
        np.array(B2): a boolean vector that represent the slots filled by bunches for beam2
    '''
    f = open(filename)
    data = json.load(f)
    string = ""
    B1 = np.zeros(3564)
    B2 = np.zeros(3564)
    n_injection = int(data['fills'][f'{FILLN}']['name'][0:1000].split("_")[6].split("inj")[0])
    beam = np.fromstring(string.join(string.join(data['fills'][f'{FILLN}']['csv'].split("\t")[3:n_injection*2*10:10]).split("ring_")[0:n_injection*2+1]),dtype = int,sep = ',')

    n_bunches = np.fromstring(string.join(data['fills'][f'{FILLN}']['csv'].split("\t")[8:n_injection*2*10:10]),dtype = int,sep = ',')

    initial = np.fromstring(string.join(data['fills'][f'{FILLN}']['csv'].split("\t")[4:n_injection*2*10:10]),dtype = int,sep = ',')

    n_batches = [int(ii.split('\n')[0]) for ii in data['fills'][f'{FILLN}']['csv'].split("\t")[11:(n_injection)*2*10:10]]
    n_batches = np.append(n_batches,max(n_batches))
    initial = [int(ii) for ii in (initial-1)/10]
    for i in np.arange(n_injection*2):
        counter = 0
        if beam[i] == 1:
            for j in np.arange(n_batches[i]):
                init_batch = initial[i]+counter*(n_bunches[i]+7)
                B1[init_batch:init_batch+n_bunches[i]] = np.ones(n_bunches[i])
                counter += 1
        else:
            for j in np.arange(n_batches[i]):
                init_batch = initial[i]+counter*(n_bunches[i]+7)
                B2[init_batch:init_batch+n_bunches[i]] = np.ones(n_bunches[i])
                counter += 1
    return B1,B2

def filling_scheme_from_scheme_editor(filename):
    '''    
    This function return two filling schemes from a .json file that is downoladed from the scheme editor of LPC
    Args:
        filename: is a string with the filename with its extection(json) given by the scheme editor of LPC
    Returns:
        np.array(B1): a boolean vector that represent the slots filled by bunches for beam1
        np.array(B2): a boolean vector that represent the slots filled by bunches for beam2
    '''
    f = open(filename)
    data = json.load(f)
    B1 = data["beam1"]
    B2 = data["beam2"]
    return B1,B2
