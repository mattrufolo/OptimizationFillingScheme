# FillingStudies

## Installation instruction

Go to your preferred folder and do the following:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b  -p ./miniconda -f
source miniconda/bin/activate
python -m pip install numpy scipy matplotlib pandas ipython
git clone ssh://git@gitlab.cern.ch:7999/mrufolo/fillingstudies.git
cd fillingstudies
python -m examples.example_01
python -m examples.example_02
```

## Analysis

In this package we provide a toolbox (bb_toolbox.py) to compute beam-beam encounters schedules for circular colliders.
The function takes as input the filling scheme, as boolean vectors, of length of not bucket RF but bunch slots(3564), two dictionaries that give the positions of the detectors and also how many LongRange are considered for every detector, for every beam (in the function are suggested the position of LHC and 20 LongRange for every detector), and returns an output of two DataFrames, associated to the two beams, with the information about the partner bunch both for HeadOn and LongRange and also the information about the number and the positions of LongRange with respect to each detector.
Precisely, the output of the functions are two DataFrame with several columns that are:
```python
df_B1.columns
#Index(['# LR in ATLAS/CMS','HO partner in ATLAS/CMS', 'BB partners in ATLAS/CMS', 'Positions in ATLAS/CMS',
#'# of LR in LHCB,'HO partner in LHCB', 'BB partners in LHCB', 'Positions in LHCB',
#'# of LR in ALICE','HO partner in ALICE', 'BB partners in ALICE', 'Positions in ALICE'])
```
Here is illustrated an example, with two filling schemes shorter than LHC ones, in order to understand how the function works:
```python
import numpy as np
from tools_box.analysis import bb_tool_box as btb



B1 = np.array([1,0,1,1,0,0,1,0])
B2 = np.array([1,1,1,0,1,0,0,1])

Dict1 = {'detector':3}
Dict_LR = {'detector':2}


[df_B1,df_B2] = btb.bbschedule(B1,B2,numberOfLRToConsider = np.nan,Dict_Detectors = Dict1,Dict_nLRs =Dict_LR)
df_B1
#output:
#            # of LR in detector  HO partner in detector BB partners in detector Positions in detector
# 0                  3.0                     NaN         [1.0, 2.0, 4.0]     [-2.0, -1.0, 1.0]
# 2                  2.0                     NaN              [4.0, 7.0]           [-1.0, 2.0]
# 3                  3.0                     NaN         [4.0, 7.0, 0.0]      [-2.0, 1.0, 2.0]
# 6                  3.0                     1.0         [7.0, 0.0, 2.0]     [-2.0, -1.0, 1.0]
```

In order to facilitate the use of this function when the user has to study longer filling scheme, like the ones in LHC, has been created other functions that allows the user to load the filling schemes in different ways, i.e. a .json file from LPC enviroment. 
Here there are illustrated the ways:
- From a json file (like given by the filling scheme editor of LPC):
```python
from tools_box.analysis import bb_tool_box as btb
from tools_box.analysis import tool_box_to_bool as tbtb
numberOfLRToConsider = 20
[B1,B2] = tbtb.filling_scheme_from_scheme_editor(filename)
[df_B1,df_B2] = btb.bbschedule(B1,B2,numberOfLRToConsider)
```
- From a json file (like the one downloaded using LPC instruction):
```python
from tools_box.analysis import bb_tool_box as btb
from tools_box.analysis import tool_box_to_bool as tbtb
numberOfLRToConsider = 20
[B1,B2] = tbtb.filling_scheme_from_lpc_url(filename,FILLN)# FILLN is the number of the fill
[df_B1,df_B2] = btb.bbschedule(B1,B2,numberOfLRToConsider)
```
- From a dataframe (that should have at least have the information about the BMODE and the bunch intensity of the two beams):
```python
from tools_box.analysis import bb_tool_box as btb
from tools_box.analysis import tool_box_to_bool as tbtb
numberOfLRToConsider = 20
[B1,B2] = tbtb.filling_scheme_from_dataframe(df,bmode = 'STABLE')
# in this case the function returns the two filling schemes at the first possible istant of 'STABLE' BMODE 
[df_B1,df_B2] = btb.bbschedule(B1,B2,numberOfLRToConsider)
```
-From a parquet file (that should have at least have the information about the BMODE and the bunch intensity of the two beams):
```python
from tools_box.analysis import bb_tool_box as btb
from tools_box.analysis import tool_box_to_bool as tbtb
numberOfLRToConsider = 20
df = tbtb.dataframe_from_parquet(path)
# here the path indicates where to pick the parquet file
[B1,B2] = tbtb.filling_scheme_from_dataframe(df,bmode = 'STABLE')
[df_B1,df_B2] = btb.bbschedule(B1,B2,numberOfLRToConsider)
```

Here below, there are some simple examples, in order to understand better the function.


### Simple example
This example is the file "Example_1.py", it uses the easiest filling schemes possible, the one in which both the beam have an empty filling scheme. In this situation the function works normally, without giving you any tipe of error and it returns two DataFrame with all the columns exposed in the introduction but without any index, because there is no information from these filling scheme.

### Example using the filling scheme downloaded by the url of LPC
This example is the file "Example_2.py", it uses the filling scheme from a json file downloaded using an url from LPC. This example should return a plot of the number of LRs around the 4 detectors of LHC:

![Semantic description of image](/images/B1_bb_summary.png "Image Title")

## Synthesis

In this package we provide a toolbox (bb_toolbox.py) to compute a MonteCarlo simluation on a given filling scheme, in a circular collide, shifting its batches for both the beams at the same time in order to find other beam-beam encounters schedules for circular colliders.
The function takes as input the filling scheme, as boolean vectors, of length of not bucket RF but bunch slots(3564), two dictionaries that give the positions of the detectors and also how many LongRange are considered for every detector, for every beam (in the function are suggested the position of LHC and 20 LongRange for every detector), and returns an output of two DataFrames, associated to the two beams, with the information about the partner bunch both for HeadOn and LongRange and also the information about the number and the positions of LongRange with respect to each detector.
Precisely, the output of the functions are two DataFrame with several columns that are: