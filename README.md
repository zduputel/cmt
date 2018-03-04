# CMT

Custom CMT inversion package

## Dependencies
* python
* python modules numpy, scipy, basemap and matplotlib
* sacpy module (see https://github.com/eost/sacpy.git)

## Some instructions
To use cmt, the cmt directory must be placed in a path pointed by the PYTHONPATH environment variable.

To use cmt, simply import the cmt module
```
import cmt
```

## Development
Development is hosten on GitHub in the [eost/cmt repository](https://github.com/eost/cmt).

## Example
An example is provided in test. Green's functions (after filtering and STF convolution are placed in GFs). Data (filtered in the same passband as Green's functions are placed in DATA and listed in i_sac_lst.txt (we use sac file format).

To perform the inversion:
```
python cmt_inv.py
```
Output CMT solution is provided in o_CMTSOLUTION. 

To plot a comparison between observations and predictions, use:
```
python plot_traces.py
```
The resulting observed (black) and predicted (red) traces is in traces.pdf

-- 

**Report bugs to: <zacharie.duputel@unistra.fr>**
