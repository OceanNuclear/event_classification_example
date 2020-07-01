from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks, find_peaks_cwt
from sklearn.metrics import confusion_matrix

rise_min, rise_max = 0.85, 1.15
# rise_min, rise_max = 0.75, 1.2
dec_min, dec_max = 1/0.065, 1/0.055
# dec_min, dec_max = 10, 30

rise_resolution, dec_resolution = 100, 100
# average values to be used
rise_time, decay_time = 1.025, 16

norm_plot = lambda x: plt.plot(x/abs(x).max())

def get_basis(rise_time, decay_time, length=200, offset=0):
    """
    Generate the basis vector for a given rise time and decay time.
    for the simulated dataset training_pm_nosat_150k.dat,
    Normalized by its area.
    rise_time = (0.85, 1.15]
    decay_time=(15.38463905329085, 18.18]
    """
    t = np.arange(-length+offset, length+offset)
    basis = exp(t *(1/rise_time-1/decay_time))/(1+exp(t/rise_time)) 
    return basis/basis.sum()

def get_smooth_diff_conv_matrix(rise_time, decay_time, resolution=500, length=200):
    """
    Get the smoothened version of the convolution matrix.
    Kind of bodged together by allowing an 0<=offset<1 to be applied on get_basis, so that I can slide it along by a non-integer value.
    I'm not super proud of it but it works, and it's fast enough, so why change it.
    """
    matrix = []
    for start_point in np.linspace(0, -length, resolution):
        decimal_offset = start_point%1
        int_offset = int(start_point//1)
        basis = get_basis(rise_time, decay_time, length=length, offset=decimal_offset)[length+int_offset:2*length+int_offset]
        matrix.append(np.diff(basis))
    return ary(matrix)

def is_local_max_1d(series):
    mleft = series[1:-1]>=series[:-2]
    mright = series[1:-1]>=series[2:]
    return np.logical_and(mleft, mright)

def negative_curvature(series):
    """
    Checks if the slope is smoothly decreasing. 
    returns a boolean series of len n-2
    """
    return np.diff(np.diff(series))<0

def get_ricker_matrix(length, widths):
    """
    get the ricker wavelet convolution matrix
    provided a list of widths
    May be able to apply this instead?
    """
    vary_width = ary([ricker(length, w) for w in widths])
    return ary([np.roll(vary_width, i) for i in range(length)])

if __name__=='__main__':
    df = pd.read_csv('training_pm_nosat_150k.dat', sep=' ', header=None) # lazy way to read the data
    # df[1] is just an integer 200. I don't know why.
    # Below are Information which won't be available in the experimental data: (i.e. labels usable for training)
    num_events = df[0]
    amp1, amp2 = df[2], df[7]
    rise1, rise2 = df[3], df[8]                 # default decay2=0
    decay1, decay2 = df[4], df[9]               # default pos2=0
    offset, pos1, pos2 = df[6], df[5], df[10]   # default pos2=0
    # Information which will actually be avialble in the experiment
    wave_forms = df[df.columns[11:]]
    print('Data Read. Extracting useful information out of the data...')
    wave_forms.columns = range(wave_forms.shape[1])
    print("Renamed columns, shifting upwards...")

    time_derivative = np.diff(wave_forms.values, axis=1)
    
    window_length = wave_forms.shape[-1]
    conv_matrix = get_smooth_diff_conv_matrix((rise_min + rise_max)/2, (dec_min+dec_max)/2, 3*window_length, length=window_length)
    peak_findable = (conv_matrix @ time_derivative.T).T

    prediction = []
    for line_of_peaks in peak_findable:
        peak_loc = find_peaks(line_of_peaks, height=0.5) # tune height according to line_of_peaks.max()?
        prediction.append(line_of_peaks[peak_loc[0]].__len__())
    prediction = ary(prediction)

    labels = num_events.values

    print(confusion_matrix(labels, prediction))

    truth_pred = ary([labels, prediction]).T

    diff_peaks = np.diff(peak_findable, axis=1) # this gives extra information?

    # examine the incorrectly predicted data:
    for num, ind in enumerate(np.arange(len(peak_findable))[(truth_pred==[2,1]).all(axis=1)]):
        norm_plot(peak_findable[ind])
        norm_plot(wave_forms.loc[ind])
        plt.title(f'{ind=}, amp={amp1[ind]}, {amp2[ind]}')
        plt.show()
        if num>20: # don't have time to examine every single wrongly plotted data
            break

    """
    Tunables:
    1. Optimum rise_time and decay_time.
        (Or may even use a few (rise_time, decay_time) combination?? e.g. five of them, one for the average, four for the max and min rise and decay time.s)
    2. Method to extract the peaks:
        2.1 find_peaks_cwt seems slow; but would it have a better result if given the exact right parameter? (my intuition says no, but that could be proven wrong.)
        2.2 find_peaks can use a more properly scaled values.
        2.3 find_peaks should also have a more properly tuned parameter for height, width, etc.
        2.4 should also add a post-processing step that filters out the peaks that are clearly just noise?
            e.g. use is_local_max_1d, the negative_curvature.
            2.4.2 may even consider the cross-over points of the negative_curvature line?
    Log of failed methods:

    I also tried creating a 3D instead of a 1D convolution result,
    where the other two dimension are the variation of the rise_time and decay_time in the basis.
    However, this was proven to be a terrible idea as
    1. The variation in these two new directions are basically zero, compared to the variation in the time_difference dimension. So a specialized algorithm would be needed to extract where the hot spots are.
    2. Even if we did manage to extract these hot spots, their rise_time and decay_time value do not match the rise_time and decay_time values exactly. 
        i.e. where the the convolution result is highest is NOT where (the (rise_time, decay_time) of the basis)==(the (rise_time, decay_time) of the signal).
    So in the end I just stuck with using these values.
    3. I also tried using the Fourier and Laplace transform of the expected signal shape, but got stuck after
        3.1 Not getting any analytical form of the integral using wolframalpha
        3.2 Changing the basis shape to a linear-rise exponential fall so that it does have an analytical solution after integration; but even then I ask myself:
            so what? I have an analytical equation of what a signal containting is expected to look like wrt. (omega) or (s). And now what am I going to do with that information ¯\\_(ツ)_/¯
    """