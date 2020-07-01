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
    rise_time = (0.85, 1.15]
    decay_time=(15.38463905329085, 18.18]
    """
    t = np.arange(-length+offset, length+offset)
    basis = exp(t *(1/rise_time-1/decay_time))/(1+exp(t/rise_time)) 
    return basis/basis.sum()

def get_diff_basis(rise_time, decay_time, length=200, offset=0):
    """
    Get the derivative of the basis, normalized by its rms area.
    rms_area seems to be the only way to ensure that it works
    """
    t = np.arange(-length+1, length)
    diff_basis = 1/rise_time *exp((-1/rise_time-1/decay_time)*t)/(1+exp(-t/rise_time))**2
    diff_basis -= 1/decay_time * exp(-t/decay_time)/(1+exp(-t/rise_time))

    #These other four normalization scalars are useless
    # first_moment = abs(t @ diff_basis )
    # second_moment = abs(t**2 @ diff_basis)
    # area = abs(diff_basis).sum()
    # max_point = abs(diff_basis).max()
    rms_area = sqrt((diff_basis**2).mean())
    return diff_basis/( rms_area**0.91 ) # after some experimentation, the exponent of 0.91 seems to be the best at minimizing the stuck_point_count.
    # Other normalization scalars usable include: decay_time**n, and rise_time**n.

def get_smooth_diff_conv_matrix(rise_time, decay_time, resolution=500, length=200):
    """
    Get the smoothened version of the convolution matrix
    """
    matrix = []
    for start_point in np.linspace(0, -length, resolution):
        decimal_offset = start_point%1
        int_offset = int(start_point//1)
        basis = get_basis(rise_time, decay_time, length=length, offset=decimal_offset)[length+int_offset:2*length+int_offset]
        matrix.append(np.diff(basis))
    return ary(matrix)

def get_smooth_conv_matrix(rise_time, decay_time, resolution=500, length=200):
    """
    Get the smoothened version of the convolution matrix
    """    
    matrix = []
    for start_point in np.linspace(0, -length, resolution):
        decimal_offset = start_point%1
        int_offset = int(start_point//1)
        basis = get_basis(rise_time, decay_time, length=length, offset=decimal_offset)[length+int_offset:2*length+int_offset]
        matrix.append(basis)
    return ary(matrix)

def get_curve_basis(rise_time, decay_time, length=200):
    """
    get the second order derivative of the basis
    """ 
    t = np.arange(-length+1, length)
    diff_basis = 1/rise_time *exp((-1/rise_time-1/decay_time)*t)/(1+exp(-t/rise_time))**2
    diff_basis -= 1/decay_time * exp(-t/decay_time)/(1+exp(-t/rise_time))
    return np.diff(diff_basis)

def get_diff_convolve_matrix(trace_length=200):
    """
    Get the convolution matrix where matrix @ trace = strength of that basis
    """
    vlong_matrix = []
    for rise in np.linspace(rise_min, rise_max, rise_resolution):
        long_matrix = []
        for dec in np.linspace(dec_min, dec_max, dec_resolution):
            long_matrix.append(get_diff_basis(rise, dec, length=trace_length))
        vlong_matrix.append(long_matrix)
    vlong_matrix = ary(vlong_matrix, dtype=float)
    total_length = vlong_matrix.shape[-1]
    return ary([vlong_matrix[:,:, trace_length-i:total_length-i] for i in range(trace_length)], dtype=float)

def get_curve_convolve_matrix(trace_length=200):
    """
    Get the convolution matrix where matrix @ trace = strength of that basis
    (This matrix uses the second derivative (i.e. curvature) of the basis)
    """
    vlong_matrix = []
    for rise in np.linspace(rise_min, rise_max, rise_resolution):
        long_matrix = []
        for dec in np.linspace(dec_min, dec_max, dec_resolution):
            long_matrix.append(get_curve_basis(rise, dec, length=trace_length))
        vlong_matrix.append(long_matrix)
    vlong_matrix = ary(vlong_matrix, dtype=float)
    total_length = vlong_matrix.shape[-1]
    return ary([vlong_matrix[:,:, trace_length-i:total_length-i] for i in range(trace_length)], dtype=float)

def is_local_max_3d(image):
    """
    Determine if pixel is a local maximum or not.
    Note that it returns an array with trimmed edges,
    i.e. if input has shape (n,m,k), it will return (n-1, m-1, k-1)
    """
    mright = image[1:-1, 1:-1, 1:-1]>=image[1:-1, 1:-1, 2:]
    mleft = image[1:-1, 1:-1, 1:-1]>=image[1:-1, 1:-1, :-2]
    mtop = image[1:-1, 1:-1, 1:-1]>=image[1:-1, :-2, 1:-1]
    mbottom = image[1:-1, 1:-1, 1:-1]>=image[1:-1, 2:,1:-1]
    mfront = image[1:-1, 1:-1, 1:-1]>=image[:-2, 1:-1, 1:-1]
    mback = image[1:-1, 1:-1, 1:-1]>=image[2:, 1:-1, 1:-1]

    return np.logical_and(
        np.logical_and(np.logical_and(mright, mleft), np.logical_and(mtop, mbottom)),
        np.logical_and(mfront, mback)
    )

def is_local_max_1d(series):
    mleft = series[1:-1]>=series[:-2]
    mright = series[1:-1]>=series[2:]
    return np.logical_and(mleft, mright)

def get_ricker_matrix(length, widths):
    """
    get the ricker wavelet convolution matrix
    provided a list of widths
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
    # curvature = np.diff(time_derivative, axis=1)
    
    # long_matrix = get_diff_convolve_matrix()
    conv_matrix = get_smooth_diff_conv_matrix((rise_min + rise_max)/2, (dec_min+dec_max)/2, 600)
    peak_findable = (conv_matrix @ time_derivative.T).T

    labels=num_events.values
    prediction = []
    for line_of_peaks in peak_findable:
        peak_loc = find_peaks(line_of_peaks, height=0.5) # tune height according to line_of_peaks.max()?
        prediction.append(line_of_peaks[peak_loc[0]].__len__())
    prediction = ary(prediction)

    print(confusion_matrix(labels, prediction))

    truth_pred = ary([labels, prediction]).T

    diff_peaks = np.diff(peak_findable, axis=1) # this gives extra information

    # examine the incorrectly predicted data:
    for num, ind in enumerate(np.arange(len(peak_findable))[(truth_pred==[2,1]).all(axis=1)]):
        norm_plot(peak_findable[ind])
        norm_plot(wave_forms.loc[ind])
        plt.title(f'{ind=}, amp={amp1[ind]}, {amp2[ind]}')
        plt.show()
        if num>20: # don't have time to examine every single wrongly plotted data
            break

    """
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