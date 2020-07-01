from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
import seaborn as sns
from skimage.feature.blob import blob_dog, blob_doh, blob_log, peak_local_max
from skimage.feature import hessian_matrix, hessian_matrix_det, hessian_matrix_eigvals
from skimage.feature import local_binary_pattern
from sklearn.metrics import confusion_matrix
from scipy.ndimage import maximum_filter

# Set path to data and filename. You can also store it a single variable
# DATA_PATH = "../data/CeBr10k_1.txt" if you prefer. Here we expect to use at least the path itself later,
# so we separate them.
DATA_PATH = ""
fname = "CeBr10k_1.txt"
num_samples = 30
# Lists to store each image, energy, position, and labels. We know the filesize, so we could create
# arrays that perfectly match the data, but let's assume we don't know how many lines we're going to
# read. 
images, energies, positions = [], [], []
# Open the file
with open(DATA_PATH + fname, "r") as datafile:
    # Loop over the file, line by line
    for line in datafile:
        
        # The line is still a string when read from the file. We use numpys fromstring()
        # to convert the line to a numpy array, specifying that each element is separated
        # by a space. This does not convert the line in the file, only the "copy" that we have
        # read into memory. fromstring() also removes any trailing newline ('\n') characters
        # so we don't have to worry about that. The values will be interpreted as floats.
        line = np.fromstring(line, sep=' ')
        
        # Now we pick slices of the array. The first 256 elements are 'pixels' of the detector image
        image = line[:256]
        
        # Get the two energies, at index 256 and 259.
        energy = np.array((line[256], line[259]))
        
        # And the four position values
        pos = np.array((line[257], line[258], line[260], line[261]))

        # Set label for the event. If Energy2 is 0 it is a single event. Any other values corresponds 
        # to a double event. We label single events as type 0, and doubles as type 1. We could also 
        # use Xpos2 or Ypos2 for this purpose.
        if energy[1] == 0:
            label = 0
        else:
            label = 1

        # Finally, we take the separated arrays and add them to their respective "storage" lists.
        images.append(image)
        energies.append(energy)
        positions.append(pos)

    image = image.reshape([16,16])

# We've now looped over the entire file. The only thing that remains is to convert the lists
# to numpy arrays.
images = ary(images)
energies = ary(energies)
positions = ary(positions)
images = np.transpose(images.reshape([-1, 16, 16]), axes=[0,2,1])

def is_local_max_2d(image):
    """
    Determine whether each pixel is a local maximum or not by comparing to horizonal and vertical neighbours.
    Note that it will trim all edges,
    i.e. if input has shape(n,m), it will return (n-1,m-1)
    """
    mright = image[1:-1, 1:-1]>=image[1:-1, 2:]
    mleft = image[1:-1, 1:-1]>=image[1:-1, :-2]
    mtop = image[1:-1, 1:-1]>=image[:-2, 1:-1]
    mbottom = image[1:-1, 1:-1]>=image[2:,1:-1]

    return np.logical_and(np.logical_and(mright, mleft), np.logical_and(mtop, mbottom))

def diagonal_max_2d(image):
    """
    Compare pixel with diagonal neighbours to see if it is a maximum or not
    """
    mtl = image[1:-1, 1:-1]>=image[:-2, :-2]
    mtr = image[1:-1, 1:-1]>=image[:-2, 2:]
    mbl = image[1:-1, 1:-1]>=image[2:, :-2]
    mbr = image[1:-1, 1:-1]>=image[2:, 2:]
    return np.logical_and(np.logical_and(mtl, mtr), np.logical_and(mbl, mbr))

def make_wide(image):
    """
    Expand the image from nxm to (n+2)x(m+2)
    by copying the values to the neighbouring edges.
    """
    wide_image = np.zeros(ary(image.shape)+2)
    wide_image[1:-1, 1:-1] = image
    wide_image[[0,-1],1:-1] = image[[0,-1],:]
    wide_image[1:-1,[0,-1]] = image[:,[0,-1]]
    wide_image[0,0] = image[0,0]
    wide_image[-1,-1] = image[-1,-1]
    wide_image[0,-1] = image[0,-1]
    wide_image[-1,0] = image[-1,0]
    return wide_image

def is_local_max_2d_wide(image, diagonal=False):
    """
    does not trim the edges.
    This is done by padding the edges with zeros.
    """
    wide_image = make_wide(image)
    mask = is_local_max_2d(wide_image)
    if diagonal:
        mask = np.logical_and(diagonal_max_2d(wide_image), mask)
    return mask

if __name__=='__main__':
    laplacian = np.zeros([3,3])
    laplacian[np.unravel_index(range(9)[1::2], (3,3))] = -1
    laplacian[1,1] = 4

    gaussian = np.ones([3,3])
    gaussian[np.unravel_index(range(9)[1::2], (3,3))] = 2
    gaussian[1,1] = 4
    gaussian *= 1/16

    prediction1 = []
    for ind, image in enumerate(images):
        # image_curvature = convolve(image, laplacian)
        smooth_image = convolve(image, gaussian)
        smooth_curvature = convolve(smooth_image, laplacian)
        bulged_mask = smooth_curvature>smooth_curvature.max()*0.05
        local_max_mask = is_local_max_2d_wide(image)

        pred1 = np.logical_and(local_max_mask, bulged_mask).sum() # >90 % accuracy
        """
        Failed ideas:
        pred1 = (image[is_local_max_2d_wide(image_curvature, diagonal=True)] > np.percentile(image.flatten(), 245/256*100)).sum() # 87.76 % accuracy
        pred2 = (image[is_local_max_2d_wide(smooth_image, diagonal=True)] > np.percentile(image.flatten(), 245/256*100)).sum() # 85.13 % accuracy
        pred2 = (image[peak_local_max(image_curvature)]>np.percentile(image.flatten(), 250/256*100)).sum()
        pred3 = (image[peak_local_max(smooth_image)]>np.percentile(image.flatten(), 250/256*100)).sum()
        """
        prediction1.append(pred1)

    # get the labels (truths)
    labels = 2 - (energies==0).sum(axis=1)
    # Peak must be at the center of a blob (contiguous area with 3+ pixels) where curvature is negative in both directions

    print(confusion_matrix(labels, prediction1)) # display result

    truth_pred = ary([labels, prediction1]).T

    # Examine reasons for failed cases
    input('Next, we will display the improperly predicted cases:')

    chosen_to_be_examined = (truth_pred==[2,1]).all(axis=1)
    for num, image in enumerate(images[chosen_to_be_examined][:num_samples]):
        ind = np.where(chosen_to_be_examined)[0][num]
        replay = True
        while replay:
            # introduce information about neighbourhood
            image_curvature = convolve(image, laplacian)/image_curvature.max() # introduce some global information
            smooth_image = convolve(image, gaussian)/smooth_image.max()
            smooth_curvature = convolve(smooth_image, laplacian)/smooth_curvature.max()
            bulged_mask = smooth_curvature>smooth_curvature.max()/1
            local_max_mask = is_local_max_2d_wide(image)

            for img, name in zip([image, image_curvature, smooth_image, smooth_curvature, bulged_mask, local_max_mask],
                                ['image', 'image_curvature', 'smooth_image', 'smooth_curvature', 'bulged_mask', 'local_max_mask']):
                sns.heatmap(img)
                plt.title(f'{ind}: '+name+f' #events={labels[ind]}')
                plt.show()
            if len(input('Enter any character before to move onto the next plot, or ctrl+c to stop. Press only enter to replay this plot.'))>0:
                replay=False

    """
    Optimization ideas:
    1. tune the 0.05 parameter?
    2. Add division by different values before making the masks
    3. May even assign confidence score, and raise all of the cases to a human inspector  where the confidence score is low?
    """