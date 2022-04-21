# Data processing
import numpy as np
import cv2 
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors
import torch
import pandas as pd
import pickle

# For passing arguments across systems
import sys
import getopt
from pathlib import Path


######################################################################
# Utility Functions
######################################################################
######################################################################

# Reads camera calibration file, returns a dictionary containing 
# K: intrinsic parameters
# omega: distortion paramater
def ReadCalibration(data_filename):
    calib = {}

    fid = open(data_filename)

    data = fid.readline().split()
    imageWidth = int(data[1])

    data = fid.readline().split()
    imageHeight = int(data[1])

    data = fid.readline().split()
    focal_x = float(data[1])

    data = fid.readline().split()
    focal_y = float(data[1])

    data = fid.readline().split()
    princ_x = float(data[1])

    data = fid.readline().split()
    princ_y = float(data[1])

    data = fid.readline().split()
    omega = float(data[1])


    K_data = np.array([[focal_x, 0, princ_x],[ 0, focal_y, princ_y],[ 0, 0, 1]])

    calib['K'] = K_data
    calib['omega'] = omega

    fid.close()

    return calib

# Parses a string representing a trajectory
# First 3 values are the scaled plane normal
# Fourth value is number of trajectory samples
# Remaining of values are interleaved trajectory samples:
#   frame X Y Z u v
# u and v are currently unused
def ParseTraj(traj_string):
    
    traj = {}

    data = traj_string.split()
    data = np.array(data, dtype=np.float32)

    # Plane normal, with metric scale
    traj['up'] = data[0:3]
    
    # trajectory length
    nTrjFrame = int(data[3])
    
    # trajectory data
    data = data[4:]
    data = np.reshape(data, (nTrjFrame,6) ).T

    traj['frame'] = data[0, :] # Frame in time of samples (10 fps)
    traj['XYZ'] = data[1:4, :] # 3D coordinates of samples
    traj['uv'] = data[4:6, :]  # Dummy variables

    return traj

# Returns the distortion of points x
# x is 2xN point set
# omega is nonlinear fisheye distortion parameter
# K is linear camera intrinsic
def Distort(x, omega, K):
    x_n = np.linalg.inv(K) @ np.concatenate( ( x,np.ones((1,x.shape[1])) ),axis=0)
    r_u = np.sqrt(x_n[0,:]**2 + x_n[1,:]**2)
    r_d = 1/omega * np.arctan(2*r_u*np.tan(omega/2))
    x_n_dis_x = r_d / r_u * x_n[0,:]
    x_n_dis_y = r_d / r_u * x_n[1,:]
    x_n_dis = np.concatenate( (x_n_dis_x[None], x_n_dis_y[None], np.ones((1,x.shape[1]))), axis=0)
    x_dis = K @ x_n_dis
    return x_dis[:2,:]


def PrintHelp():
    print('REQUIRED:\n\t--json path/to/data.json\n'
          +'\t--images path/to/images_or_features\n'
          +'\t--output path/to/output\n'
          +'OPTIONAL:\n\t--processed\n'
          +'\t\tuse pre-processed features as .npy files')
    return




######################################################################
# Main
######################################################################
######################################################################
if __name__ == '__main__':
    
    # Global flags
    PROCESS_IMAGES = True # If pre-processed features should be loaded instead of pre-processed features

    # sys.argv[1:] = "--json E:/datasets/train.json --images E:/datasets/features/train --output E:/datasets --processed".split() # Debug example

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hpi:o:l:s:j:",["help","processed","images=","output=","json="])
    except getopt.GetoptError:
        print('Arguments are malformed.')
        PrintHelp()
        sys.exit(2)

    has_json = False
    has_images = False
    has_output = False

    for opt, arg in opts:
        
        if opt in ("-j", "--json"):
            __data_json = Path(arg)
            if not __data_json.exists():
                print("JSON path does not exist.")
                sys.exit(3)
            has_json = True

        if opt in ("-i", "--images"):
            __data_images = Path(arg)
            if not __data_images.exists():
                print("Images path does not exist.")
                sys.exit(3)
            has_images = True

        if opt in ("-o", "--output"):
            __data_target = Path(arg)
            if not __data_target.exists():
                __data_target.mkdir()
            has_output = True

        if opt in ("-p", "--processed"):
           PROCESS_IMAGES = False

        if opt in ("-h", "--help"):
            PrintHelp()
            sys.exit(-1)

    if not (has_json and has_images and has_output):
        print('Must have data json, image directory, and output directory.')
        PrintHelp()
        sys.exit(4)


    # Load data
    dataFrame = pd.read_json(__data_json, orient='index')
    total_items = len(dataFrame.index)
    
    # Set up pre-trained network for generating features from images
    model_Alex = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
    model_Alex.eval() # IMPORTANT!!
    mods = list(model_Alex.named_modules())
    model_Alex.named_modules()
    children = model_Alex.children()
    childrenchildren = list( list(children)[-1].children())
    penultimate_layer = childrenchildren[4]

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    penultimate_layer.register_forward_hook(get_activation('classifier.4'))

    img_shape_Alex = (256, 256)
    
    # Data containers for final KNN
    __id2feature_buffer = {} # key is frame, and AlexNet feature is val 
    __id2traj_buffer = {} # key is frame, and traj is val 


    item_counter = 1
    num_missing = 0
    is_new_folder = True # flag for reading new calibration file
    previous_folder = None

    for index, row in dataFrame.iterrows():
        
        # Dictionary of numpy arrays
        traj = ParseTraj(row['traj'])
        __id2traj_buffer[index] = traj

        # Image processing involves reading the distorted image from disk,
        # undistorting it, and running it through a CNN.
        # The alternative is to pre-process images into features and save to
        # disk as npy files.
        if PROCESS_IMAGES:
            # Determine if a new folder has been seen for loading the calibration file
            # Note: this dataset is continguous in folders. This reduces number of loads of
            # the calibration file
            folder_and_image_path = Path(index)
            data_folder = folder_and_image_path.parent

            if previous_folder is None or data_folder.name is not previous_folder.name:
                # Load calibration
                calibfile = __data_images / data_folder / Path('calib_fisheye.txt')
                if not calibfile.is_file():
                    print('[{}/{}] Could not find {}'.format( item_counter, total_items, calibfile ))
                    num_missing += 1
                    item_counter += 1
                    continue
                calib = ReadCalibration(calibfile) # returns dictionary of camera parameters

                # Undistorted pixel mapping for processing images
                img_shape = (720, 1280, 3) 
                all_pixel_coords = np.array( [ [j+.5,i+.5] for i in range(img_shape[0]) for j in range(img_shape[1]) ], dtype=np.float32).T # SLOW. Can cache this if every calib file is the same
                all_pixel_coords_dist = Distort(all_pixel_coords, calib['omega'], calib['K']) # Since this is a backwards mapping, we call forward Distort
                pix_coords = np.zeros(all_pixel_coords_dist.shape) # permute indices for interpn
                pix_coords[0] = all_pixel_coords_dist[1]
                pix_coords[1] = all_pixel_coords_dist[0]

                previous_folder = data_folder


            # Load image
            folder_and_image_path = Path(index)
            full_image_path = __data_images / folder_and_image_path
            if not full_image_path.exists():
                print('[{}/{}] Could not find {}'.format( item_counter, total_items, full_image_path ))
                num_missing += 1
                item_counter += 1
                continue
            img = cv2.cvtColor(cv2.imread(str(full_image_path)), cv2.COLOR_BGR2RGB).astype(np.float64)/255.0

            # Undistort
            img_undistorted = interpolate.interpn( (range(img.shape[0]),range(img.shape[1])), img, pix_coords.T , method = 'linear',bounds_error = False, fill_value = 0).astype(np.float32).reshape(img.shape[0], img.shape[1],3)
            
            # Adjust for pre-trained network
            img_resized = cv2.resize(img_undistorted, (img_shape_Alex[1], img_shape_Alex[0]))
            img_channel_swap = np.moveaxis(img_resized,-1,0).astype(np.float32)

            # Get feature
            img_channel_swap_unsqueezed = torch.unsqueeze(torch.from_numpy(img_channel_swap),0)
            returns = model_Alex(img_channel_swap_unsqueezed)
            feature = activation['classifier.4']
            __id2feature_buffer[index] = feature[0].numpy()
            
            print('[{}/{}] Feature from {}'.format( item_counter, total_items, full_image_path ))

        else: # use pre-processed data
            folder_and_image_path = Path(index)
            id = folder_and_image_path.stem
            feature_path = __data_images / folder_and_image_path.parent / '{}.{}'.format(id,'npy')
            if not feature_path.exists():
                print('[{}/{}] Could not find {}'.format( item_counter, total_items, feature_path ))
                num_missing += 1
                item_counter += 1
                continue
            feature = np.load(feature_path)
            __id2feature_buffer[index] = feature
            
            print('[{}/{}] Feature from {}'.format( item_counter, total_items, feature_path ))

        item_counter += 1


    print('Finished reading data. There were {} missing/unprocessable files.'.format(num_missing))
    
    print('Building descriptors buffer...')
    descriptors = np.zeros((len(__id2feature_buffer.keys()),4096)) # alexnet feature size
    i = 0
    for frame in __id2traj_buffer.keys():
        descriptors[i] = __id2feature_buffer[frame]
        i += 1

    print('Fitting KNN...')
    knn = NearestNeighbors(n_neighbors = 5).fit(descriptors)
       
    # Save data
    knnPath = __data_target / Path('future_localization.knn'.format())
    knnPickle = open(knnPath,'wb')
    pickle.dump(knn,knnPickle)
    knnPickle.close()
    
    # The kth index returned by KNN model is the index of the row in the training-json dataframe; traj = dataframe.iloc[idx[0,k]]['traj']

    print("Finished.")