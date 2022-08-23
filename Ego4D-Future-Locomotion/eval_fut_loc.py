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


def PrintHelp():
    print('REQUIRED:\n\t--ref path/to/ref.json\n'
          +'\t--new path/to/new.json\n'
          +'OPTIONAL:\n\t--output path/to/output\n')
    return




######################################################################
# Main
######################################################################
######################################################################
if __name__ == '__main__':
    
    # Global flags
    FPS = 10                # framerate of data
    N_SEC = 7               # first n second error
    MAX_TRAJ_LENGTH = 150   # number of frames

    #sys.argv[1:] = "--ref E:/datasets/ref.json --new E:/datasets/new.json --output E:/datasets".split() # Debug example

    try:
        opts, args = getopt.getopt(sys.argv[1:],"ho:n:r:",["help","output=","new=","ref="])
    except getopt.GetoptError:
        print('Arguments are malformed.')
        PrintHelp()
        sys.exit(2)

    has_ref = False
    has_new = False
    has_output = False # Determines if metric should be saved

    for opt, arg in opts:
        
        if opt in ("-r", "--ref"):
            __data_ref = Path(arg)
            if not __data_ref.suffix == '.json':
                print("Must be a .json file: {}".format(__data_ref))
                sys.exit(3)
            if not __data_ref.exists():
                print("JSON path does not exist: {}".format(__data_ref))
                sys.exit(3)
            has_ref = True

        if opt in ("-n", "--new"):
            __data_new = Path(arg)
            if not __data_new.suffix == '.json':
                print("Must be a .json file: {}".format(__data_new))
                sys.exit(3)
            if not __data_new.exists():
                print("JSON path does not exist: {}".format(__data_new))
                sys.exit(3)
            has_new = True

        if opt in ("-o", "--output"):
            __data_target = Path(arg)
            if not __data_target.exists():
                __data_target.mkdir()
            has_output = True

        if opt in ("-h", "--help"):
            PrintHelp()
            sys.exit(-1)

    if not (has_ref and has_new):
        print('Must have ref json and new json, and output directory.')
        PrintHelp()
        sys.exit(4)

    # Load data
    df_ref = pd.read_json(__data_ref, orient='index')
    df_new = pd.read_json(__data_new, orient='index')
    total_items_ref = len(df_ref.index)
    total_items_new = len(df_new.index)
    if not total_items_ref == total_items_new:
        print('WARNING: size mismatch. Ref({}), New({})'.format(total_items_ref, total_items_new))
        
    # Data containers for keeping track of results
    mae_str = 'MAE'
    maeFirstN_str = 'MAE_first_{}_seconds'.format(str(N_SEC))
    cols = [mae_str, maeFirstN_str]
    scores = pd.DataFrame(index=df_ref.index, columns=cols)


    item_counter = 1
    num_missing = 0

    for index, row in df_ref.iterrows():

        # Dictionary of numpy arrays
        traj_ref = ParseTraj(row['traj'])

        if not index in df_new.index:
            print('[{}/{}] Reference index {} missing in new data'.format( item_counter, total_items_ref, index ))
            item_counter += 1
            num_missing += 1
            continue

        traj_new = ParseTraj(df_new.loc[index]['traj'])

        # Generate a rotation matrix which aligns camera space to
        # A space where the plane normal is the down vector
        # ref:
        r_y = -traj_ref['up']/np.linalg.norm(traj_ref['up'])
        old_r_z = np.array([0,0,1]) # forward vector
        r_z = old_r_z - (old_r_z@r_y)*r_y
        r_z /= np.linalg.norm(r_z)
        r_x = np.cross(r_y, r_z) # should be normal
        r_x /= np.linalg.norm(r_x)
        R_rect_ref = np.stack((r_x,r_y,r_z),axis=0)
        # new:
        r_y = -traj_new['up']/np.linalg.norm(traj_new['up'])
        old_r_z = np.array([0,0,1]) # forward vector
        r_z = old_r_z - (old_r_z@r_y)*r_y
        r_z /= np.linalg.norm(r_z)
        r_x = np.cross(r_y, r_z) # should be normal
        r_x /= np.linalg.norm(r_x)
        R_rect_new = np.stack((r_x,r_y,r_z),axis=0)

        # Align 
        traj_ref['XYZ_aligned'] = R_rect_ref @ (traj_ref['XYZ'].T - traj_ref['up'].T).T
        traj_new['XYZ_aligned'] = R_rect_new @ (traj_new['XYZ'].T - traj_new['up'].T).T

        # Project onto plane
        proj = np.array([[1.0,0,0],[0,0,1.0]])
        traj_ref['XZ_plane'] = proj @ traj_ref['XYZ_aligned']
        traj_new['XZ_plane'] = proj @ traj_new['XYZ_aligned']

        # Measure MAE
        t = 1
        error = []
        firstnseconderror = []
        FPS = 10
        start_frame = -1
        while t <= MAX_TRAJ_LENGTH:
            try: # match like indices
                idx_ref = np.where(traj_ref['frame'] == t)[0][0]
                idx_new = np.where(traj_new['frame'] == t)[0][0]
                if start_frame < 0:
                    start_frame = t
            except:
                t += 1
                continue
            
            dist = np.linalg.norm(traj_ref['XZ_plane'][:,idx_ref] - traj_new['XZ_plane'][:,idx_new])
            error.append(dist)
            if t < start_frame + (FPS * N_SEC):
                firstnseconderror.append(dist)
            t += 1

        # Log results
        scores.loc[index][cols[0]] = np.mean(error)
        scores.loc[index][cols[1]] = np.mean(firstnseconderror)
        
        print('[{}/{}] MAE: {}m, MAE first {} seconds: {}m'.format( item_counter, total_items_ref, scores.loc[index][cols[0]], N_SEC, scores.loc[index][cols[1]] ))

        item_counter += 1

    mean = scores[cols[0]].mean()
    median = scores[cols[0]].median()

    mean_firstN = scores[cols[1]].mean()
    median_firstN = scores[cols[1]].median()
    
    if has_output:
        print('Saving results.csv to {}'.format(__data_target))
        filePath = __data_target / 'results.csv'
        scores.to_csv(filePath)
    else:
        print('Output parameter not specified. Data was not saved.')

    print("Finished.")
    print('Mean/Median | MAE: {}m/{}m, MAE first {} seconds: {}m/{}m'.format( mean, median, N_SEC, mean_firstN, median_firstN))