from __future__ import division
import numpy as np
import h5py, sys
import nibabel as nib
import os
import shutil
from glob import glob

from collections import namedtuple, OrderedDict
from skimage import morphology, transform, draw
from skimage.feature import canny
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman

import scipy.ndimage as snd

import matplotlib.pyplot as plt
from matplotlib import animation
import cv2
from cv2 import bilateralFilter

##############################################################################
###           Weight Map Generation and mini-Batch Class Weights           ###
##############################################################################
def find_edge(image):
    can = canny(image, sigma=1)
    edge = morphology.binary_dilation(can, footprint=morphology.disk(1))
    return edge.astype(np.float32)

def getEdgeEnhancedWeightMap(label, label_ids=[0,1,2,3],
                             scale=1, edgescale=1, assign_equal_wt=False):
    shape = (0, ) + label.shape[1:]
    weight_map = np.emtpy(shape, dtype='uint8')
    pixel_cnt = label[0, :, :].size

    if assign_equal_wt:
        return np.ones_like(label)
    for i in range(label.shape[0]):
        # Estimate weight maps:
        weights = np.ones(len(label_ids))
        slice_map = np.ones(label[i, :, :].shape)
        for _id in label_ids:
            selected_idx = label[i, :, :] == _id

            class_freq = np.sum(selected_idx)
            if class_freq:
                weights[_id] = scale * pixel_cnt / class_freq
                slice_map[selected_idx] = weights[_id]
                edge = find_edge(np.float32(selected_idx))
                edge_frequency = np.sum(edge==1.0)
                if edge_frequency:
                    slice_map[np.where(edge==1.0)] += edgescale*pixel_cnt/edge_frequency
        
        weight_map = np.append(weight_map, np.expand_dims(slice_map, axis=0), axis=0)
    return np.float32(weight_map)

def getAvgBatchClassWeights(label, label_ids=[0,1], scale=1, assign_equal_wt=False):
    batch_size = label.shape[0]
    batch_weights = np.zeros((batch_size, len(label_ids)))
    if assign_equal_wt:
        return np.ones(len(label_ids), dtype=np.uint8)
    pixel_cnt = label[0, :, :].size
    eps =  1e-3
    for i in range(batch_size):
        for _id in label_ids:
            batch_weights[i, _id] = scale * pixel_cnt / np.float(np.sum(label[i, :, :] == _id)+eps)
    return np.float32(np.mean(batch_weights+1, axis=0))

def normalize(image, scheme='zscore'):
    if scheme == 'zscore':
        image = normalize_zscore(image, z=0.5, offset=0, clip=True)
    elif scheme == 'minmax':
        image = normalize_minmax(image)
    return image

def normalize_mean_std(image, mean, std):
    if std == 0.0:
        std += 1e-6
    return (image - mean) / std

def normalize_zscore(data, z=2, offset=0.5, clip=False):
    '''
    Normalize contrast across volume
    '''
    mean = np.mean(data)
    std = np.std(data)
    img = ((data-mean)/(2*std*z) + offset)
    if clip:
        img = np.clip(img, -0.0, 1.0)
    return img

def normalize_minmax(data):
    _min = np.float(np.min(data))
    _max = np.float(np.max(data))
    if (_max - _min) != 0:
        img = (data - _min) / (_max - _min)
    else:
        img = np.zeros_like(data)
    return img

def slicewise_normalize(img_data4D, scheme='minmax'):
    x_dim, y_dim, n_slices, n_phases = img_data4D.shape

    data_4d = np.zeros([x_dim, y_dim, n_slices, n_phases])
    for slice in range(n_slices):
        for phase in range(n_phases):
            data_4d[:,:,slice, phase] = normalize(img_data4D[:,:,slice,phase], scheme)
    return data_4d


###################################################################################
###                       Prepare ACDC Datasets                                 ###
###################################################################################
def heart_metrics(seg_3dmap, voxel_size, classes=[3,1,2]):
    '''
    Compute the volumes of each classes
    '''
    volumes = []
    for c in classes:
        # make a copy so that original data not altered
        seg_3dmap_copy = np.copy(seg_3dmap)
        seg_3dmap_copy[seg_3dmap_copy != c ] = 0

        # clip value to compute volumes
        seg_3dmap_copy = np.clip(seg_3dmap_copy, 0, 1)

        # compute volume
        volume = seg_3dmap_copy.sum() * np.prod(voxel_size) / 1000.
        volumes += [volume]
    return volumes

def ejection_fraction(ed_vol, es_vol):
    stroke_vol = ed_vol - es_vol
    return (np.float(stroke_vol) / np.float(ed_vol))*100.

def myomass(myo_vol):
    '''
    Specific gravity of heart muscle (1.05 g/ml)
    '''
    return myo_vol * 1.05

def plot_4D(data4D):
    '''
    data4D: [h,w,slices,tframe]
    '''
    slices, tframes = data4D.shape[-2:]

    slice_cnt = 0
    for slice in [data4D[:,:,z,:] for z in range(slices)]:
        outdata = np.transpose(slice, (-1, 0, 1))
        fig = plt.figure(1)
        fig.canvas.set_window_title('slice_No.' + str(slice_cnt))
        slice_cnt +=1 

        def init_out():
            im.set_data(outdata[0])
        def animate_out(i):
            im.set_data(outdata[i])
            return im
        
        im = fig.gca().imshow(outdata[0], cmap='gray')
        anim = animation.FuncAnimation(fig, animate_out, init_func=init_out,
                                       frames=tframes,
                                       intervals=50)
        plt.show()

def parse_cfg(path):
    '''Reads patient cfg file and returns a dict'''
    infos = {}
    with open(os.path.join(path, 'Info.cfg')) as f:
        for line in f:
            l = line.rstrip().split(': ')
            infos[l[0]] = l[1]
    return infos

class ACDC(object):
    NORMAL = 'NOR'
    MINF = 'MINF'
    DCM = 'DCM'
    HCM = 'HCM'
    RV = 'RV'

    def group_patient_cases(self, src_root, dsc_root, force=False):
        '''
        Group the patient data according to cardiac pathology
        '''
        cases = sorted(next(os.walk(src_root))[1]) # extract patient directories
        dsc_path = os.path.join(dsc_root, 'pathology_groups')
        if force:
            shutil.rmtree(dsc_path)
        if os.path.exists(dsc_path):
            return dsc_path
        
        for sub in [self.NORMAL, self.MINF, self.DCM, self.HCM, self.RV]:
            os.makedirs(os.path.join(dsc_path, sub))
        
        for case in cases:
            src_full_path = os.path.join(src_root, case)
            case_info = parse_cfg(src_full_path)['Group']
            dsc_full_path = os.path.join(dsc_path, case_info, case)
            print('Grouped {} to {}'.format(case, dsc_full_path))

            shutil.copytree(src_full_path, dsc_full_path, ignore=shutil.ignore_patterns())
        
    def train_valid_test_spit(self, src_root, dst_root,
                              train_ratio=0.7, val_ratio=0.15):
        '''Split the data into 70:15:15 for train-valid-test set'''

        dst_path = os.path.join(dst_root, 'dataset')
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        
        for sub in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dst_path, sub))

        groups = next(os.walk(src_root))[1]
        for group in groups:
            print('split {} ....'.format(group))
            group_path = next(os.walk(os.path.join(src_root, group)))[0]
            patients_folders = next(os.walk(group_path))[1]
            np.random.shuffle(patients_folders)
            
            train_idx = int(train_ratio * len(patients_folders))
            val_idx = int((val_ratio + train_ratio) * len(patients_folders))

            train_set = patients_folders[:train_idx]
            val_set = patients_folders[train_idx:val_idx]
            test_set = patients_folders[val_idx:]

            for patient in train_set:
                folder_path = os.path.join(group_path, patient)
                dst_full_path = os.path.join(dst_path, 'train', patient)
                shutil.copytree(folder_path, dst_full_path,ignore=shutil.ignore_patterns())
            
            for patient in val_set:
                folder_path = os.path.join(group_path, patient)
                dst_full_path = os.path.join(dst_path, 'val', patient)
                shutil.copytree(folder_path, dst_full_path,ignore=shutil.ignore_patterns())
            
            for patient in test_set:
                folder_path = os.path.join(group_path, patient)
                dst_full_path = os.path.join(dst_path, 'test', patient)
                shutil.copytree(folder_path, dst_full_path,ignore=shutil.ignore_patterns())

def grouping_patients(src, dsc):
    # grouping patients by pathology
    ACDC().group_patient_cases(src, dsc)
    inter_src = os.path.join(dsc, 'pathology_groups')
    # split patients to train-val-test sets
    ACDC().train_valid_test_spit(inter_src, dsc)   

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=str)
    parser.add_argument('--dsc', default=str)
    args = parser.parse_args()
