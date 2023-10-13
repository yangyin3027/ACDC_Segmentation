import glob
import os
import shutil

import h5py
import numpy as np
import nibabel as nib
import argparse

def copy_files(path='../ACDC_datasets', saved_folder='../ACDC_datasets/images'):

    os.makedirs(saved_folder, exist_ok=True)

    num_files = 0
    for root, dir, file in os.walk(path):
        for f in file:
            if f.endswith('.gz') and f.find('_4d')==-1:
                src = os.path.join(root, f)
                des = os.path.join(saved_folder, f)
                print(src)
                shutil.copy(src, des)
                num_files +=1
    
    print(f'{num_files//2} paris of images and masks have been transfered')

def traverse_files(path='../ACDC_datasets'):
    subdirs = [root for root, _, _ in os.walk(path)]
    files = []

    for dir in subdirs:
        new_files = glob.glob(dir + '/*.nii.gz')
        # remove files contain '_4d'
        new_files = [f for f in new_files if f.find('_4d')==-1]
        files.extend(new_files)
    return files

def volume_to_2D(path='../ACDC_datasets', saved_folder='../ACDC_datasets/images'):

    slice_num = 0

    os.makedirs(saved_folder, exist_ok=True)

    files = sorted(traverse_files(path))

    for case in files:
        image_nib = nib.load(case)
        image = image_nib.get_fdata()

        # clip 0.5-99.5% for nonzero pixels
        p5, p95 = np.percentile(image, [0.5, 99.5])
        image = np.clip(image, p5, p95)

        # rescale image data to [0, 1]
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())

        msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            msk_nib = nib.load(msk_path)
            mask = msk_nib.get_fdata()

            mask = mask.astype(np.uint8)
            item = case.split("/")[-1].split(".")[0]
            assert image.shape == mask.shape, 'image and mask shape should be identical'

            for slice_ind in range(image.shape[2]):

                f = h5py.File(saved_folder + f'/{item}_{slice_ind}.h5', 'w')
                f.create_dataset(
                    'image', data=image[:, :, slice_ind], compression="gzip")
                f.create_dataset('label', data=mask[:, :, slice_ind], compression="gzip")
                f.close()
                slice_num += 1
    print("Converted all {} cases ACDC volumes to 2D slices".format(len(files)))
    print("Total {} slices".format(slice_num))


'''def train_val_split(path='../ACDC_datasets/images', train_ratio=0.8):
    files = glob.glob(path + '/*.h5')
    inds = np.random.permutation(len(files))

    num_trains = int(len(files) * train_ratio)

    os.makedirs(path + '/train', exist_ok=True)
    os.makedirs(path + '/val', exist_ok=True)

    for i in range(len(files)):
        src = files[inds[i]]
        if i < num_trains:
            des = os.path.join(path, 'train', src.split('/')[-1])
        else:
            des = os.path.join(path, 'val', src.split('/')[-1])
        shutil.move(src, des)
    print(f'train_val split @ {train_ratio} completed!')'''
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/Users/yangzn/Documents/ComputerVision/MedicalImageAI/ACDC_datasets/training',
                        help='root directories for the nii datasets')
    parser.add_argument('--saved-dir', type=str, default='../data')
    args = parser.parse_args()


    volume_to_2D(args.data, args.saved_dir)

    import random
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    seed = 30
    patients = traverse_files(args.data)
    patients = sorted([f for f in patients if f.find('_gt') == -1])

    img_file = patients[random.randint(0,len(patients))]
    img = nib.load(img_file).get_fdata().transpose(-1, 0, 1)
    fig, ax = plt.subplots()
    aximg = ax.imshow(img[0], cmap='gray')

    def update(frame):
        aximg.set_data(img[frame])
        return aximg

    ani = animation.FuncAnimation(fig=fig, func=update, frames=range(len(img)), interval=500)
    plt.show()
