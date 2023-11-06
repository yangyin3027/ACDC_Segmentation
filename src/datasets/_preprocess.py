import os
import shutil
from glob import glob

import h5py
import numpy as np
import nibabel as nib

from tqdm import tqdm

def traverse_files(path, val_ratio=0.2, data='acdc'):
    folders = [root for root, _, _ in os.walk(path)]
    files = []
    
    for dir in folders:
        new_files = glob(dir + '/*.nii.gz')
        # remove files contain '_4d' or '_gt'
        if data == 'acdc':
            new_files = [f for f in new_files 
                        if (f.find('_4d')==-1 and f.find('_gt')==-1)]
        elif data == 'synapse':
            new_files = [f for f in new_files 
                         if f.find('label')==-1]
        files.extend(new_files)
    
    files = np.array(files)
    
    if val_ratio != 0:
        index = np.arange(len(files))
        index = np.random.permutation(index)

        train_split = int(len(files) * (1-val_ratio))
        train_files = files[index[:train_split]]
        val_files = files[index[train_split:]]
        return train_files, val_files
    return files
   
def volume_to_2d(files, saved_dir, data='acdc'):
    
    os.makedirs(saved_dir, exist_ok=True)

    slice_num = 0
    with tqdm(files, desc='convert nii volume to h5 2d slice',
                     total=len(files)) as pbar:
        for i, case in enumerate(pbar):
            image_nib = nib.load(case)
            image = image_nib.get_fdata()

            # clip 0.5-99.5% for nonzero pixels
            p5, p95 = np.percentile(image, [0.5, 99.5])
            image = np.clip(image, p5, p95)

            # normalize data to [0, 1]
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())

            if data == 'acdc':
                msk_file = case.replace('.nii.gz', '_gt.nii.gz')
            elif data == 'synapse':
                msk_file = case.replace('img', 'label')
            
            msk_nib = nib.load(msk_file)
            mask = msk_nib.get_fdata()

            fname = case.split('/')[-1].split('.')[0]
            assert image.shape == mask.shape, 'image and mask shape should be identical'

            for slice_ind in range(image.shape[2]):
                # remove the slices contain no labels
                f = h5py.File(saved_dir + f'/{fname}_{slice_ind}.h5', 'w')
                f.create_dataset(
                    'image', data=image[:, :, slice_ind], compression='gzip'
                )
                f.create_dataset(
                    'label', data=mask[:, :, slice_ind], compression='gzip'
                )
                f.close()
                slice_num += 1

                pbar.set_postfix(slice_num=f'{slice_num}', converted_case=f'{i}')
            
            pbar.set_postfix(slice_num=f'{slice_num}', converted_case=f'{i}')
    
    print('Converted all {} cases to 2D slices'.format(len(files)))
    print('Total {} slices was generated'.format(slice_num))

def main(path, saved_dir, val_ratio, data):
    if val_ratio != 0:
        train_dir = os.path.join(saved_dir, 'train')
        val_dir = os.path.join(saved_dir, 'val')

        train_files, val_files = traverse_files(path, val_ratio, data)

        volume_to_2d(train_files, train_dir, data)
        volume_to_2d(val_files, val_dir, data)
    else:
        files = traverse_files(path, val_ratio, data)
        volume_to_2d(files, saved_dir, data)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='synapse',
                        help='acdc or synapse')
    parser.add_argument('--path', type=str,
                        help='root directory for datasets')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--saved-dir', type=str,
                        help='output directory for 2D slices')
    
    args = parser.parse_args()

    main(args.path, args.saved_dir, args.val_ratio, args.data)
    

    

    

        