import scipy.ndimage as ndim
from network import *
from metrics_acdc import *
from losses import dice_coeff

import nibabel as nib
import torchvision.transforms.functional as F
import numpy as np
from math import ceil
import os
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from skimage.util import montage as smon

from tqdm import tqdm
from math import ceil

class Process(object):
    def __init__(self, size=256):
        self.size = size
        if not isinstance(self.size, (tuple, list)):
            self.size = tuple((self.size, self.size))
        self.origin_size = None
    
    def preprocess(self, image):
        p5, p95 = np.percentile(image, (0.5, 99.5))
        image = np.clip(image, p5, p95)
        image = (image - np.min(image, axis=(0,1))) / (np.max(image, axis=(0,1)) - np.min(image, axis=(0,1)))
        self.origin_size = image.shape[:-1]

        image = torch.from_numpy(image).permute(-1, 0, 1)
        if self.origin_size[0] != self.size[0] or self.origin_size[1] != self.size[1]:         
            image = F.resize(image, self.size,
                             antialias=False)
        return image
    
    def postprocess(self, preds: torch.tensor):
        '''
        Args:
            preds: unnormalized logits as the model output
            img_size: tuple of (h, w) of original image
        Returns:
            segmentation_masks: numpy.ndarray
        '''
        preds = F.resize(preds, self.origin_size, 
                         interpolation=F.InterpolationMode.NEAREST,
                         antialias=False)
        preds = torch.nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)
        preds = preds.detach().cpu().numpy()
        return preds

class Inference(object):
    def __init__(self, model, 
                    checkpoint_file=None,
                    size=256,
                    batch_size=16,
                    device=None):
        '''
        Load pretrained model weight to predict mask for inividual 
        nifti image file
        Args:
            model: pytorch model (unet or attenunet)
            checkpoint_file: saved _pth.tar weight file
        '''
        self.model = model
        if checkpoint_file is not None:
            self.model.load_state_dict(
                torch.load(checkpoint_file,
                           map_location=torch.device('cpu'))
            )
        if device is None:

            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                torch.cuda.empty_cache()
            else:
                self.device = device
        else:
            self.device = device 
        self.size = size
        self.batch_size = batch_size
    
    def _load_gt(self, gt_file):
        '''reorient gt labels
        old labels info of synapse:{1: spleen, 2: right kidney, 3: left kidney, 4: gallbladder,
            5: esophagus, 6: liver, 7: stomach, 8: aorta, 9: inferior vena cava,
            10: portal vein and splenic vein, 11: pancreas, 12 right adrenal gland,
            13: left adrenal gland}
        '''
        gt, _, _ = load_nii(gt_file)
        gt[gt!=6.] = 0.
        gt[gt==6.] = 1.
        return gt

    def _predict(self, image):
        self.model.to(self.device)
        self.model.eval()

        image = image.to(device=self.device,
                         dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(image)
        return preds  
    
    def predict(self, image_file, save=False):
        '''
        image_file: nifti file
        save: booleen value
        '''
        nimage = nib.load(image_file)
        image = nimage.get_fdata()
        affine, header = nimage.affine, nimage.header

        # preprocess the image for model
        process = Process(size=self.size)
        processed_image = process.preprocess(image)

        if processed_image.ndim < 4:
            processed_image.unsqueeze_(1)
                
        if processed_image.shape[0] > self.batch_size:
            chunk_size = ceil(processed_image.shape[0]/self.batch_size)
            processed_image = torch.chunk(processed_image, chunk_size, dim=0)

            preds = []
            for images in processed_image:
                preds.append(self._predict(images))

            preds = torch.cat(preds, dim=0)
        else:
            preds = self._predict(processed_image)

        # postprocess the prediction
        pred_masks = process.postprocess(preds)

        if save:
            saved_fname = image_file.replace('.nii.gz', '_pred.nii.gz')
            save_nii(saved_fname, pred_masks.transpose(1,2,0), affine, header)

        return pred_masks
    
    def plot(self, image_file, gt_file, save=False):

        image, _, _ = load_nii(image_file)
        gt = self._load_gt(gt_file)

        pred = self.predict(image_file)

        # filter non liver contained slices
        non_zero_index = [i for i in range(len(gt)) if np.sum(gt[i]) != 0]
        image = image[non_zero_index]
        gt = gt[non_zero_index]
        pred = pred[non_zero_index]

        # make a montage image
        if len(image)//2 <= 10:
            grid_shape = (2, ceil(len(image)/2))
        else:
            grid_shape = (ceil(len(image)/10), 10)

        mimage = smon(image.transpose(0,2,1), grid_shape=grid_shape, fill=0.)
        mgt = smon(gt.transpose(0,2, 1), grid_shape=grid_shape, fill=0.)
        mpred = smon(pred.transpose(0, 2, 1), grid_shape=grid_shape, fill=0.)

        gt_ma = np.ma.masked_where(mgt==0, mgt)
        pred_ma = np.ma.masked_where(mpred==0, mpred)

        fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                                 gridspec_kw=dict(hspace=0))
        axes[0].imshow(mimage, cmap='gray')
        axes[0].imshow(gt_ma, alpha=0.7, cmap='jet')
        axes[0].set_title('Ground Truth Segmentation')

        axes[1].imshow(mimage, cmap='gray')
        axes[1].imshow(pred_ma, alpha=0.7, cmap='jet')
        axes[1].set_title('Predicted Segmentation')

        for a in axes:
            a.axis('off')
        
        if save:
            fig.savefig('./src/examples/predicted.png', transparent=False,
                        bbox_inches='tight', pad_inches=0)
        
        plt.tight_layout()
        plt.show()
    
    def plot_animation(self, image_file, gt_file, save=False):

        image, _, _ = load_nii(image_file)
        pred = self.predict(image_file)

        gt = self._load_gt(gt_file)

         # filter non liver contained slices
        non_zero_index = [i for i in range(len(gt)) if np.sum(gt[i]) != 0]
        image = image[non_zero_index]
        gt = gt[non_zero_index]
        pred = pred[non_zero_index]

        gt_ma = np.ma.masked_where(gt==0., gt)
        pred_ma = np.ma.masked_where(pred==0., pred)

        fig, axes = plt.subplots(1, 2, layout='constrained')
        aximg_0 = axes[0].imshow(image[-1], cmap='gray')
        aximg_1 = axes[1].imshow(image[-1], cmap='gray')
        
        axgt = axes[0].imshow(gt_ma[-1], alpha=0.7, cmap='jet')
        axpred = axes[1].imshow(pred_ma[-1], alpha=0.7, cmap='jet')

        axes[0].set_title('Ground Truth Segmentation')
        axes[1].set_title('Predicted Segmentation')
        for a in axes:
            a.axis("off")
        
        def update(frame):
            aximg_0.set_data(image[frame])
            aximg_1.set_data(image[frame])
            axgt.set_data(gt_ma[frame])
            axpred.set_data(pred_ma[frame])
        
        anim = animation.FuncAnimation(fig=fig, func=update,
                                       frames=range(len(image)-1, -1, -1),
                                       interval=500)
        plt.show()
        if save:
            anim.save('./src/examples/predicted.mp4')

def compute_metrics(model, checkpoint, image_files, gt_file, size=None, fname='ED'):
    image_files = sorted(image_files, key=natural_order)
    inference = Inference(model, checkpoint,
                          size=size)
    res = []
    for i, img_file in tqdm(enumerate(image_files),
                            desc=f'Segmentation prediction on {fname}',
                            total=len(image_files)):
        image, _, _ = load_nii(img_file)
        gt, _, header = load_nii(gt_file)
        zoom = header.get_zooms()
        pred = inference.predict(img_file)
        gt = gt.astype(np.uint8)

        res.append(metrics(gt, pred, zoom))
    
    lst_name_gt = [os.path.basename(gt).split('.')[0] for gt in image_files]
    res = [[n, ] + r for r, n in zip(res, lst_name_gt)]
    df = pd.DataFrame(res, columns=HEADER)
    # df.to_csv('{}_{}_{}.csv'.format(fname, model.__class__.__name__,time.strftime("%Y%m%d_%H%M%S")), index=False)
    return df

def traverse_files(path='../ACDC_datasets', data='acdc'):
    subdirs = [root for root, _, _ in os.walk(path)]
    files = []

    for dir in subdirs:
        new_files = glob(dir + '/*.nii.gz')
        # remove files contain '_4d'
        if data == 'acdc':
            new_file = [f for f in new_files 
                        if (f.find('_4d')==-1 and f.find('_gt')==-1)]
        elif data == 'synapse':
            new_files = [f for f in new_files 
                         if f.find('label')==-1]
        files.extend(new_files)
    return files

def remove_files(files):
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    import argparse
    import random
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../ACDC_datasets/testing')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--checkpoint', type=str, default='./src/tmp/exp13/checkpoint.pth.tar')
    parser.add_argument('--cpu', action='store_true',default=False )
    parser.add_argument('--type', type=str, default='animation',
                        help='inference types: prediction, plot, animation, metrics')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--model', type=str, default='attenunet')
    parser.add_argument('--fname', type=str, default='result')
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--img-channels', type=int, default=1)
    parser.add_argument('--data-type', type=str, default='synapse')
    parser.add_argument('--batch-size', type=int, default=16)

    args = parser.parse_args()

    files = traverse_files(args.root, data=args.data_type)
    
    if args.data_type == 'acdc':
        files_ed = [f for f in files if f.find('_frame01') != -1]
        files_es = [f for f in files if f not in files_ed]

    if args.model == 'attenunet':
        model = AttenUnet(args.num_classes, args.img_channels)
    elif args.model == 'attenunet_shallow':
        model = AttenUnet_shallow(args.num_classes, args.img_channels)
    elif args.model == 'unet':
        model = UNet(args.num_classes, args.img_channels)
    elif args.model == 'unet_shallow':
        model = UNet_shallow(args.num_classes, args.img_channels)
    else:
        NotImplementedError
        
    if args.type == 'plot':

        random.seed(args.seed)
        image_file = files[random.randint(0, len(files))]

        inference = Inference(model, args.checkpoint,
                              size=args.size, batch_size=args.batch_size)
        if args.data_type == 'synapse':
            gt_file = image_file.replace('img', 'label')
        elif args.data_type == 'acdc':
            gt_file = image_file.replace('.nii.gz', '_gt.nii.gz')
        inference.plot(image_file, gt_file=gt_file, save=args.save)
    
    elif args.type == 'animation':
        random.seed(args.seed)
        image_file = files[random.randint(0, len(files))]

        inference = Inference(model, args.checkpoint,
                              size=args.size,
                              batch_size=args.batch_size)
        if args.data_type == 'synapse':
            gt_file = image_file.replace('img', 'label')
        elif args.data_type == 'acdc':
            gt_file = image_file.replace('.nii.gz', '_gt.nii.gz')
        inference.plot_animation(image_file, gt_file,args.save)
    
    elif args.type == 'metrics':

        df_ed = compute_metrics(model, args.checkpoint, files_ed,
                             size=args.size, fname='ED')
        df_es = compute_metrics(model, args.checkpoint, files_es,
                             size=args.size, fname='ES')
        df_ed['mean Dice LV'] = df_ed['Dice LV'].mean()
        df_ed['mean Dice RV'] = df_ed['Dice RV'].mean()
        df_ed['mean Dice MYO'] = df_ed['Dice MYO'].mean()

        df_es['mean Dice LV'] = df_es['Dice LV'].mean()
        df_es['mean Dice RV'] = df_es['Dice RV'].mean()
        df_es['mean Dice MYO'] = df_es['Dice MYO'].mean()

        dirname = os.path.dirname(args.checkpoint)
        fname = os.path.join(dirname, args.fname)
        df_ed.to_csv(f'{fname}_ed.csv')
        df_es.to_csv(f'{fname}_es.csv')
        print(f'{fname} results saved!')
        
        print('ED Mean Dice LV:\t ', df_ed['Dice LV'].mean())
        print('ED Mean Dice RV:\t', df_ed['Dice RV'].mean())
        print('ED Mean Dice MYO:\t', df_ed['Dice MYO'].mean())

        print('ES Mean Dice LV:\t ', df_es['Dice LV'].mean())
        print('ES Mean Dice RV:\t', df_es['Dice RV'].mean())
        print('ES Mean Dice MYO:\t', df_es['Dice MYO'].mean())
    
    else:
        random.seed(args.seed)
        image_file = files[random.randint(0, len(files))]

        inference = Inference(model, args.checkpoint,
                              size=args.size)
        preds = inference.predict(image_file)

        print("{} predicted masks as shape {}".format(image_file, preds.shape))



    
