from datasets import *
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

class Process(object):
    def __init__(self, size=None):
        self.size = size
        if self.size is None:
            self.size = 224
        self.origin_size = None
    
    def _preprocess(self, image):
        processed_image = []
        slices = image.shape[-1]
        for s in range(slices):
            img = image[:, :, s]
            p5, p95 = np.percentile(img, (0.5, 99.5))
            img = np.clip(img, p5, p95)
            img = (img - img.min()) / (img.max() - img.min())
            
            img = self._transform(img)
            processed_image.append(img)
        
        processed_image = torch.stack(processed_image)
        return processed_image
    
    def _transform(self, img):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1365,),(0.1661,))
        ])
        img = (img * 255).astype(np.uint8)
        img = transform(img)
        return img

    def preprocess(self, image):
        h, w = image.shape[:2]

        # memorize original image size
        self.origin_size = (h, w)
        # min-max normalization
        image = self._preprocess(image)
        # apply tensor and resize transform
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
                         interpolation=F.InterpolationMode.BILINEAR,
                         antialias=False)
        preds = torch.nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)
        preds = preds.detach().cpu().numpy()
        return preds

class Inference(object):
    def __init__(self, model, 
                    checkpoint_file=None,
                    size=None,
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
        if self.size is None:
            self.size =  224

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
        
        preds = self._predict(processed_image)

        # postprocess the prediction
        pred_masks = process.postprocess(preds)

        if save:
            saved_fname = image_file.replace('.nii.gz', '_pred.nii.gz')
            save_nii(saved_fname, pred_masks.transpose(1,2,0), affine, header)

        return pred_masks

    def predict_h5py(self, h5_file):
        h5f = h5py.File(h5_file, 'r')
        image = h5f['image'][()].astype(np.float32)
        image = np.expand_dims(image, -1)
        process = Process(self.size)
        processed_image = process.preprocess(image)

        if processed_image.ndim < 4:
            processed_image.unsqueeze_(1)
        
        preds = self._predict(processed_image)
        preds_masks = process.postprocess(preds)
        return preds_masks
    
    def plot(self, image_file, save=False):

        image, _, _ = load_nii(image_file)
        gt, _, _ = load_nii(image_file.replace('.nii.gz', '_gt.nii.gz'))
        pred = self.predict(image_file)

        # make a montage image
        grid_shape = (2, ceil(len(image)/2))
        mimage = smon(image, grid_shape=grid_shape, fill=0)
        mgt = smon(gt, grid_shape=grid_shape, fill=0)
        mpred = smon(pred, grid_shape=grid_shape, fill=0)

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
            fig.savefig('predicted.png', transparent=False,
                        bbox_inches='tight', pad_inches=0)
        
        plt.tight_layout()
        plt.show()
    
    def plot_animation(self, image_file, save=False):

        image, _, _ = load_nii(image_file)
        gt, _, _ = load_nii(image_file.replace('.nii.gz', '_gt.nii.gz'))
        pred = self.predict(image_file)

        gt_ma = np.ma.masked_where(gt==0, gt)
        pred_ma = np.ma.masked_where(pred==0, pred)

        fig, axes = plt.subplots(1, 2, layout='constrained')
        aximg_0 = axes[0].imshow(image[0], cmap='gray')
        aximg_1 = axes[1].imshow(image[0], cmap='gray')
        
        axgt = axes[0].imshow(gt_ma[0], alpha=0.7, cmap='jet')
        axpred = axes[1].imshow(pred_ma[0], alpha=0.7, cmap='jet')

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
                                       frames=range(len(image)),
                                       interval=500)
        plt.tight_layout()
        plt.show()
        if save:
            anim.save('predicted.mp4')

def compute_metrics(model, checkpoint, image_files, size=None, fname='ED'):
    image_files = sorted(image_files, key=natural_order)
    inference = Inference(model, checkpoint,
                          size=size)
    res = []
    for i, img_file in tqdm(enumerate(image_files),
                            desc=f'Segmentation prediction on {fname}',
                            total=len(image_files)):
        image, _, _ = load_nii(img_file)
        gt, _, header = load_nii(img_file.replace('.nii.gz', '_gt.nii.gz'))
        zoom = header.get_zooms()
        pred = inference.predict(img_file)
        gt = gt.astype(np.uint8)

        res.append(metrics(gt, pred, zoom))
    
    lst_name_gt = [os.path.basename(gt).split('.')[0] for gt in image_files]
    res = [[n, ] + r for r, n in zip(res, lst_name_gt)]
    df = pd.DataFrame(res, columns=HEADER)
    df.to_csv('{}_{}_{}.csv'.format(fname, model.__class__.__name__,time.strftime("%Y%m%d_%H%M%S")), index=False)
    return df

def traverse_files(path='../ACDC_datasets'):
    subdirs = [root for root, _, _ in os.walk(path)]
    files = []

    for dir in subdirs:
        new_files = glob(dir + '/*.nii.gz')
        # remove files contain '_4d'
        new_files = [f for f in new_files if f.find('_4d')==-1]
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
    parser.add_argument('--data', type=str, default='../ACDC_datasets/testing')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--checkpoint', type=str, default='./src/tmp/exp13/checkpoint.pth.tar')
    parser.add_argument('--cpu', action='store_true',default=False )
    parser.add_argument('--type', type=str, default='animation',
                        help='inference types: prediction, plot, animation, metrics')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--size', type=int, default=256)

    args = parser.parse_args()

    files = traverse_files(args.data)
    files = sorted([f for f in files if f.find('_pred') == -1
                   and f.find('_gt') == -1])
    files_ed = [f for f in files if f.find('_frame01') != -1]
    files_es = [f for f in files if f not in files_ed]

    model = UNet()

    if args.type == 'plot':

        random.seed(args.seed)
        image_file = files[random.randint(0, len(files))]

        inference = Inference(model, args.checkpoint,
                              size=args.size)
        inference.plot(image_file, args.save)
    
    elif args.type == 'animation':
        random.seed(args.seed)
        image_file = files[random.randint(0, len(files))]

        inference = Inference(model, args.checkpoint,
                              size=args.size)
        inference.plot_animation(image_file, args.save)
    
    elif args.type == 'metrics':

        df_ed = compute_metrics(model, args.checkpoint, files_ed,
                             size=args.size, fname='ED')
        df_es = compute_metrics(model, args.checkpoint, files_es,
                             size=args.size, fname='ES')
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



    
