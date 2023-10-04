from datasets import *
from network import *
from metrics_acdc import *
from nii_acdc import traverse_files

import nibabel as nib
import torchvision.transforms.functional as F
import numpy as np
from math import ceil
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from skimage.util import montage as smon


class Process(object):
    def __init__(self, mean=(55.6606,), std=(79.5647,),
                 size=224):
        self.mean = mean
        self.std = std
        self.size = 224
        self.img_size = None
    
    def _pad(self, image):
        h, w = image.shape[-2:]
        padded = []
        for i in range(len(image)):
            img = image[i]
            if h < w:
                pad_width = (((w-h)//2, w-h-(w-h)//2), (0, 0))
            else:
                pad_width = ((0, 0), ((h-w)//2, h-w-(h-w)//2))
            
            img = np.pad(img, pad_width, mode='reflect')
            padded.append(img)
        padded = np.stack(padded)

        return np.stack(padded)

    def preprocess(self, image):
        h, w = image.shape[-2: ]

        if h != w:
            image = self._pad(image)

        if not isinstance(image, torch.Tensor):
            image = torch.Tensor(image)

        # reserve original img_size
        self.img_size = (h, w)

        if self.mean is not None and self.std is not None:
            image = F.normalize(image, self.mean, self.std)
        if self.size is not None:
            image = F.resize(image, self.size)

        return image

    def postprocess(self, preds: torch.tensor):
        '''
        Args:
            preds: unnormalized logits as the model output
            img_size: tuple of (h, w) of original image
        Returns:
            segmentation_masks: numpy.ndarray
        '''
        preds = torch.nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)

        if self.img_size is not None:
            h, w = self.img_size
            preds = F.resize(preds, max(h, w))

            if h > w:
                start = (h-w)//2
                end = (h-w)//2 + w
                preds = preds[:, :, start:end]
            elif h < w:
                start = (w-h)//2
                end = (w-h)//2 + h
                preds = preds[:, start:end, :]
        return preds.detach().cpu().numpy()

class Inference(object):
    def __init__(self, model, 
                    checkpoint_file=None,
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
    
    def predict(self, image_file, save=False):
        '''
        image_file: nifti file
        save: booleen value
        '''
        image, affine, header = load_nii(image_file)

        # preprocess the image for model
        process = Process()
        processed_image = process.preprocess(image)
        
        # perform model prediction
        self.model.to(self.device)
        self.model.eval()
    
        processed_image = processed_image.to(device=self.device,
                                             dtype=torch.float32)

        # add channel dimension as 1
        processed_image.unsqueeze_(1)
        
        with torch.no_grad():
            preds = self.model(processed_image)
        # postprocess the prediction
        pred_masks = process.postprocess(preds)

        if save:
            saved_fname = image_file.replace('.nii.gz', '_pred.nii.gz')
            save_nii(saved_fname, pred_masks.transpose(1,2,0), affine, header)

        return pred_masks
    
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
            fig.savefig('predicted.png', transparent=False)
        
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

def compute_metrics(model, checkpoint, image_files):
    image_files = sorted(image_files, key=natural_order)
    inference = Inference(model, checkpoint)
    res = []
    for i, img_file in enumerate(image_files):
        image, _, _ = load_nii(img_file)
        gt, _, header = load_nii(img_file.replace('.nii.gz', '_gt.nii.gz'))
        zoom = header.get_zooms()
        pred = inference.predict(img_file)

        res.append(metrics(gt, pred, zoom))
        print("{} been processed".format(img_file))
        print("{} cases have been process".format(i+1))
    
    lst_name_gt = [os.path.basename(gt).split('.')[0] for gt in image_files]
    res = [[n, ] + r for r, n in zip(res, lst_name_gt)]
    df = pd.DataFrame(res, columns=HEADER)
    df.to_csv('results_{}.csv'.format(time.strftime("%Y%m%d_%H%M%S")), index=False)
    return df

def remove_files(files):
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    import argparse
    import random
    import subprocess
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../ACDC_datasets/testing')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--checkpoint', type=str, default='./src/tmp/exp13/checkpoint.pth.tar')
    parser.add_argument('--cpu', action='store_true',default=False )
    parser.add_argument('--type', type=str, default='animation',
                        help='inference types: prediction, plot, animation, metrics')

    args = parser.parse_args()

    files = traverse_files(args.data)
    files = sorted([f for f in files if f.find('_pred') == -1
                   and f.find('_gt') == -1])

    model = AttenUnet()

    if args.type == 'plot':

        random.seed(args.seed)
        image_file = files[random.randint(0, len(files))]

        inference = Inference(model, args.checkpoint)
        inference.plot(image_file)
    
    elif args.type == 'animation':
        random.seed(args.seed)
        image_file = files[random.randint(0, len(files))]

        inference = Inference(model, args.checkpoint)
        inference.plot_animation(image_file)
    
    elif args.type == 'metrics':

        df = compute_metrics(model, args.checkpoint, files)
        print('Mean Dice LV:\t ', df['Dice LV'].mean())
        print('Mean Dice RV:\t', df['Dice RV'].mean())
        print('Mean Dice MYO:\t', df['Dice MYO'].mean())
    
    else:
        random.seed(args.seed)
        image_file = files[random.randint(0, len(files))]

        inference = Inference(model, args.checkpoint)
        preds = inference.predict(image_file)

        print("{} predicted masks as shape {}".format(image_file, preds.shape))



    
