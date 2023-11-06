from augment import *
from glob import glob
import h5py

from torch.utils.data import Dataset, DataLoader, distributed

def collate_fn(batch):
    images = [sample['image'] for sample in batch]
    labels = [sample['label'] for sample in batch]

    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

class SegDataset(Dataset):
    def __init__(self, root, type='train', transform=None):
        self.root = root
        self.type = type
        self.transform = transform

        if self.type =='train':
            self.sample_list = glob(self.root + '/train/*.h5')
        else:
            self.sample_list = glob(self.root + '/val/*.h5')
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        case = self.sample_list[index]
        h5f = h5py.File(case, 'r')
        image = h5f['image'][()].astype(np.float32)
        label = h5f['label'][()].astype(np.float32)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample, type=self.type)
        return sample

class SynpaseDataset(SegDataset):
    def __getitem__(self, index):
        '''reorient gt labels
        Original 13+1 classes reoriented to 6+1 classes
        old labels info:{1: spleen, 2: right kidney, 3: left kidney, 4: gallbladder,
            5: esophagus, 6: liver, 7: stomach, 8: aorta, 9: inferior vena cava,
            10: portal vein and splenic vein, 11: pancreas, 12 right adrenal gland,
            13: left adrenal gland}
        new labels: {1: spleen, 2: kidneys, 3: liver, 4: stomach, 5: aorta, 6: pancreas}'''
        case = self.sample_list[index]
        h5f = h5py.File(case, 'r')
        image = h5f['image'][()].astype(np.float32)
        label = h5f['label'][()].astype(np.float32)
        
        # reorient labels
        label[label==3.] = 2.0
        label[label==6.] = 3.0
        label[label==7.] = 4.0
        label[label==8.] = 5.0
        label[label==11.] = 6.0
        label[label>6.] = 0.

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample, type=self.type)
        return sample
        
def get_dataloader(root,
                   batch_size=24,
                   collate_fn=collate_fn,
                   output_size=256,
                   transform=True,
                   dds=True,
                   data_type='synapse',
                   **kwargs):
    if isinstance(output_size, int):
        output_size = tuple((output_size, output_size))
    if transform:
        transform = RandomAugment(output_size)
    
    if data_type == 'synapse':
        train_datasets = SynpaseDataset(root=root,
                            type='train',
                            transform=transform
                            )
        val_datasets = SynpaseDataset(root=root,
                                type='val',
                                transform=transform
                                )
    else:
        train_datasets = SegDataset(root=root,
                            type='train',
                            transform=transform
                            )
        val_datasets = SegDataset(root=root,
                                type='val',
                                transform=transform
                                )
    if dds:
        train_data_sampler = distributed.DistributedSampler(
            train_datasets, shuffle=True
        )
        val_data_sampler = distributed.DistributedSampler(
            val_datasets, shuffle=True
        )
        
    else:
        train_data_sampler = None
        val_data_sampler = None
    
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  shuffle=(train_data_sampler is None),
                                  collate_fn=collate_fn,
                                  sampler=train_data_sampler,
                                  drop_last=True,
                                  **kwargs
                                  )
    val_dataloader = DataLoader(val_datasets,
                                batch_size=batch_size,
                                shuffle=(val_data_sampler is None),
                                collate_fn=collate_fn,
                                sampler=val_data_sampler,
                                drop_last=True,
                                **kwargs)
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='root directory contained train and val datasets')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-size', type=int, default=224)
    
    args = parser.parse_args()

    train_dataloader, val_dataloader = get_dataloader(args.root,
                                                      args.batch_size,
                                                      output_size=args.output_size,
                                                      dds=False)
    images, labels = next(iter(train_dataloader))
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.util import montage as smon

    images = [img.squeeze().numpy() for img in images]
    labels = [gt.numpy() for gt in labels]

    images = smon(images, grid_shape=(4,8), fill=0, padding_width=2,)
    labels = smon(labels, grid_shape=(4,8), fill=0, padding_width=2)

    mask = np.ma.masked_where(labels==0, labels)
    plt.imshow(images, cmap='gray')
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.show()

