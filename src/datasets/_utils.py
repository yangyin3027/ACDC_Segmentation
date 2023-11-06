import torch
from tqdm import tqdm

def compute_class_weights(dataloader, num_classes=7):
    '''Adapted from sklearn.utils.weights.compute_class_weights
    using n_samples/(n_classes * np.bincount(y))
    '''
    class_weights = []
    with tqdm(dataloader, desc='computing through dataloader',
                          total=len(dataloader)) as pbar:
        for _, labels in pbar:
            weights = torch.zeros(num_classes)
            classes, freqs = torch.unique(labels, return_counts=True)
            if torch.is_floating_point(classes):
                classes = classes.long()
            weights[classes] = labels.numel()/(classes.numel() * freqs)
            pbar.set_postfix(weights=weights.numpy().tolist())
            class_weights.append(weights)
    
    class_weights = torch.stack(class_weights)
    return class_weights.mean(0)

def compute_mean_std(dataloader, img_channel=1):
    '''compute mean and std for batch of images 
    for data normalization
    '''
    psum = torch.zeros(img_channel, dtype=torch.float32)
    psum_sq = torch.zeros(img_channel, dtype=torch.float32)
    counts = 0.0

    for data, _ in dataloader:
        psum += data.sum(dim=[0, 2, 3])
        psum_sq += (data**2).sum(dim=[0, 2, 3])
        counts += data.size(0) * data.size(2) * data.size(3)
    
    mean = psum / counts
    var = psum_sq / counts - mean**2
    std = torch.sqrt(var)
    return mean, std

if __name__ == '__main__':
    import argparse
    from dataloader import get_dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='root directory for datasets')
    parser.add_argument('--output-size', type=int, default=224)
    parser.add_argument('--data-type', type=str, default='synapse')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--task', type=str, default='class-weights')

    args = parser.parse_args()

    train_loader, val_loader = get_dataloader(args.root, args.batch_size,
                                              dds=False,
                                              data_type=args.data_type)

    if args.task == 'class-weights':
        class_weights = compute_class_weights(train_loader)
        print('computed class weights', class_weights)
    
    elif args.task == 'msd':
        mean, std = compute_mean_std(train_loader)
        print('mean: ', mean)
        print('std ', std)
    