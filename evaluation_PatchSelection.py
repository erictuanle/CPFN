# Importation of packages
import os
import sys
import h5py
import torch
import argparse
import numpy as np

# Importing the Dataset file
from Dataset import dataloaders
# Importing the Network file
from PointNet2 import pn2_network
# Importing the Utils files
from Utils import config_loader, sampling_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='YAML configuration file', default='Configs/config_patchSelec.yml')
    parser.add_argument('--lowres_dataset', help='Directory of the Lowres Dataset', default=os.path.expanduser('data/TraceParts_v2_lowres/'))
    parser.add_argument('--highres_dataset', help='Directory of the Highres Dataset', default=os.path.expanduser('data/TraceParts_v2/'))
    parser.add_argument('--heatmap_folder', help='Directory to save the heatmaps in', default=os.path.expanduser('data/TraceParts_v2_heatmaps/'))
    parser.add_argument('--scale', help='Scale of the Primitives', type=float, default=0.05)
    args = parser.parse_args()

    heatmap_folder = os.path.join(args.heatmap_folder, str(args.scale))
    os.makedirs(heatmap_folder, exist_ok=True)

    # Loading the config file
    conf = config_loader.Patch_SelecConfig(args.config_file)

    # Selecting the visible GPUs
    visible_GPUs = conf.get_CUDA_visible_GPUs()
    device = torch.device('cuda')
    if visible_GPUs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(visible_GPUs)

    # Test Dataset
    csv_path_test = os.path.join('Dataset', conf.get_test_data_file())
    noisy_test = conf.get_test_data_first_n()
    first_n_test = conf.is_test_data_noisy()

    # Launching the Network
    if args.scale<1:
    	patchselec_module_filename = 'patchselec_%s_module'%str(round(args.scale, 2)) + '.pth'
    	patchselec_module = pn2_network.PointNet2(dim_input=3, dim_pos=3, output_sizes=[2]).to(device)
    	dict = torch.load(os.path.join(conf.get_weights_folder(), patchselec_module_filename))
    	patchselec_module.load_state_dict(dict, strict=True)
    	patchselec_module.eval()

    test_dataset = dataloaders.Dataset_PatchSelection(csv_path_test, args.lowres_dataset, args.highres_dataset, args.scale, n_points=8192, normalisation=True)
    test_datasampler = dataloaders.Sampler(data_source=test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_datasampler, batch_size=1, num_workers=0, pin_memory=True)

    # Initialisation
    if args.scale<1:
    	confusion_matrix = np.zeros([2, 2])
    for batch_id, data in enumerate(test_dataloader, 0):
        if batch_id%100==0: print('Iteration %d / %d' % (batch_id, len(test_dataloader)))
        # Computing the prediction
        points = data[0].type(torch.FloatTensor).to(device)
        batch_size_current, num_points, _ = points.size()
        output_labels = data[1].type(torch.LongTensor).to(device)
        shuffled_indices = data[2].type(torch.LongTensor).to(device)
        if args.scale<1:
        	predicted_labels = patchselec_module(points)[0]
        	predicted_labels = torch.argmax(predicted_labels, dim=2)
        else:
        	predicted_labels = output_labels[0]
        if not os.path.isdir(os.path.join(heatmap_folder, test_dataset.hdf5_file_list_lowres[batch_id].split('/')[-1].replace('.h5',''))):
            os.mkdir(os.path.join(heatmap_folder, test_dataset.hdf5_file_list_lowres[batch_id].split('/')[-1].replace('.h5', '')))
        # Computing the confusion matrix
        if args.scale<1:
        	confusion_matrix[0, 0] += torch.sum((predicted_labels == 0) * (output_labels == 0)).item()
        	confusion_matrix[0, 1] += torch.sum((predicted_labels == 0) * (output_labels == 1)).item()
        	confusion_matrix[1, 0] += torch.sum((predicted_labels == 1) * (output_labels == 0)).item()
        	confusion_matrix[1, 1] += torch.sum((predicted_labels == 1) * (output_labels == 1)).item()
        	predicted_labels = torch.gather(predicted_labels[0], 0, shuffled_indices[0])
        # Selecting the indices
        with h5py.File(os.path.join(args.highres_dataset, test_dataset.hdf5_file_list_lowres[batch_id].split('/')[-1]), 'r') as f:
            gt_points_hr = f['gt_points'][()]
            gt_labels_hr = f['gt_labels'][()]
        with h5py.File(os.path.join(os.path.join(args.lowres_dataset, test_dataset.hdf5_file_list_lowres[batch_id].split('/')[-1])), 'r') as f:
            gt_points_lr = f['gt_points'][()]
            gt_labels_lr = f['gt_labels'][()]
        pool_indices = np.where(predicted_labels.detach().cpu().numpy())[0]
        if len(pool_indices) > 0:
            patch_indices = sampling_utils.sample(gt_points_lr, gt_points_hr, pool_indices, max_number_patches=len(pool_indices))
            np.save(os.path.join(heatmap_folder, test_dataset.hdf5_file_list_lowres[batch_id].split('/')[-1].replace('.h5',  '_indices.npy')), patch_indices)
    if args.scale<1:
    	confusion_matrix = confusion_matrix / np.sum(confusion_matrix)
    	print('Confusion Matrix', confusion_matrix)
    	np.save(os.path.join(heatmap_folder, 'confusion_matrix.npy'), confusion_matrix)