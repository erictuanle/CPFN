# Importation of packages
import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd

# Importing the Dataset files
from Dataset import dataloaders
# Importing the Network files
from SPFN import fitter_factory, metric_implementation, losses_implementation
from PointNet2 import pn2_network
# Importing Utils files
from Utils import config_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='YAML configuration file', default='Configs/config_globalSPFN.yml')
    parser.add_argument('--lowres_dataset', help='Directory of the Lowres Input Dataset', default=os.path.expanduser('data/TraceParts_v2_lowres/'))
    parser.add_argument('--highres_dataset', help='Directory of the Highres Input Dataset', default=os.path.expanduser('data/TraceParts_v2/'))
    parser.add_argument('--path_patches', help='Path to Sampled Patches h5 files', default=os.path.expanduser('data/TraceParts_v2_patches/'))
    parser.add_argument('--scale', help='Scale to select the smallest primitive', default=0.05, type=float)
    parser.add_argument('--output_folder', help='Directory of the output folder', default=os.path.expanduser('data/TraceParts_v2_globalspfn/'))
    parser.add_argument('--evaluation_set', help='Whether to evaluate on the train or test set', default='test')
    args = parser.parse_args()

    path_patches = os.path.join(args.path_patches, str(round(args.scale,2)))

    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    # Loading the config file
    conf = config_loader.SPFNConfig(args.config_file)

    # Selecting the visible GPUs
    visible_GPUs = conf.get_CUDA_visible_GPUs()
    device = torch.device('cuda')
    if visible_GPUs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(visible_GPUs)

    # Primtive Types and Numbers
    fitter_factory.register_primitives(conf.get_list_of_primitives())
    n_registered_primitives = fitter_factory.get_n_registered_primitives()
    n_max_global_instances = conf.get_n_max_global_instances()

    # Test Dataset
    if args.evaluation_set == 'train':
        csv_path_test = os.path.join('Dataset', conf.get_train_data_file())
    else:
        csv_path_test = os.path.join('Dataset', conf.get_test_data_file())
    noisy_test = conf.is_test_data_noisy()
    first_n_test = conf.get_test_data_first_n()

    # Launching the Network
    spfn_module_filename = 'globalspfn_module.pth'
    spfn_module = pn2_network.PointNet2(dim_input=3, dim_pos=3, output_sizes=[3, n_registered_primitives, n_max_global_instances]).to(device)
    dict = torch.load(os.path.join(conf.get_weights_folder(), spfn_module_filename))
    spfn_module.load_state_dict(dict, strict=True)
    spfn_module.eval()

    test_dataset = dataloaders.Dataset_GlobalSPFN(n_max_global_instances, csv_path_test, args.lowres_dataset, args.highres_dataset, path_patches, noisy_test, test=True, n_points=None, first_n=first_n_test, fixed_order=True)
    test_datasampler = dataloaders.Sampler(data_source=test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_datasampler, batch_size=1, num_workers=0, pin_memory=True)

    dataframe_results = pd.DataFrame(columns=['Filename', 'mIoU', 'Type', 'Normal', 'Axis', 'MeanRes', 'StdRes', 'SkCoverage0.01', 'SkCoverage0.02', 'PCoverage0.01', 'PCoverage0.02'])
    list_mIoU = []
    for batch_id, data in enumerate(test_dataloader, 0):
        if batch_id%100==0: print('Iteration %d / %d' % (batch_id, len(test_dataloader)))
        P = data[0].type(torch.FloatTensor).to(device)
        X_gt = data[1].type(torch.FloatTensor).to(device)
        points_per_instance = data[2].type(torch.FloatTensor).to(device)
        I_gt = data[3].type(torch.LongTensor).to(device)
        T_gt = data[4].type(torch.LongTensor).to(device)
        plane_n_gt = data[5].type(torch.FloatTensor).to(device)
        cylinder_axis_gt = data[6].type(torch.FloatTensor).to(device)
        cone_axis_gt = data[7].type(torch.FloatTensor).to(device)
        patch_centers = data[8].type(torch.LongTensor).to(device)
        gt_parameters = {'plane_normal': plane_n_gt, 'cylinder_axis': cylinder_axis_gt, 'cone_axis': cone_axis_gt}
        glob_features = None
        loc_features = None
        if not os.path.isdir(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5',''))):
            os.mkdir(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5','')))
        with torch.no_grad():
            X, T, W, global_feat, local_feat = spfn_module(P, glob_features=glob_features, loc_features=loc_features)
            if args.evaluation_set == 'test':
                np.save(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5',''), 'local_feat_full.npy'), local_feat[0].cpu().numpy())
            local_feat = local_feat[:,:,patch_centers[0]]
            X = X / torch.norm(X, dim=2, keepdim=True)
            W = torch.softmax(W, dim=2)
        with torch.no_grad():
            W = metric_implementation.hard_W_encoding(W)
            matching_indices, mask = metric_implementation.hungarian_matching(W, I_gt)
            mask = mask.float()
            mIoU = metric_implementation.compute_segmentation_iou(W, I_gt, matching_indices, mask)
       
        if not os.path.isdir(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5', ''))):
        	os.mkdir(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5', '')))
        if args.evaluation_set == 'test':
        	np.save(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5', ''), 'object_seg.npy'), W[0].cpu().numpy())
        	np.save(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5', ''), 'object_normals.npy'), X[0].cpu().numpy())
        	np.save(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5', ''), 'object_type.npy'), T[0].cpu().numpy())

        mIoU, type_accuracy, normal_difference, axis_difference, mean_residual, std_residual, Sk_coverage, P_coverage, W, predicted_parameters, T = metric_implementation.compute_all_metrics(P, X, X_gt, W, I_gt, T, T_gt, points_per_instance, gt_parameters, list_epsilon=[0.01, 0.02], classes=conf.get_list_of_primitives())
        list_mIoU.append(mIoU.item())
        if batch_id%100==0: print('mIoU: ', np.mean(list_mIoU))
        dataframe_results.loc[batch_id] = [test_dataset.hdf5_file_list[batch_id].replace('.h5',''), mIoU.item(), type_accuracy.item(), normal_difference.item(), axis_difference.item(), mean_residual.item(), std_residual.item(), Sk_coverage[0].item(), Sk_coverage[1].item(), P_coverage[0].item(), P_coverage[1].item()]
        
        np.save(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5',''), 'global_feat.npy'), global_feat[0,:,0].cpu().numpy())
        np.save(os.path.join(args.output_folder, test_dataset.hdf5_file_list[batch_id].replace('.h5',''), 'local_feat.npy'), local_feat[0].cpu().numpy())
        
    dataframe_results.to_csv(os.path.join(args.output_folder, 'Results.csv'))