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
# Importing utils files
from Utils import config_loader, merging_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='YAML configuration file', default='Configs/config_localSPFN.yml')
    parser.add_argument('--lowres_dataset', help='Directory of the Lowres Input Dataset', default=os.path.expanduser('data/TraceParts_v2_lowres/'))
    parser.add_argument('--highres_dataset', help='Directory of the Highres Input Dataset', default=os.path.expanduser('data/TraceParts_v2/'))
    parser.add_argument('--dir_spfn', help='Directory of the global SPFN output', default=os.path.expanduser('data/TraceParts_v2_globalspfn/'))
    parser.add_argument('--dir_indices', help='Directory of the indices', default=os.path.expanduser('data/TraceParts_v2_heatmaps/'))
    parser.add_argument('--output_folder', help='Directory of the output folder', default=os.path.expanduser('data/TraceParts_v2_localspfn/'))
    parser.add_argument('--scale', help='Scale of the primitives', default=0.05, type=float)
    args = parser.parse_args()

    dir_indices = os.path.join(args.dir_indices, str(round(args.scale,2)))

    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    # Loading the config file
    conf = config_loader.Local_SPFNConfig(args.config_file)

    # Selecting the visible GPUs
    visible_GPUs = conf.get_CUDA_visible_GPUs()
    device = torch.device('cuda')
    if visible_GPUs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(visible_GPUs)

    # Primtive Types and Numbers
    fitter_factory.register_primitives(conf.get_list_of_primitives())
    n_registered_primitives = fitter_factory.get_n_registered_primitives()
    n_max_global_instances = conf.get_n_max_global_instances()
    n_max_local_instances = conf.get_n_max_local_instances()

    # Test Dataset
    csv_path_test = os.path.join('Dataset', conf.get_test_data_file())
    noisy_test = conf.get_test_data_first_n()
    first_n_test = conf.is_test_data_noisy()

    # Launching the Network
    spfn_module_filename = 'localspfn_%s_module.pth'%str(round(args.scale, 2))
    spfn_module = pn2_network.PointNet2(dim_input=3, dim_pos=3, output_sizes=[3, n_registered_primitives, n_max_local_instances], use_glob_features=False, use_loc_features=False).to(device)
    dict = torch.load(os.path.join(conf.get_weights_folder(), spfn_module_filename))
    spfn_module.load_state_dict(dict, strict=True)
    spfn_module.eval()

    test_dataset = dataloaders.Dataset_TestLocalSPFN(n_max_global_instances, n_max_local_instances, csv_path_test, args.dir_spfn, args.lowres_dataset, args.highres_dataset,
                                         dir_indices, noisy_test, first_n=first_n_test, fixed_order=True)
    test_datasampler = dataloaders.Sampler(data_source=test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_datasampler, batch_size=1, num_workers=0, pin_memory=True)

    dataframe_results = pd.DataFrame(columns=['Filename', 'mIoU', 'Type', 'Normal', 'Axis', 'MeanRes', 'StdRes', 'SkCoverage0.01', 'SkCoverage0.02', 'PCoverage0.01', 'PCoverage0.02'])
    cpt_df_stats = 0
    dataframe_results_stats = pd.DataFrame(columns=['Filename', 'Primitive Id', 'Mask', 'Nb Points', 'mIoU'])
    list_mIoU = []
    for batch_id, data in enumerate(test_dataloader, 0):
        if batch_id%100==0: print('Iteration %d / %d' % (batch_id, len(test_dataloader)))

        P = data[0].type(torch.FloatTensor).squeeze(0).to(device)
        nb_patches, num_points, _ = P.size()
        P_gt = data[2].type(torch.FloatTensor).squeeze(0).to(device)
        I_gt = data[3].type(torch.LongTensor).squeeze(0).to(device)
        T_gt = data[4].type(torch.LongTensor).squeeze(0).to(device)
        patch_indices = data[5].type(torch.LongTensor).squeeze(0).to(device)
        spfn_labels = data[6].type(torch.LongTensor).squeeze(0).to(device)
        num_global_points = spfn_labels.size(0)
        spfn_normals = data[7].type(torch.FloatTensor).squeeze(0).to(device)
        spfn_type = data[8].type(torch.FloatTensor).squeeze(0).to(device)
        glob_features = data[9].type(torch.FloatTensor).squeeze(0).to(device)
        loc_features = data[10].type(torch.FloatTensor).squeeze(0).to(device)
        P_global = data[11].type(torch.FloatTensor).squeeze(0).to(device)
        X_gt_global = data[12].type(torch.FloatTensor).squeeze(0).to(device)
        I_gt_global = data[13].type(torch.LongTensor).squeeze(0).to(device)
        plane_n_gt = data[14].type(torch.FloatTensor).to(device)
        cylinder_axis_gt = data[15].type(torch.FloatTensor).to(device)
        cone_axis_gt = data[16].type(torch.FloatTensor).to(device)
        gt_parameters = {'plane_normal': plane_n_gt,
                        'cylinder_axis': cylinder_axis_gt,
                        'cone_axis': cone_axis_gt}

        if nb_patches > 0:
            X, T, W, _, _ = spfn_module(P, glob_features=glob_features, loc_features=loc_features)
            X = X / torch.norm(X, dim=2, keepdim=True)
            W = torch.softmax(W, dim=2)

            with torch.no_grad():
                W_fusion = W
                similarity_fusion = merging_utils.similarity_soft(spfn_labels, W_fusion, patch_indices)
                labels_fusion = merging_utils.run_heuristic_solver(similarity_fusion.cpu().numpy(), nb_patches, n_max_global_instances, n_max_local_instances)
                point2primitive_fusion = torch.zeros([num_global_points, nb_patches * n_max_local_instances + n_max_global_instances]).float().to(device)
                for b in range(nb_patches):
                    point2primitive_fusion[patch_indices[b], b * n_max_local_instances:(b + 1) * n_max_local_instances] = W_fusion[b]
                point2primitive_fusion[:, (b+1)*n_max_local_instances:] = spfn_labels
                # Deleting the patch prediction for points within any patches
                flag = torch.sum(point2primitive_fusion[:,:(b+1)*n_max_local_instances], dim=1)>0
                point2primitive_fusion[flag,(b+1)*n_max_local_instances:] = 0
                W_fusion = merging_utils.get_point_final(point2primitive_fusion, torch.from_numpy(labels_fusion).to(device))

            with torch.no_grad():
                patch_indices = patch_indices.contiguous()
                X = X.contiguous()
                T = T.contiguous()
                # Normal estimation
                X_global = torch.zeros_like(X_gt_global)
                X_global = X_global.scatter_add_(0, patch_indices.view(-1).unsqueeze(1).expand(-1, 3), X.view(-1, 3))
                empty_indices = torch.all(X_global==0, axis=1)
                X_global[empty_indices] = spfn_normals[empty_indices]
                X_global = torch.nn.functional.normalize(X_global, p=2, dim=1, eps=1e-12)
                # Type estimation
                T_gt_perpoint = torch.gather(T_gt, 0, I_gt_global)
                patch_indices = patch_indices.view(-1).unsqueeze(1).expand(-1, len(conf.get_list_of_primitives()))
                num = torch.zeros_like(T_gt_perpoint).float().unsqueeze(1).expand(-1, len(conf.get_list_of_primitives()))
                num = num.scatter_add(0, patch_indices, T.view(-1, len(conf.get_list_of_primitives())))
                den = torch.zeros_like(T_gt_perpoint).float().unsqueeze(1).expand(-1, len(conf.get_list_of_primitives()))
                den = den.scatter_add(0, patch_indices, torch.ones_like(patch_indices).float())
                T_global = num / den.clamp(min=1)
                T_global[empty_indices] = spfn_type[empty_indices]
        else:
            W_fusion = torch.eye(n_max_global_instances + 1).to(spfn_labels.device)[torch.argmax(spfn_labels, dim=1) + 1]
            W_fusion = W_fusion[:, 1:]
            X_global = spfn_normals
            T_global = spfn_type
        
        W_fusion = W_fusion[:,torch.sum(W_fusion, dim=0)>1]
        if W_fusion.shape[1] < n_max_global_instances:
        	W_fusion = torch.cat((W_fusion, torch.zeros([W_fusion.shape[0], n_max_global_instances-W_fusion.shape[1]]).to(device)), dim=1)

        with torch.no_grad():
        	W_fusion = metric_implementation.hard_W_encoding(W_fusion.unsqueeze(0))
        	matching_indices_fusion, mask_fusion = metric_implementation.hungarian_matching(W_fusion, I_gt_global.unsqueeze(0))
        	mask_fusion = mask_fusion.float()
        	mIoU_fusion = metric_implementation.compute_segmentation_iou(W_fusion, I_gt_global.unsqueeze(0), matching_indices_fusion, mask_fusion)
        	mIoU_fusion_per_primitive = 1 - losses_implementation.compute_miou_loss(W_fusion, I_gt_global.unsqueeze(0), matching_indices_fusion)[0]
        	_, unique_counts_primitives_fusion = np.unique(I_gt_global.cpu().numpy(), return_counts=True)
        
        for j in range(len(unique_counts_primitives_fusion)):
        	dataframe_results_stats.loc[cpt_df_stats] = [test_dataset.hdf5_file_list[batch_id], j, mask_fusion[0, j].item(), unique_counts_primitives_fusion[j], mIoU_fusion_per_primitive[0, j].item()]
        	cpt_df_stats += 1

        with torch.no_grad():
        	mIoU, type_accuracy, normal_difference, axis_difference, mean_residual, std_residual, Sk_coverage, P_coverage, W, predicted_parameters, T = metric_implementation.compute_all_metrics(
        			P_global.unsqueeze(0), X_global.unsqueeze(0), X_gt_global.unsqueeze(0), W_fusion, I_gt_global.unsqueeze(0),
        			T_global.unsqueeze(0), T_gt.unsqueeze(0), P_gt.unsqueeze(0), gt_parameters,
        			list_epsilon=[0.01, 0.02], classes=['plane', 'sphere', 'cylinder', 'cone'])
        list_mIoU.append(mIoU.item())
        if batch_id%100==0: print('mIoU: ', np.mean(list_mIoU))
        dataframe_results.loc[batch_id] = [test_dataset.hdf5_file_list[batch_id], mIoU.item(), type_accuracy.item(),
                                                  normal_difference.item(), axis_difference.item(), mean_residual.item(),
                                                  std_residual.item(), Sk_coverage[0].item(), Sk_coverage[1].item(), P_coverage[0].item(), P_coverage[1].item()]

    dataframe_results.to_csv(os.path.join(args.output_folder, 'Results.csv'), index=False)
    dataframe_results_stats.to_csv(os.path.join(args.output_folder, 'Results_Stats.csv'), index=False)