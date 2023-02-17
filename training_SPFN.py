# Importation of packages
import os
import sys
import torch
import argparse
import numpy as np

# Importing the Dataset files
from Dataset import dataloaders
# Importing the Network files
from SPFN import fitter_factory, losses_implementation
from PointNet2 import pn2_network
# Importing Utils files
from Utils import config_loader, training_utils, training_visualisation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='YAML configuration file', type=str, default='Configs/config_globalSPFN.yml')
    parser.add_argument('--lowres_dataset', help='Directory of the Input Dataset', type=str, default=os.path.expanduser('data/TraceParts_v2_lowres/'))
    parser.add_argument('--network', help='Network to train: GlobalSPFN, LocalSPFN', type=str, default='GlobalSPFN')
    parser.add_argument('--path_patches', help='Path to Sampled Patches h5 files', type=str, default=os.path.expanduser('data/TraceParts_v2_patches'))
    parser.add_argument('--scale', help='Scale to select the smallest primitive', type=float, default=0.05)
    parser.add_argument('--spfn_weigths', help='Filename of the model weights to load', type=str, default='') 
    args = parser.parse_args()

    # Loading the config file
    assert (args.network in ['GlobalSPFN', 'LocalSPFN'])
    if args.network == 'GlobalSPFN':
        conf = config_loader.Global_SPFNConfig(args.config_file)
    elif args.network == 'LocalSPFN':
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
    if args.network == 'LocalSPFN':
        n_max_local_instances = conf.get_n_max_local_instances()

    # Training Parameters
    nb_epochs = conf.get_n_epochs()
    init_learning_rate = conf.get_init_learning_rate()
    val_interval = conf.get_val_interval()
    snapshot_interval = conf.get_snapshot_interval()

    # Training Dataset
    csv_path_train = os.path.join('Dataset', conf.get_train_data_file())
    noisy_train = conf.is_train_data_noisy()
    first_n_train = conf.get_train_data_first_n()
    num_workers_train = conf.get_nb_train_workers()
    path_patches = os.path.join(args.path_patches, str(round(args.scale, 2)))

    # Validation Dataset
    csv_path_val = os.path.join('Dataset', conf.get_val_data_file())
    noisy_val = conf.is_val_data_noisy()
    first_n_val = conf.get_val_data_first_n()
    num_workers_val = conf.get_nb_val_workers()

    # Launching the Network
    if args.network == 'GlobalSPFN':
        spfn_weights_filename = 'globalspfn_module'
        spfn_module = pn2_network.PointNet2(dim_input=3, dim_pos=3, output_sizes=[3, n_registered_primitives, n_max_global_instances]).to(device)
    elif args.network == 'LocalSPFN':
        spfn_weights_filename = 'localspfn_%s_module'%str(round(args.scale, 2))
        spfn_module = pn2_network.PointNet2(dim_input=3, dim_pos=3, output_sizes=[3, n_registered_primitives, n_max_local_instances]).to(device)
    if os.path.isfile(os.path.join(conf.get_weights_folder(), args.spfn_weigths)):
        dict = torch.load(os.path.join(conf.get_weights_folder(), args.spfn_weigths))
        spfn_module.load_state_dict(dict, strict=True)

    # Loading the dataset
    if args.network == 'GlobalSPFN':
        train_dataset = dataloaders.Dataset_GlobalSPFN(n_max_global_instances, csv_path_train, args.lowres_dataset, None, None,  noisy_train, n_points=8192, first_n=first_n_train, fixed_order=False)
        train_datasampler = dataloaders.RandomSampler(data_source=train_dataset, seed=12345, identical_epochs=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_datasampler, batch_size=conf.get_batch_size(), num_workers=num_workers_train, pin_memory=True)
        
        val_dataset = dataloaders.Dataset_GlobalSPFN(n_max_global_instances, csv_path_val, args.lowres_dataset, None, None, noisy_val, n_points=8192, first_n=first_n_val, fixed_order=False)
        val_datasampler = dataloaders.RandomSampler(data_source=val_dataset, seed=12345, identical_epochs=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=val_datasampler, batch_size=conf.get_batch_size(), num_workers=conf.get_nb_val_workers(), pin_memory=True)

    elif args.network == 'LocalSPFN':
        train_dataset = dataloaders.Dataset_TrainLocalSPFN(n_max_local_instances, csv_path_train, path_patches, noisy_train, first_n=first_n_train, fixed_order=False, lean=True)
        train_datasampler = dataloaders.RandomSampler(data_source=train_dataset, seed=12345, identical_epochs=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_datasampler, batch_size=conf.get_batch_size(), num_workers=num_workers_train, pin_memory=True)
        
        val_dataset = dataloaders.Dataset_TrainLocalSPFN(n_max_local_instances, csv_path_val, path_patches, noisy_val, first_n=first_n_val, fixed_order=False, lean=True)
        val_datasampler = dataloaders.RandomSampler(data_source=val_dataset, seed=12345, identical_epochs=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=val_datasampler, batch_size=conf.get_batch_size(), num_workers=conf.get_nb_val_workers(), pin_memory=True)

    # Optimizer
    optimizer = torch.optim.Adam(spfn_module.parameters(), lr=init_learning_rate)

    # Visualisation
    visualiser = training_visualisation.Visualiser(conf.get_visualisation_interval())

    # Initialisation
    global_step = 0
    old_learning_rate = init_learning_rate
    best_loss = np.inf
    for epoch in range(nb_epochs):
        global_step, _ = training_utils.spfn_train_val_epoch(train_dataloader, spfn_module, epoch, optimizer, global_step, visualiser, args, conf, device, network_mode='train')
        if (epoch % conf.get_val_interval() == 0) and (epoch > 0):
            with torch.no_grad():
                _, loss = training_utils.spfn_train_val_epoch(val_dataloader, spfn_module, epoch, optimizer, global_step, visualiser, args, conf, device, network_mode='val')
            if loss < best_loss:
                torch.save(spfn_module.state_dict(), os.path.join(conf.get_weights_folder(), spfn_weights_filename + '.pth'))
                best_loss = loss
        if (epoch % conf.get_snapshot_interval() == 0) and (epoch > 0):
            torch.save(spfn_module.state_dict(), os.path.join(conf.get_weights_folder(), spfn_weights_filename + '%d.pth' % epoch))
    torch.save(spfn_module.state_dict(), os.path.join(conf.get_weights_folder(), spfn_weights_filename + '%d.pth' % epoch))