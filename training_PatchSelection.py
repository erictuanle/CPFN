# Importation of packages
import os
import sys
import torch
import argparse
import numpy as np

# Importing the Dataset file
from Dataset import dataloaders
# Importing the Network file
from PointNet2 import pn2_network
# Importing the Utils files
from Utils import config_loader, training_utils, training_visualisation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='YAML configuration file', default='Configs/config_patchSelec.yml')
    parser.add_argument('--lowres_dataset', help='Directory of the Lowres Dataset', default=os.path.expanduser('data/TraceParts_v2_LowRes/'))
    parser.add_argument('--highres_dataset', help='Directory of the Highres Dataset', default=os.path.expanduser('data/TraceParts_v2/'))
    parser.add_argument('--scale', help='Scale of the Primitives', type=float, default=0.05)
    parser.add_argument('--patchselec_weigths', help='Filename of the model weights to load', default='')
    args = parser.parse_args()

    # Loading the config file
    conf = config_loader.Patch_SelecConfig(args.config_file)

    # Selecting the visible GPUs
    visible_GPUs = conf.get_CUDA_visible_GPUs()
    device = torch.device('cuda')
    if visible_GPUs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(visible_GPUs)

    # Training Parameters
    nb_epochs = conf.get_n_epochs()
    init_learning_rate = conf.get_init_learning_rate()
    val_interval = conf.get_val_interval()
    snapshot_interval = conf.get_snapshot_interval()

    # Training Dataset
    csv_path_train = os.path.join('Dataset', conf.get_train_data_file())
    noisy_train = conf.get_train_data_first_n()
    first_n_train = conf.is_train_data_noisy()
    num_workers_train = conf.get_nb_train_workers()
    if not os.path.isdir(conf.get_weights_folder()):
        os.mkdir(conf.get_weights_folder())

    # Validation Dataset
    csv_path_val = os.path.join('Dataset', conf.get_val_data_file())
    noisy_val = conf.get_val_data_first_n()
    first_n_val = conf.is_val_data_noisy()
    num_workers_val = conf.get_nb_val_workers()

    # Launching the Network
    patchselec_weights_filename = 'patchselec_%s_module'%str(round(args.scale, 2))
    patchselec_module = pn2_network.PointNet2(dim_input=3, dim_pos=3, output_sizes=[2]).to(device)
    if os.path.isfile(os.path.join(conf.get_weights_folder(), args.patchselec_weigths)):
        dict = torch.load(os.path.join(conf.get_weights_folder(), args.patchselec_weigths))
        patchselec_module.load_state_dict(dict, strict=True)

    train_dataset = dataloaders.Dataset_PatchSelection(csv_path_train, args.lowres_dataset, args.highres_dataset, args.scale, n_points=8192, normalisation=True)
    train_datasampler = dataloaders.RandomSampler(data_source=train_dataset, seed=12345, identical_epochs=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_datasampler, batch_size=conf.get_batch_size(), num_workers=conf.get_nb_train_workers(), pin_memory=True)

    val_dataset = dataloaders.Dataset_PatchSelection(csv_path_val, args.lowres_dataset, args.highres_dataset, args.scale, n_points=8192, normalisation=True)
    val_datasampler = dataloaders.RandomSampler(data_source=val_dataset, seed=12345, identical_epochs=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=val_datasampler, batch_size=conf.get_batch_size(), num_workers=conf.get_nb_val_workers(), pin_memory=True)

    # Optimizer
    optimizer = torch.optim.Adam(patchselec_module.parameters(), lr=init_learning_rate)

    # Visualisation
    visualiser = training_visualisation.Visualiser(conf.get_visualisation_interval())

    # Initialisation
    global_step = 0
    best_loss = np.inf
    for epoch in range(nb_epochs):
        global_step, _ = training_utils.patch_selection_train_val_epoch(train_dataloader, patchselec_module, epoch, optimizer, global_step, visualiser, args, conf, device, network_mode='train')
        if (epoch % conf.get_val_interval() == 0) and (epoch > 0):
            with torch.no_grad():
                _, loss = training_utils.patch_selection_train_val_epoch(val_dataloader, patchselec_module, epoch, optimizer, global_step, visualiser, args, conf, device, network_mode='val')
            if loss < best_loss:
                torch.save(patchselec_module.state_dict(), os.path.join(conf.get_weights_folder(), patchselec_weights_filename + '.pth'))
                best_loss = loss
        if (epoch % conf.get_snapshot_interval() == 0) and (epoch > 0):
            torch.save(patchselec_module.state_dict(), os.path.join(conf.get_weights_folder(), patchselec_weights_filename + '%d.pth'%epoch))
    torch.save(patchselec_module.state_dict(), os.path.join(conf.get_weights_folder(), patchselec_weights_filename + '%d.pth' % epoch))