# Importation of packages
import sys
import torch
import numpy as np

from SPFN import losses_implementation

# BN Decay
def get_batch_norm_decay(global_step, batch_size, bn_decay_step, staircase=True):
    BN_INIT_DECAY = 0.5
    BN_DECAY_RATE = 0.5
    BN_DECAY_CLIP = 0.99
    p = global_step * batch_size / bn_decay_step
    if staircase:
        p = int(np.floor(p))
    bn_momentum = max(BN_INIT_DECAY * (BN_DECAY_RATE ** p), 1-BN_DECAY_CLIP)
    return bn_momentum

def update_momentum(module, bn_momentum):
    for name, module_ in module.named_modules():
        if 'bn' in name:
            module_.momentum = bn_momentum

# LR Decay
def get_learning_rate(init_learning_rate, global_step, batch_size, decay_step, decay_rate, staircase=True):
    p = global_step * batch_size / decay_step
    if staircase:
        p = int(np.floor(p))
    learning_rate = init_learning_rate * (decay_rate ** p)
    return learning_rate

# Train For One Epoch
def patch_selection_train_val_epoch(dataloader, patchselec_module, epoch, optimizer, global_step, visualiser, args, conf, device, network_mode='train'):
    assert(network_mode in ['train', 'val'])
    # Loading conf information related to current file
    batch_size = conf.get_batch_size()
    bn_decay_step = conf.get_bn_decay_step()
    decay_step = conf.get_decay_step()
    decay_rate = conf.get_decay_rate()
    init_learning_rate = conf.get_init_learning_rate()
    # Iteration over the dataset
    old_bn_momentum = get_batch_norm_decay(global_step, batch_size, bn_decay_step, staircase=True)
    old_learning_rate = get_learning_rate(init_learning_rate, global_step, batch_size, decay_step, decay_rate, staircase=True)
    total_loss = 0
    if network_mode == 'train':
        patchselec_module.train()
    elif network_mode == 'val':
        patchselec_module.eval()
    patchselec_module.train()
    for batch_id, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        # Updating the BN decay
        bn_momentum = get_batch_norm_decay(global_step, batch_size, bn_decay_step, staircase=True)
        if old_bn_momentum != bn_momentum:
            update_momentum(patchselec_module, bn_momentum)
            old_bn_momentum = bn_momentum
        # Updating the LR decay
        learning_rate = get_learning_rate(init_learning_rate, global_step, batch_size, decay_step, decay_rate, staircase=True)
        if old_learning_rate != learning_rate:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            old_learning_rate = learning_rate
        # Proper training
        points = data[0].type(torch.FloatTensor).to(device)
        batch_size_current, num_points, _ = points.size()
        output_labels = data[1].type(torch.LongTensor).to(device)
        predicted_labels = patchselec_module(points)[0]
        predicted_labels = predicted_labels.contiguous().view(batch_size_current * num_points, 2)
        output_labels = output_labels.view(batch_size_current * num_points)
        loss = torch.nn.functional.cross_entropy(predicted_labels, output_labels)
        total_loss += batch_size_current * loss.item()
        # Printing Values
        if batch_id%100==0: print('[%s][Epoch %d - Iteration %d] Loss: %f' % (network_mode, epoch, batch_id, loss.item()))
        if network_mode == 'train':
            # Backward Pass
            loss.backward()
            optimizer.step()
            global_step += 1
        # Updating the visualiser
        visualiser.log_loss(loss.item(), '%s_loss' % network_mode)
        visualiser.update()
    return global_step, total_loss

def spfn_train_val_epoch(dataloader, spfn_module, epoch, optimizer, global_step, visualiser, args, conf, device, network_mode='train'):
    assert(network_mode in ['train', 'val'])
    # Loading conf information related to current file
    batch_size = conf.get_batch_size()
    bn_decay_step = conf.get_bn_decay_step()
    decay_step = conf.get_decay_step()
    decay_rate = conf.get_decay_rate()
    init_learning_rate = conf.get_init_learning_rate()
    # Losses
    miou_loss_multiplier = conf.get_miou_loss_multiplier()
    normal_loss_multiplier = conf.get_normal_loss_multiplier()
    type_loss_multiplier = conf.get_type_loss_multiplier()
    parameter_loss_multiplier = conf.get_parameter_loss_multiplier()
    residue_loss_multiplier = conf.get_residue_loss_multiplier()
    total_loss_multiplier = conf.get_total_loss_multiplier()
    # Iteration over the dataset
    old_bn_momentum = get_batch_norm_decay(global_step, batch_size, bn_decay_step, staircase=True)
    old_learning_rate = get_learning_rate(init_learning_rate, global_step, batch_size, decay_step, decay_rate, staircase=True)
    total_loss_ = 0
    if network_mode == 'train':
        spfn_module.train()
    elif network_mode == 'val':
        spfn_module.eval()
    for batch_id, data in enumerate(dataloader, 0):
        if batch_id%100==0: print('[%s][Epoch %d - Iteration %d]' % (network_mode, epoch, batch_id))
        optimizer.zero_grad()
        # Updating the BN decay
        bn_momentum = get_batch_norm_decay(global_step, batch_size, bn_decay_step, staircase=True)
        if old_bn_momentum != bn_momentum:
            update_momentum(spfn_module, bn_momentum)
            old_bn_momentum = bn_momentum
        # Updating the LR decay
        learning_rate = get_learning_rate(init_learning_rate, global_step, batch_size, decay_step, decay_rate, staircase=True)
        if old_learning_rate != learning_rate:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            old_learning_rate = learning_rate
        # Loading the inputs
        P = data[0].type(torch.FloatTensor).to(device)
        batch_size_current, num_points, _ = P.size()
        X_gt = data[1].type(torch.FloatTensor).to(device)
        points_per_instance = data[2].type(torch.FloatTensor).to(device)
        _, nb_primitives, nb_points_primitives, _ = points_per_instance.size()
        I_gt = data[3].type(torch.LongTensor).to(device)
        T_gt = data[4].type(torch.LongTensor).to(device)
        plane_n_gt = data[5].type(torch.FloatTensor).to(device)
        cylinder_axis_gt = data[6].type(torch.FloatTensor).to(device)
        cone_axis_gt = data[7].type(torch.FloatTensor).to(device)
        gt_parameters = {'plane_normal': plane_n_gt, 'cylinder_axis': cylinder_axis_gt, 'cone_axis': cone_axis_gt}
        if args.network == 'GlobalSPFN':
            glob_features = None
            loc_features = None
        elif args.network == 'LocalSPFN':
            glob_features = data[8].type(torch.FloatTensor).to(device)
            loc_features = data[9].type(torch.FloatTensor).to(device)
        # Forward Pass
        X, T, W, _, _ = spfn_module(P, glob_features=glob_features, loc_features=loc_features)
        X = torch.nn.functional.normalize(X, p=2, dim=2, eps=1e-12)
        W = torch.softmax(W, dim=2)
        total_loss, total_normal_loss, total_type_loss, total_miou_loss, total_residue_loss, total_parameter_loss, _, _, _ = losses_implementation.compute_all_losses(
            P, W, I_gt, X, X_gt, T, T_gt, gt_parameters, points_per_instance, normal_loss_multiplier,
            type_loss_multiplier, miou_loss_multiplier, residue_loss_multiplier, parameter_loss_multiplier,
            total_loss_multiplier, False, mode_seg='mIoU', classes=conf.get_list_of_primitives())
        total_loss_ += batch_size_current * total_loss.item()
        if network_mode == 'train':
            # Backward Pass
            total_loss.backward()
            # Unecessary check for the gradient
            flag = False
            for param in spfn_module.parameters():
                if param.requires_grad and ((torch.any(torch.isinf(param.grad))) or torch.any(torch.isnan(param.grad))):
                    flag = True
                    break
            if not flag:
                optimizer.step()
            global_step += 1
        # Printing Values
        if batch_id%100==0:
            print('Loss Value: ', total_loss.item())
            print('Normal Loss', total_normal_loss.item())
            print('Type Loss', total_type_loss.item())
            print('mIoU Loss', total_miou_loss.item())
            print('Residue Loss', total_residue_loss.item())
            print('Parameter Loss', total_parameter_loss.item())
        # Updating the visualiser
        visualiser.log_loss(total_loss.item(), '%s_loss'%network_mode)
        visualiser.log_loss(total_normal_loss.item(), '%s_normal_loss'%network_mode)
        visualiser.log_loss(total_type_loss.item(), '%s_type_loss'%network_mode)
        visualiser.log_loss(total_miou_loss.item(), '%s_miou_loss'%network_mode)
        visualiser.log_loss(total_residue_loss.item(), '%s_residue_loss'%network_mode)
        visualiser.log_loss(total_parameter_loss.item(), '%s_parameter_loss'%network_mode)
        visualiser.update()
    return global_step, total_loss_