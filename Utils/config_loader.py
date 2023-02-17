# Importation of packages
import yaml

class Config(object):
    def __init__(self, filename):
        self.conf = yaml.safe_load(open(filename, 'r'))

    def fetch(self, name, default_value=None):
        result = self.conf.get(name, default_value)
        assert result is not None
        return result

    def get_CUDA_visible_GPUs(self):
        return self.fetch('CUDA_visible_GPUs')

    def get_batch_size(self):
        return self.fetch('batch_size')

    def get_train_data_file(self):
        return self.fetch('train_data_file')

    def get_train_data_first_n(self):
        return self.fetch('train_first_n')

    def is_train_data_noisy(self):
        return self.fetch('train_data_noisy')

    def get_nb_train_workers(self):
        return self.fetch('train_workers')

    def get_val_data_file(self):
        return self.fetch('val_data_file')

    def get_val_data_first_n(self):
        return self.fetch('val_first_n')

    def is_val_data_noisy(self):
        return self.fetch('val_data_noisy')

    def get_nb_val_workers(self):
        return self.fetch('val_workers')

    def get_test_data_file(self):
        return self.fetch('test_data_file')

    def get_test_data_first_n(self):
        return self.fetch('test_first_n')

    def is_test_data_noisy(self):
        return self.fetch('test_data_noisy')

    def get_n_epochs(self):
        return self.fetch('n_epochs')

    def get_bn_decay_step(self):
        return self.fetch('bn_decay_step', -1)

    def get_decay_step(self):
        return self.fetch('decay_step')

    def get_decay_rate(self):
        return self.fetch('decay_rate')

    def get_init_learning_rate(self):
        return self.fetch('init_learning_rate')

    def get_val_interval(self):
        return self.fetch('val_interval', 5)

    def get_snapshot_interval(self):
        return self.fetch('snapshot_interval', 100)

    def get_visualisation_interval(self):
        return self.fetch('visualisation_interval', 50)

    def get_weights_folder(self):
        return self.fetch('weights_folder')

class SPFNConfig(Config):
    def __init__(self, filename):
        Config.__init__(self, filename)

    def get_miou_loss_multiplier(self):
        return self.fetch('miou_loss_multiplier')

    def get_normal_loss_multiplier(self):
        return self.fetch('normal_loss_multiplier')

    def get_type_loss_multiplier(self):
        return self.fetch('type_loss_multiplier')

    def get_parameter_loss_multiplier(self):
        return self.fetch('parameter_loss_multiplier')

    def get_residue_loss_multiplier(self):
        return self.fetch('residue_loss_multiplier')

    def get_total_loss_multiplier(self):
        return self.fetch('total_loss_multiplier')

    def get_list_of_primitives(self):
        return self.fetch('list_of_primitives')

    def get_n_max_global_instances(self):
        return self.fetch('n_max_global_instances')

class Global_SPFNConfig(SPFNConfig):
    def __init__(self, filename):
        SPFNConfig.__init__(self, filename)

class Local_SPFNConfig(SPFNConfig):
    def __init__(self, filename):
        SPFNConfig.__init__(self, filename)

    def get_n_max_local_instances(self):
        return self.fetch('n_max_local_instances')

class Patch_SelecConfig(Config):
    def __init__(self, filename):
        Config.__init__(self, filename)