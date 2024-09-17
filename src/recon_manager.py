import os
from cfl import writecfl, readcfl
import sys
import shutil
import warnings
import time
import random
import string
from abc import ABC, abstractmethod
from enum import Enum

class cflExchanger(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def save_debug_cfls(self):
        pass

    @abstractmethod
    def save_raw_cfls(self):
        pass

    @abstractmethod
    def load_raw_cfls(self):
        pass

    @abstractmethod
    def save_clinic_cfls(self):
        pass

    @abstractmethod
    def load_clinic_cfls(self):
        pass

    def save_cfls(self, out_dir: str, vars_to_save_in_order_of_keys: list, keys: list, tag=None):
        add_tag = lambda x: x if tag is None else x + tag
        os.makedirs(out_dir, exist_ok=True)

        assert(len(vars_to_save_in_order_of_keys) == len(keys))
        get_out_filename = lambda x: os.path.join(out_dir, add_tag(x))    

        for k, v in zip(keys, vars_to_save_in_order_of_keys):
            writecfl(get_out_filename(k), v)    

    def load_cfls(self, in_dir: str, keys_to_read: list):
        return (readcfl(os.path.join(in_dir, key), remove_single_dimensions_at_end=False) for key in keys_to_read)

# class to manage paths for storing temporary files
class PathManager:

    def __init__(self, archive_dir: str, clinic_mode: bool):

        self.data_base_dir = '/home/sdkuser/data/'
        self.clinic_mode = clinic_mode
        print('pathManager archive_dir ' + archive_dir)
        archive_dir_relative_path = archive_dir.strip(self.data_base_dir)
        
        data_folder = id_generator(archive_dir) if self.clinic_mode else archive_dir_relative_path

        root_out_dir = '/tmp/' if clinic_mode else os.getcwd() 

        cfl_manager_dir = os.path.join(root_out_dir, data_folder)

        self.raw_cfl_dir = PathManager.get_raw_cfl_path(cfl_manager_dir)

        print('pathManager raw_cfl_dir: ' + self.raw_cfl_dir)
        os.makedirs(self.raw_cfl_dir, exist_ok=True)

        self.recon_cfl_dir = PathManager.get_recon_cfl_path(cfl_manager_dir)

        print('pathManager recon_cfl_dir: ' + self.recon_cfl_dir)
        os.makedirs(self.recon_cfl_dir, exist_ok=True)

    def get_raw_cfl_path(root_cfl_dir: str):
        return os.path.join(root_cfl_dir, 'data/')
    
    def get_recon_cfl_path(root_cfl_dir: str):
        return os.path.join(root_cfl_dir, 'recon/')

    def delete_tmp_dir(self, dir: str):
        if self.clinic_mode:
            print('Deleting directory ' + dir)
            remove(dir)
        else:
            warnings.warn('Did not delete' + dir + ' since not in clinic mode')

    def delete_raw_cfl_dir(self):
        self.delete_tmp_dir(self.raw_cfl_dir)            

    def delete_recon_cfl_dir(self):
        self.delete_tmp_dir(self.recon_cfl_dir)

    def get_raw_data_cfl_dir(self):
        return self.raw_cfl_dir
    
    def get_recon_cfl_dir(self):
        return self.recon_cfl_dir
    
class EntryPoint(Enum):
    MAIN = 'main.py'

def id_generator(seed, size=6, chars=string.ascii_uppercase + string.digits):
    random.seed(seed)
    return ''.join(random.choice(chars) for _ in range(size))        
    
def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        warnings.warn('file {} is not a file or dir.'.format(path))      

def launch_recon(entry_point, data_dir, command_line_args, device_id, do_wet, extra_args=None, out_dir=None):

    data_dir_arg = ' --data-dir ' + data_dir

    cmd = 'python3 ' + entry_point.value + data_dir_arg + command_line_args + ' --device ' + str(device_id) 

    if out_dir is not None:
        cmd = cmd + ' --out-dir ' + out_dir

    if extra_args is not None:
        cmd = cmd + ' ' + extra_args

    print(cmd)
    if do_wet:
        os.system(cmd)

    time.sleep(1) 