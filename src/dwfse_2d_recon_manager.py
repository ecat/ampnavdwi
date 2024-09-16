from cfl import readcfl, writecfl
import numpy as np
import sigpy as sp
import os
import json
import typing

from recon_manager import cflExchanger
from itertools import chain

class dwfse_2d_cflExchanger(cflExchanger):

    clinic_cfl_keys = ['im_recon_vzxy', 'adc_zxy']
    debug_cfl_keys = ['im_phasenav_vzsxy', 'im_normalized_phasenav_weights_vzsxy', 'im_phasenav_weights_vzsxy', 'sens_zcxy']
    raw_cfl_keys = ['ksp_vz_sxrc', 'mask_vz_sxy', 'kspace_cal_z_sxrc', 'phase_encode_table', 'phase_encode_table_cal']

    def save_debug_cfls(self, out_dir: str, vars_to_save_in_order_keys: list, keys=debug_cfl_keys):
        self.save_cfls(out_dir, vars_to_save_in_order_keys, keys, 'debug')

    def save_clinic_cfls(self, out_dir: str, vars_to_save_in_order_keys: list, keys=clinic_cfl_keys):
        self.save_cfls(out_dir, vars_to_save_in_order_keys, keys)

    def load_clinic_cfls(self, in_dir: str):
        return self.load_cfls(in_dir, self.clinic_cfl_keys)
    
    def load_debug_cfls(self, in_dir):
        return None

    def save_raw_cfls(self, out_dir: str, sequence_opts: dict, vars_to_save_in_order_keys: list, keys=raw_cfl_keys):

        self.save_cfls(out_dir, vars_to_save_in_order_keys, keys)

        with open(os.path.join(out_dir, 'sequence_opts.json'), 'w') as json_file:
            json.dump(sequence_opts, json_file)

    def load_raw_cfls(self, in_dir: str, device=sp.cpu_device):

        to_device_helper = lambda x: sp.to_device(x, device)
        raw_data_cfls = map(to_device_helper, self.load_cfls(in_dir, self.raw_cfl_keys))
        
        with open(os.path.join(in_dir, 'sequence_opts.json'), 'r') as json_file:
            sequence_opts = json.load(json_file)

        return chain(raw_data_cfls, (sequence_opts,))



def add_dwfse_2d_recon_command_line_args_to_parser(parser):

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lambda-reg', type=float, default=-1)

    return parser

def get_dwfse_2d_recon_command_line_args():
    args = ' '
    return args