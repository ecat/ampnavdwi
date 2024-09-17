import os
import shutil
import json
import copy
import random
import sys
import argparse

from recon_manager import launch_recon, EntryPoint
from dwfse_2d_recon_manager import get_dwfse_2d_recon_command_line_args

do_wet = True

original_root_dir, N_shots_in_acquisition, do_montecarlo = ('subject1_n2m0_nostab_15nex', 60, True)
#original_root_dir, N_shots_in_acquisition, do_montecarlo = ('subject1_n2m1_nostab_15nex', 60, True)
#original_root_dir, N_shots_in_acquisition, do_montecarlo = ('subject1_n2m0_nostab_8nex', 32, True)
#original_root_dir, N_shots_in_acquisition, do_montecarlo = ('subject2_n2m0_nostab_15nex', 60, True)
#original_root_dir, N_shots_in_acquisition, do_montecarlo = ('subject2_n2m1_nostab_15nex', 60, True)
#original_root_dir, N_shots_in_acquisition, do_montecarlo = ('subject2_n2m0_nostab_8nex', 32, True)

N_montecarlo = [1]
N_shots = [N_shots_in_acquisition]
N_repetitions = 10

if do_montecarlo:
    for ii in range(2, 6):
        N_shots_to_try = 2 ** ii

        if N_shots_to_try < N_shots_in_acquisition:
            N_montecarlo.append(N_repetitions)
            N_shots.append(N_shots_to_try)


original_data_dir = os.path.join('/home/sdkuser/workspace/data/', original_root_dir, 'data/')

montecarlo_root_dir = os.path.join('/home/sdkuser/workspace/src/monte_carlo/', 'src/', original_root_dir)
montecarlo_data_dir = os.path.join(montecarlo_root_dir, 'data/')
root_output_dir = os.path.join('monte_carlo/', 'out/', original_root_dir)

os.makedirs(montecarlo_data_dir, exist_ok=True)
os.makedirs(root_output_dir, exist_ok=True)

shutil.copytree(original_data_dir, montecarlo_data_dir, dirs_exist_ok=True)
shutil.copy(montecarlo_data_dir + 'sequence_opts.json', montecarlo_data_dir + 'original_sequence_opts.json')

with open(montecarlo_data_dir + 'original_sequence_opts.json') as f:
    original_json = json.load(f)

# splice different shot lists into sequence_opts
for s, N_repeats in zip(N_shots, N_montecarlo):
    
    for n in range(N_repeats):
        montecarloed_json = copy.deepcopy(original_json)

        random.shuffle(montecarloed_json['valid_shots_range'][1])
        montecarloed_json['valid_shots_range'][1] = montecarloed_json['valid_shots_range'][1][0:s]

        trial_output_dir = os.path.join(root_output_dir, 'shots' + str(s).zfill(2) + 'trial' + str(n).zfill(2))
        os.makedirs(trial_output_dir, exist_ok=True)

        with open(montecarlo_data_dir + 'sequence_opts.json', 'w') as f:
            json.dump(montecarloed_json, f)

        
        launch_recon(EntryPoint.MAIN, montecarlo_data_dir, get_dwfse_2d_recon_command_line_args(), device_id=0, 
                     do_wet=do_wet, out_dir=trial_output_dir)
        

        if do_wet:
            files_to_remove = []
            files_in_output_dir = os.listdir(trial_output_dir)

            for f in files_in_output_dir: # delete irrelevant files
                file_is_in_files_to_remove = any(t in f for t in files_to_remove)
                if file_is_in_files_to_remove:
                    os.remove(os.path.join(trial_output_dir, f))
                    
            
