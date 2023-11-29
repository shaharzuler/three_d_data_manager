import os

import numpy as np

# TODO move this elsewhere

for timestep in [30, 90]:
    lumen_np_path = f"/home/shahar/data/magix/nati_lumen/manual_np_seg/CIR_{timestep}_NEW_to_CIR_{timestep}_NEW.npy"
    myo_np_path = f"/home/shahar/data/magix/shahar_myo/ts_{timestep}/manual_np_shahar_myo_seg/CIR_{timestep}_NEW_myo_shahar_to_CIR_{timestep}_NEW_myo_shahar_lps.npy"
    shell_np_path = "/home/shahar/data/magix/shell_from_lumen_xor_myo"
    lumen = np.load(lumen_np_path)
    myo = np.load(myo_np_path)
    shell = lumen^myo
    print(f"myocardioum vol {myo.sum()}")
    print(f"lumen vol {lumen.sum()}")
    print(f"shell vol {shell.sum()}")

    np.save(os.path.join(shell_np_path, f"shell_{timestep}.npy"),shell)
    
