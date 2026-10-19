from datetime import datetime
from os import makedirs, listdir

## point to the src files 
from pathlib import Path
import sys
PROJECT_ROOT = "/users/sd22284/b2sll_open_charm/repo/Rratio/kmatrix/src/"
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from chew_mandelstams import Chew_Mandelstamm
from physics import precompute_kinematics
from reporting import make_CM_plot, save_lookup_table
from loading import load_config, build_channel_setup, build_cm_filename, validate_cm_file, USER_PATH


if __name__ == "__main__":

    CONFIG_FILE = USER_PATH + "kmatrix/config/temp_redo_Pwaves.yml"

    config = load_config(CONFIG_FILE)

    setup = build_channel_setup(config)

    outfolder = config["paths"]["save_path"]
    today = datetime.now().strftime("%Y_%m_%d")
    makedirs(today+'/'+outfolder, exist_ok=True)

    ## Validation block: avoid overwriting existing files
    try:
        validate_cm_file(setup)
    except FileExistsError as e:
        print(e)
        sys.exit(1)

    ## s-axis from plotting limits in config:
    plot_limits = config["plotting"]["limits"]
    s_min = plot_limits["low"] ## ** 2
    s_max = plot_limits["high"] ## ** 2
    sqrts_vals = np.linspace(s_min, s_max, 500)
    s_vals = sqrts_vals**2
    i_max = np.argmax(s_vals) ## find max s-value not in penalty to print out later
    penalty_s_val = setup.penalty["s"]**2 ## expect one extra point well out of range
    s_vals = np.hstack([s_vals, penalty_s_val])

    q0_val = 1.0 / setup.z0

    # --- loop over all channels using index-aligned arrays from ChannelSetup
    for i, name in enumerate(setup.names):
        mass1 = setup.masses[i]
        mass2 = setup.masses2[i]
        oam   = setup.oam[i]
        tex   = setup.tex[i]

        cm_vals = Chew_Mandelstamm(s_vals, mass1, mass2, oam, q0_val=q0_val)

        print(f"\033[1;93m[CM INFO]\033[0m Highest mass:         \033[1;93ms = {s_vals[i_max]:.3f}\033[0m GeV^2 has \033[1;93mCM = {cm_vals[i_max]:.3f}\033[0m")
        print(f"          and penalty term has: \033[1;93ms = {s_vals[-1]:.3f}\033[0m GeV^2 has \033[1;93mCM = {cm_vals[-1]:.3f}\033[0m ")

        out_pdf = f"{today}/{outfolder}/{name}.pdf"
        make_CM_plot(cm_vals[:-1], s_vals[:-1], mass1, mass2, oam, tex, out_pdf=out_pdf) ## don't plot the extra point
        out_txt = build_cm_filename(setup, i, USER_PATH)
        save_lookup_table(cm_vals, s_vals, out_txt=out_txt)