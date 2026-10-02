from datetime import datetime
from os import makedirs

## point to the src files 
from pathlib import Path
import sys
PROJECT_ROOT = "/users/sd22284/b2sll_open_charm/repo/Rratio/kmatrix/src/"
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from chew_mandelstams import Chew_Mandelstamm, make_CM_plot
from loading import load_config, validate_config, build_channel_setup

if __name__ == "__main__":

    user_path = "/users/sd22284/b2sll_open_charm/repo/Rratio/"
    CONFIG_FILE = user_path + "kmatrix/config/safe_config.yml"

    config = load_config(CONFIG_FILE)
    # validate_config(config) ## - don't need it to be physcally validated with couplings etc

    setup = build_channel_setup(config)

    outfolder = "chew_mands"
    today = datetime.now().strftime("%Y_%m_%d")
    makedirs(today+'/'+outfolder, exist_ok=True)

    # --- s-axis from plotting limits in config
    plot_limits = config["plotting"]["limits"]
    s_min = plot_limits["low"] ** 2
    s_max = plot_limits["high"] ** 2
    s_vals = np.linspace(s_min, s_max, 250)
    q0_val = 1.0 / setup.z0

    # --- loop over all channels using index-aligned arrays from ChannelSetup
    for i, name in enumerate(setup.names):
        mass1 = setup.masses[i]
        mass2 = setup.masses2[i]
        oam   = setup.oam[i]
        tex   = setup.tex[i]

        cm_vals = Chew_Mandelstamm(s_vals, mass1, mass2, oam, q0_val=q0_val)

        out_pdf = f"{today}/{outfolder}/{name}.pdf"
        make_CM_plot(cm_vals, s_vals, mass1, mass2, oam, tex, out_pdf=out_pdf)
    