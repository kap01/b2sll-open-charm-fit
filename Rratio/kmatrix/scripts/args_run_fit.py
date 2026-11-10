#!/usr/bin/env python3
"""
Driver script for the generalised coupled-channel K-matrix fit.

this version is for batch job submission as the config file is set
using an argument in the commandline and it doesn't make a today
folder, assuming you are submitting the job from a today folder
"""

import argparse
# from datetime import datetime
from os import makedirs

## point to the src files 
from pathlib import Path
import sys
PROJECT_ROOT = "/users/sd22284/b2sll_open_charm/repo/Rratio/kmatrix/src/"
sys.path.insert(0, str(PROJECT_ROOT))

from loading   import USER_PATH, load_config, validate_config, build_channel_setup, load_data
from fitting   import build_parameter_spec, KMatrixFit
from reporting import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the generalised coupled-channel K-matrix fit.")
    parser.add_argument("--config", help="Path to the config YAML file")
    args = parser.parse_args()

    CONFIG_FILE = USER_PATH+ args.config

    config = load_config(CONFIG_FILE)
    try:
        validate_config(config)
    except ValueError as e:
        print(e)
        raise SystemExit(1) from None ## will still make noisy printout if in interactive mode (python -i run_fit.py)

    setup = build_channel_setup(config)
    param_spec = build_parameter_spec(config, setup)

    rootfiles = [USER_PATH+p for p in config["paths"]["data_path"]]
    outfolder = config["paths"]["save_path"]

    # today = datetime.now().strftime("%Y_%m_%d")
    makedirs(outfolder, exist_ok=True)

    ## load the data
    data = load_data(rootfiles, config['fit']['fit_range'], return_id=False)

    ## setup fitter
    fit_cfg = config.get("fit", {})
    fitter = KMatrixFit(data, param_spec, setup)

    ## run the fit!
    best = fitter.run(n_starts=fit_cfg.get("n_starts", 6),   ## execute the multi-start MIGRAD loop
                      seed=fit_cfg.get("seed", 1),
                      jitter=fit_cfg.get("jitter"))

    ## sanity check - is the complex part of R_model = 0?
    R = fitter.evaluate(data[0])
    if not np.all(R.imag == 0):
        print("\033[1;93m[FITTER WARNING]\033[0m ***WARNING***")
        print("              check the R_model as it is returning some complex values")

    ## what is the biggest coupling to set the scale in a minute?
    par = [best.values[name] for name in param_spec.names]
    bare_masses, g, c, baseline = param_spec.unpack(par)
    max_coupling = max(np.nanmax(abs(g)), np.nanmax(abs(c)))

    ## save output                       
    save_conditions(param_spec, setup,  config['fit']['fit_range'], out_txt=f"{outfolder}/fit_conditions.txt")
    report(best, fitter.status, param_spec, data, out_txt=f"{outfolder}/fit_result.txt")
    limit_warning(best, param_spec) ## after the above function as report is a big chunk of text in the terminal
    make_plot(best, param_spec, setup, data, config, out_pdf=f"{outfolder}/fit_result.pdf")
    couplings_plot(best, param_spec, setup, config['plotting'], vmax=max_coupling, out_pdf=f"{outfolder}/couplings.pdf")
    nonresonant_plot(best, param_spec, setup, config['plotting'], vmax=max_coupling, out_pdf=f"{outfolder}/nonresonant.pdf")
    plot_covariance(best, param_spec, out_pdf=f"{outfolder}/correlations.pdf")
