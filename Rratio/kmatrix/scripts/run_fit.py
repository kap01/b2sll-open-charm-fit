#!/usr/bin/env python3
"""
Driver script for the generalised coupled-channel K-matrix fit.

All physics lives in physics.py, all config/data loading in
loading.py, all fit setup/execution in fit.py, and all
reporting/plotting in reporting.py. This script just wires them
together in order: load config -> build channel setup + parameter spec
-> load data -> setup fit -> run fit -> report + plot.
"""

from datetime import datetime
from os import makedirs

## point to the src files 
from pathlib import Path
import sys
PROJECT_ROOT = "/users/sd22284/b2sll_open_charm/repo/Rratio/kmatrix/src/"
sys.path.insert(0, str(PROJECT_ROOT))

from loading   import load_config, validate_config, build_channel_setup, load_data
from fitting   import build_parameter_spec, KMatrixFit
from reporting import *

if __name__ == "__main__":
    user_path = "/users/sd22284/b2sll_open_charm/repo/Rratio/"
    CONFIG_FILE = user_path + "kmatrix/config/safe_config.yml"

    config = load_config(CONFIG_FILE)
    validate_config(config)

    setup = build_channel_setup(config)
    param_spec = build_parameter_spec(config, setup)

    rootfiles = [user_path+p for p in config["paths"]["data_path"]]
    outfolder = config["paths"]["save_path"]

    today = datetime.now().strftime("%Y_%m_%d")
    makedirs(today+'/'+outfolder, exist_ok=True)

    ## load the data
    x, y, eyl, eyh, source_ids = load_data(rootfiles, config['fit']['fit_range'], return_id=True)
    data = x, y, eyl, eyh

    ## setup fitter
    fit_cfg = config.get("fit", {})
    fitter = KMatrixFit(data, param_spec, setup)
    # fitter.setup()                                  ## build Minuit object, apply limits

    ## run the fit!
    best = fitter.run(n_starts=fit_cfg.get("n_starts", 6),   ## execute the multi-start MIGRAD loop
                       seed=fit_cfg.get("seed", 1),
                       jitter=fit_cfg.get("jitter"))

    ## sanity check - is the complex part of R_model = 0?
    R, _, _ = fitter.evaluate(x)
    if not np.all(R.imag == 0):
        print("[FITTER INFO] ***WARNING***")
        print("              check the R_model as it is returning some complex values")

    ## save output                       
    save_conditions(param_spec, setup,  config['fit']['fit_range'], out_txt=f"{today}/{outfolder}/fit_conditions.txt")
    report(best, param_spec, data, out_txt=f"{today}/{outfolder}/fit_result.txt")
    limit_warning(best, param_spec) ## after the above function as report is a big chunk of text in the terminal
    make_plot(best, param_spec, setup, data, config["plotting"], source_ids, out_pdf=f"{today}/{outfolder}/fit_result.pdf")
    couplings_plot(best, param_spec, setup, config['plotting'], out_pdf=f"{today}/{outfolder}/couplings.pdf")
    plot_covariance(best, param_spec, out_pdf=f"{today}/{outfolder}/correlations.pdf")