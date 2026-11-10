from os import makedirs

## point to the src files 
from pathlib import Path
import sys
PROJECT_ROOT = "/users/sd22284/b2sll_open_charm/repo/Rratio/kmatrix/src/"
sys.path.insert(0, str(PROJECT_ROOT))

from loading   import USER_PATH, load_config, validate_config, build_channel_setup, load_data, load_fit_result, load_fit_conditions
from fitting   import build_parameter_spec, KMatrixFit
from reporting import *

if __name__ == "__main__":
    today = "2026_11_09" ## set to the day you did the fit
    CONFIG_FILE = USER_PATH + "kmatrix/config/change_background.yml"

    config = load_config(CONFIG_FILE)
    try:
        validate_config(config)
    except ValueError as e:
        print(e)
        raise SystemExit(1) from None ## will still make noisy printout if in interactive mode (python -i run_fit.py)



    rootfiles = [USER_PATH+p for p in config["paths"]["data_path"]]
    outfolder = config["paths"]["save_path"]

    conditions = load_fit_conditions(f"{today}/{outfolder}/fit_conditions.txt")
    ## overwrite each thing from fit_coniditions into the config dict, if there is a mismatch, raise an error
    try:
        config["constants"]["z0"] = conditions["constants"]["z0"]
        config["fit"]["fit_range"] = conditions["fit"]["fit_range"]
        for i, c in enumerate(conditions["channels"]):
            for k in ("name", "mass", "oam", "coupling_group"):
                config["channels"][i][k] = conditions["channels"][i][k]
    except:
        raise ValueError(f"\033[1;93m[LOAD ERROR]\033[0m Mismatch in config and conditions dict structure, check channels, z0 and fit range.\n")

    setup = build_channel_setup(config)
    param_spec = build_parameter_spec(config, setup)
    
    ## check resonances number is correct
    if conditions["resonances"] != param_spec.n_res:
        raise ValueError(f"\033[1;93m[LOAD ERROR]\033[0m Mismatch in number of resonances from results file and config.\n")



    makedirs(today+'/'+outfolder + '/redo_fits', exist_ok=True)

    ## load the data
    data = load_data(rootfiles, config['fit']['fit_range'], return_id=False)

    ## setup fitter
    fit_cfg = config.get("fit", {})
    fitter = KMatrixFit(data, param_spec, setup)


    ## load the fit! 
    result_dict = load_fit_result(f"{today}/{outfolder}/fit_result.txt")
    best = fitter.new_minuit()     ## when fitting best is the minuit object returned by running the fit
    
    ## check same parameters in results dict and param spec:
    if list(result_dict["values"].keys()) != list(best.parameters):
        raise ValueError(f"\033[1;93m[LOAD ERROR]\033[0m Mismatch in parameters from results file and config.\n")

    for name, val in result_dict["values"].items():
        best.values[name] = val
    for name, err in result_dict["errors"].items():
        if err is not None:
            best.errors[name] = err

    ## what is the biggest coupling to set the scale in a minute?
    par = [best.values[name] for name in param_spec.names]
    bare_masses, g, c, baseline = param_spec.unpack(par)
    max_coupling = max(np.nanmax(abs(g)), np.nanmax(abs(c)))

    ## save output                       
    make_plot(best, param_spec, setup, data, config, out_pdf=f"{today}/{outfolder}/redo_fits/fit_result.pdf")
    # couplings_plot(best, param_spec, setup, config['plotting'], vmax=max_coupling, out_pdf=f"{today}/{outfolder}/redo_fits/couplings.pdf")
    # nonresonant_plot(best, param_spec, setup, config['plotting'], vmax=max_coupling, out_pdf=f"{today}/{outfolder}/redo_fits/nonresonant.pdf")
    # plot_covariance(best, param_spec, out_pdf=f"{today}/{outfolder}/correlations.pdf")