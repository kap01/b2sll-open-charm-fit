#!/usr/bin/env python3
"""
Text report and diagnostic plot for a completed KMatrixFit. Both take the
ParameterSpec and ChannelSetup as arguments rather than assuming particular
parameter or channel names, so they work unchanged for any number of
channels / resonances.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


from physics import R_model, precompute_kinematics

mytrapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

plt.rcParams.update({'font.family': 'serif',
                      'mathtext.fontset': 'stix',
                      'axes.labelsize': 16,
                      'xtick.labelsize': 12, 'ytick.labelsize': 12,
                      'legend.fontsize': 13, 'axes.titlesize': 18,
                      })

colourList = ["mediumvioletred", "#FF3EB7", "orchid", "whitesmoke", "#8AD291", "limegreen", "#36742D"]
my_cmap = LinearSegmentedColormap.from_list('my_cmap', colourList)


def limit_warning(m, param_spec):
    """Warn if any of the fit parameters have hit their limits when
    the fit is reported as minimised. Use result being less than 0.1% 
    of itself to the limit value as equality"""
    for name, lims in zip(param_spec.names, param_spec.limits):
        result = m.values[name]
        # print(f"[TEMP READOUT CHECK] {name} {result} {lims[0]} {lims[1]}")
        if abs(result-lims[0]) < 0.0001*result:
            print(f"[REPORT INFO] ***WARNING***")
            print(f"              parameter {name} has converged at lower limit {result}")
        elif abs(result-lims[1]) < 0.0001*result:
            print(f"[REPORT INFO] ***WARNING***")
            print(f"              parameter {name} has converged at upper limit {result}")
    return

def save_conditions(param_spec, setup, limits, out_txt=None):
    """Write a record of what a fit was actually run with: starting values
    and limits for every free parameter, and the channel / coupling-group
    setup. Meant to be called alongside report() so a fit's *inputs* are
    saved next to its *outputs*, and a past fit can be reproduced or
    audited later without re-reading the config it came from."""
    lines = ["=" * 66, " FIT CONDITIONS", "=" * 66, ""]

    lines.append("-- overview --")
    lines.append(f"   Number of channels        : {len(setup.names)}")
    lines.append(f"   Number of resonances      : {len(param_spec.res_mass_idx)}")
    lines.append(f"   Number of coupling groups : {len(set(setup.groups))}")
    lines.append(f"   z0 value (q0 value)       : {setup.z0} ({1/setup.z0:.4f})")
    lines.append(f"   Fit range                 : {limits['low']} - {limits['high']}")
    lines.append("")

    lines.append("-- channels --")
    for i, name in enumerate(setup.names):
        if i == setup.production_idx:
            role = "production"
        elif i in setup.open_charm_idxs:
            role = "open-charm"
        else:
            role = "closed"
        lines.append(f"   [{i}] {name:12s} mass={setup.masses[i]:.6f}  L={setup.oam[i]}  "
                      f"group={setup.groups[i]:8s} ({role})")
    lines.append("")
 
    groups = []
    for group in setup.groups:
        if group not in groups:
            groups.append(group)
    lines.append("-- coupling groups (unique) --")
    lines.append("   " + ", ".join(groups))
    lines.append("")
 
    lines.append("-- starting values & limits --")
    for name, start, lims in zip(param_spec.names, param_spec.start, param_spec.limits):
        lines.append(f"   {name:14s} start={start:+9.4f}   limits=({lims[0]:+7.3f}, {lims[1]:+7.3f})")
    lines.append("=" * 66)
 
    text = "\n".join(lines)
    if out_txt is not None:
        with open(out_txt, 'w') as outfile:
            outfile.write(text + "\n")
        print(f"[REPORT INFO] conditions saved to {out_txt}")
        return None
    else:
        print(text)

def report(m, param_spec, data, out_txt=None):
    x = data[0]
    ndf = len(x) - m.nfit
    lines = [
        "=" * 66,
        f" MIGRAD valid : {m.valid}     accurate covariance : {m.accurate}",
        f" chi2         : {m.fval:.2f}",
        f" ndf          : {ndf}   (N = {len(x)}, free params = {m.nfit})",
        f" chi2 / ndf   : {m.fval/ndf:.3f}",
        "-" * 66,
    ]
    for name in param_spec.names:
        lines.append(f"   {name:14s} = {m.values[name]:+9.4f}  +/- {m.errors[name]:.4f}")
    lines.append("=" * 66)

    text = "\n".join(lines)
    print(text)
    if out_txt is not None:
        with open(out_txt, 'w') as outfile:
            outfile.write(text + "\n")
        print(f"[REPORT INFO] fit result saved to {out_txt}")


def make_plot(m, param_spec, setup, data, config, ids, out_pdf=None):
    plot_lo, plot_hi = config["limits"]["low"], config["limits"]["high"]

    ## - prepare data
    ## load the data:
    x_all, y_all, eyl_all, eyh_all = data
    ## sort in terms of sqrt s value
    sorted_idx = np.argsort(x_all)
    ## shuffle all the data to be in sorted order
    x_all, y_all, eyl_all, eyh_all, ids = (x_all[sorted_idx], y_all[sorted_idx], eyl_all[sorted_idx], 
                                            eyh_all[sorted_idx], ids[sorted_idx])
    ## find the idx of lowest and highest elements to include
    lo_idx = np.searchsorted(x_all, plot_lo, side='left')
    hi_idx = np.searchsorted(x_all, plot_hi, side='left')
    ## cut all data to these indexes
    x, y, eyl, eyh, ids = (x_all[lo_idx:hi_idx], y_all[lo_idx:hi_idx],
                                eyl_all[lo_idx:hi_idx], eyh_all[lo_idx:hi_idx],
                                ids[lo_idx:hi_idx])
    ## for later see how many of the data sources this leaves:
    unique_ids = list(set(ids))

    ##  - prepare the parameters
    par = np.array([m.values[name] for name in param_spec.names])
    bare_masses, g, b, baseline = param_spec.unpack(par)

    ## - calculate model value smoothly across plot range
    grid = np.linspace(plot_lo - 0.05, plot_hi + 0.05, 5000)
    kinematics = precompute_kinematics(grid, setup)
    Rtot, Rbase, Rbg = R_model(bare_masses, g, b, baseline, kinematics, setup)

    ##  - calculate pull distribution
    ## calculate model value at each datapoint for pulls
    kinematics_atData = precompute_kinematics(x, setup)
    Rtot_atData, _, _ = R_model(bare_masses, g, b, baseline, kinematics_atData, setup)
    ## choose the right sigma
    pull_sigma = np.where(Rtot_atData >= y, eyh, eyl)
    pulls = (Rtot_atData - y) / pull_sigma
    ## colour code the pull point
    pull_colours = ['limegreen' if v <= 1 else 'forestgreen' if 1 < v <= 3
                        else 'goldenrod' if 3 < v <= 5 else 'crimson' for v in np.abs(pulls)]

    fig, axis = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06})
    ax = axis[0]

    ##  - plot the data
    markers = ['o', 'v', 's', 'P', '*'] ## expect less than 5 sources for now
    datacolours = ['black', 'rebeccapurple', 'mediumblue'] ## expect less than 5 sources for now
    for this_id in unique_ids: ## assume the ids are [0, 1, 2,....] as set up to be like this
        this_idx = np.where(ids==this_id)
        this_sourcetex = config["source_tex"][int(this_id)]
        this_x, this_y, this_eyl, this_eyh = x[this_idx], y[this_idx], eyl[this_idx], eyh[this_idx]
        ax.errorbar(this_x, this_y, yerr=[this_eyl, this_eyh], fmt=markers[int(this_id)], ms=4, color=datacolours[int(this_id)],
                    elinewidth=0.8, capsize=1.5, zorder=5, label="data ({source})".format(source=this_sourcetex))

    ##  - plot the model
    ax.plot(grid, Rbg, "--", color="lightseagreen", lw=1.3,
            label=r"non-resonant continuum ($|b|^2$)")
    ax.plot(grid, Rbase, ":", color="violet", lw=1.2, label="light-quark baseline")
    ax.plot(grid, Rtot, "-", color="mediumvioletred", lw=2.1,
            label=r"K-matrix fit ($\chi^2/\mathrm{ndf}=%.2f$)" % (m.fval/(len(x)-m.nfit)))

    ax.set_xlim(plot_lo, plot_hi)
    # ymin, ymax = ax.get_ylim()
    # ax.set_ylim(ymin, ymin + (ymax - ymin) * 1.12)   ## headroom for the labels below
    ax.set_ylim(1.85, 5.2) ## to match prev code

    mass_names = {param_spec.names[i] for i in param_spec.res_mass_idx}
    for name, val in zip(param_spec.names, par):
        if name in mass_names:
            ax.axvline(val, color="darkorange", ls=":", lw=0.8)
            ax.text(val, ax.get_ylim()[1] * 0.995, "$"+name+"$", ha="right", va='top', fontsize=11, color="darkorange")

    first_mthr = True
    for i in setup.open_charm_idxs:
        m_threshold = 2 * setup.masses[i]
        ax.axvline(m_threshold, color="deepskyblue", ls="-.", lw=0.8)
        if not(first_mthr) and m_threshold-2*setup.masses[i-1] < 0.05:
            ax.text(m_threshold, ax.get_ylim()[1] * 0.955, setup.tex[i], ha="left", va='top', fontsize=11, color="deepskyblue")
        else:
            ax.text(m_threshold, ax.get_ylim()[1] * 0.955, setup.tex[i], ha="right", va='top', fontsize=11, color="deepskyblue")
        first_mthr = False

        

    ax.set_ylabel(r"$R = \sigma_{\rm had}/\sigma_{\mu\mu}$")
    ax.set_title(f"Coupled-channel K-matrix fit ({setup.N_ch} channels, "
                    f"{len(param_spec.res_mass_idx)} resonances)")
    ax.legend(loc="lower right")

    # axis[1].bar(data_edges[:-1], pulls, width=0.8*data_widths, color=pull_colours,
                # align='edge', edgecolor="none", zorder=5)
    axis[1].scatter(x, pulls, color=pull_colours, marker='+', s=50, zorder=5)
    axis[1].axhline(0., color='slategray', linewidth=0.75, linestyle='--', zorder=-10)
    axis[1].axhline(2., color='slategray', linewidth=0.75, zorder=-10, linestyle='--', alpha=0.5)
    axis[1].axhline(-2., color='slategray', linewidth=0.75, zorder=-10, linestyle='--', alpha=0.5)
    axis[1].set_ylim(-5, 5)
    axis[1].set_ylabel(r"$(R^{fit} - R^{data}) / \sigma^{data}$")
    axis[1].set_xlim(plot_lo, plot_hi)
    axis[1].set_xlabel(r"$\sqrt{s}$  [GeV]", loc='right')

    fval_annotation = r"  $\chi^{2}$     : "+f"{m.fval:.2f}\n"+r"$\chi^{2}$/ndof :    "+f"{m.fval/(len(x_all) - m.nfit):.2f}"
    axis[0].annotate(fval_annotation, (0.98,0.97), xycoords='axes fraction', va='top', ha='right', fontsize=13)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.08, hspace=0.06)

    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"[REPORT INFO] plot saved to {out_pdf}")
        return None
    else:
        return fig


def couplings_plot(m, param_spec, setup, config, out_pdf=None):
    """Heatmap of fitted resonance couplings g_{r,c}: rows = resonances
    (r = 1, 2, 3, ...), columns = coupling groups / channels
    (c = ee, DD, Dst, ...). Reads resonance labels and group names
    straight off param_spec.names, so it works for any number of
    resonances or channels without changes."""
 
    groups = [] ## because some couplings as grouped across channels i.e. D0D0bar and DpDm
    [groups.append(c) for c in setup.groups if c not in groups] ## using list comprehsnion to preserve the order of the channels from config
    tex = [config['group_tex'][g] for g in groups]
    resonances = [param_spec.names[i] for i in param_spec.res_mass_idx]
    resonances_masses = [np.round(m.values[r], 3) for r in resonances]
    resonances_tex = [f"${r}$ = {m}" for r,m in zip(resonances, resonances_masses)]
    ## artifically add the background coupling
    resonances += ["b"]
    resonances_tex += ["(background)"]
    
    z = np.full((len(resonances), len(groups)), np.nan)
    for c, group in enumerate(groups):
        for r, l in enumerate(resonances):
            ## background has different coupling name syntax:
            if r != len(resonances)-1:
                label = l[2:] ## because they are of the form M_X
                name = f"g_{label}_{group}"
            else:
                name = f"b_{group}"
            if name in param_spec.names:
                z[r, c] = m.values[name]

    fig, ax = plt.subplots(figsize=(1.2*len(groups) + 2, 0.9*len(resonances) + 2))
    vmax = np.nanmax(np.abs(z))
    im = ax.imshow(z, cmap=my_cmap, vmin=-vmax, vmax=vmax, aspect="auto")
 
    ax.set_xticks(range(len(groups)))
    ax.set_yticks(range(len(resonances)))
    # ax.set_yticklabels([f"resonance {label}" for label in labels])
    ax.set_xticklabels(tex)
    ax.set_yticklabels(resonances_tex)
    ax.set_xlabel("coupling group")
    ax.set_ylabel("resonance / GeV")
    ax.set_title(r"Fitted coupling constants $g^{r}_{c}$")
 
    for r in range(len(resonances)):
        for c in range(len(groups)):
            if not np.isnan(z[r, c]):
                ax.text(c, r, f"{z[r, c]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(z[r, c]) > 0.6 * vmax else "black")
 
    fig.colorbar(im, ax=ax, label="coupling value")
    fig.tight_layout()
 
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"[REPORT INFO] coupling plot saved to {out_pdf}")
        return None
    else:
        return fig
 
def plot_covariance(m, param_spec, out_pdf=None):
    """Heatmap of the fit's parameter correlation matrix,
    as computed by Minuit.hesse(). Rows/columns are parameters in
    param_spec.names order."""

    if m.covariance is None:
        raise RuntimeError("Minuit has no covariance matrix -- call m.hesse() first "
                            "(KMatrixFit.run() already does this before returning)")
 
    names = param_spec.names
    n = len(names)
 
    matrix = np.array(m.covariance.correlation())
    vlim = 1.

    fig, ax = plt.subplots(figsize=(0.55*n + 3, 0.55*n + 3))
    im = ax.imshow(matrix, cmap=my_cmap, vmin=-vlim, vmax=vlim)
 
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
 
    fig.colorbar(im, ax=ax, label="correlation", shrink=0.8)
    fig.tight_layout()
 
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches='tight')
        print
        (f"[REPORT INFO] correlation plot saved to {out_pdf}")
        return None
    else:
        return fig
