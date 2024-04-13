import argparse

import einops
import matplotlib.pyplot as plt
import numpy as np
import TensorFox as tfx
import torch

from numbers_dataset.dataset_utils import get_dataset
from utils.utils_fcns import read_config


def parse_arguments(parser):
    """Parses input arguments"""
    parser.add_argument(
        "--config_file",
        default="../Configs/analysis_config.yml",
        help="The model analysis configuration file",
    )

    args = parser.parse_args()
    return vars(args)


def load_training_memory(config):
    training_memory = torch.load(config["control"]["model_path"])
    return training_memory


def get_final_model(tr_mem):
    model = tr_mem["model"]
    model.load_state_dict(tr_mem["final_ckpt"])
    return model


def prepare_inputs(voc, info):
    nums, operators = [
        torch.arange(info["indices"][in_type][0], info["indices"][in_type][1] + 1)
        for in_type in ["nums", "operators"]
    ]
    inputs = torch.cat((nums, operators), dim=0).view(-1, 1)
    ids = [voc[inp.item()] for inp in inputs]
    return inputs, ids


def extract(model, inputs, units, n_units, n_units_tot):
    _, cache = model.run_with_cache(inputs)
    n_layers = model.cfg.n_layers
    n_inputs = inputs.shape[0]
    pats = torch.full((n_inputs, n_units_tot), torch.nan).to(model.cfg.device)
    offs = 0
    unit_ids = [None] * n_units_tot  # np.array((n_units_tot), dtype=object)
    for layer in range(n_layers):
        for unit_type in units:
            p_temp = cache[f"blocks.{layer}.{unit_type}"].squeeze()
            if unit_type == "attn.hook_z":
                p_temp = einops.rearrange(
                    p_temp, "n_inputs n_heads d_head -> n_inputs (n_heads d_head)"
                )
                ids = []
                for h in range(model.cfg.n_heads):
                    ids.extend(
                        [
                            f"L_{{{layer}}}.{units[unit_type]}H_{{{h}}}.D_{{{d}}}"
                            for d in range(model.cfg.d_head)
                        ]
                    )
            else:
                ids = [
                    f"L_{{{layer}}}.{units[unit_type]}_{{{i}}}"
                    for i in range(n_units[unit_type])
                ]
            pats[:, offs : offs + n_units[unit_type]] = p_temp
            unit_ids[offs : offs + n_units[unit_type]] = ids
            offs += n_units[unit_type]
    return pats, unit_ids


def exctract_actv_patterns(
    tr_mem, vocabulary, info, units={"attn.hook_z": "A", "hook_mlp_out": "M"}
):
    model = tr_mem["model"]
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    d_model = model.cfg.d_model
    n_units_tot = n_layers * (n_heads * d_head + d_model)
    n_units = {"attn.hook_z": n_heads * d_head, "hook_mlp_out": d_model}
    n_inputs = info["n"]["languages"] * (info["n"]["operators"] + info["n"]["numbers"])
    n_epochs = len(tr_mem["checkpoints"])
    pats = torch.full((n_epochs, n_inputs, n_units_tot), torch.nan).to(model.cfg.device)
    inputs, inp_ids = prepare_inputs(vocabulary, info)

    for ind_ep, ckpt in enumerate(tr_mem["checkpoints"]):
        model.load_state_dict(ckpt)
        pats[ind_ep, :, :], unit_ids = extract(
            model, inputs, units, n_units, n_units_tot
        )
    return pats, inp_ids, unit_ids


def disentangle(pats, inp_ids, info):
    inp_ids = np.array(inp_ids, dtype=object)
    n_epochs, n_inputs, n_units = pats.shape
    n_symbols = info["n"]["numbers"] + info["n"]["operators"]
    n_languages = info["n"]["languages"]
    symbols = list(range(info["n"]["numbers"])) + info["operators"]
    dpats = np.full((n_epochs, n_symbols, n_languages, n_units), np.nan)
    for ind_s, s in enumerate(symbols):
        dpats[:, ind_s, :, :] = pats[:, inp_ids == s, :]
    return dpats


def compute_cpd(
    pats,
    n_facts=None,
    sweep_n_factors=False,
    max_nfacts=20,
    disentangle_language=True,
    normalize_pats=True,
    inp_ids=None,
    info=None,
):
    # Normalize activity patterns
    if normalize_pats:
        pats = (pats - np.mean(pats, axis=(0, 1), keepdims=True)) / np.std(
            pats, axis=(0, 1), keepdims=True
        )

    # Build language axis
    if disentangle_language:
        pats = disentangle(pats, inp_ids, info)

    # Perform cpd for different numbers of factors
    if sweep_n_factors:
        fact_ind = np.arange(2, max_nfacts + 1)
        errors = np.zeros((fact_ind.shape))
        for ind_f, nf in enumerate(fact_ind):
            factors, output = tfx.cpd(pats, int(nf))
            errors[ind_f] = output.rel_error
        plt.plot(fact_ind, errors, label="Error", color="blue")

        # Plot the sample points
        plt.scatter(fact_ind, errors, color="blue", s=50, zorder=3)

        # Enhance plot with grid, labels, and legend
        plt.grid(True)
        plt.title("CPD reconstruction error VS number of factors")
        plt.xlabel("Number of factors")
        plt.ylabel("Reconstruction error")
        plt.legend()

        # Show the plot
        plt.autoscale(True)
        plt.show()

    # Perform cpd for nfacts factors
    factors, output = tfx.cpd(pats, n_facts)

    return factors, output.rel_error


def save_activations(config, pats, inp_ids, unit_ids):
    torch.save(
        {
            "pats": pats,
            "inp_ids": inp_ids,
            "unit_ids": unit_ids,
        },
        config["control"]["activation_path"],
    )
    return


def load_activations(config):
    acts = torch.load(config["control"]["activation_path"])
    return acts["pats"], acts["inp_ids"], acts["unit_ids"]


def save_cpd(config, factors, error):
    torch.save(
        {
            "factors": factors,
            "error": error,
        },
        config["control"]["cpd_path"],
    )
    return


def load_cpd(config):
    cpds = torch.load(config["control"]["cpd_path"])
    return cpds["factors"], cpds["error"]


def plot_cpd(factors, epochs_ids, inp_ids, lan_ids, unit_ids, sel_nf=None):
    # Concatenate ids
    lan_ids = ["MA", "EN", "SP", "IT", "GE"]
    ids = [epochs_ids, inp_ids, lan_ids, unit_ids]
    bars = [False, True, True, True]

    # Define grid dimensions
    n_rows = factors[0].shape[1]
    n_cols = len(factors)

    # Select factors
    n_rows = sel_nf if sel_nf else n_rows

    # Define colors
    colors = plt.cm.Set3(np.linspace(0, 1, n_rows))

    # Create figure and axes using gridspec
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharey=True,
        sharex="col",
        gridspec_kw={"width_ratios": [1, 2, 1, 8]},
    )

    # Set column and row titles
    cols_titles = [f"{t} Coefficients" for t in ["Epoch", "Symbol", "Language", "Unit"]]
    rows_titles = ["Factor {}".format(row) for row in range(1, n_rows + 1)]
    xlabels = ["Epochs", "Symbol", "Language", "Unit"]
    for ax, col in zip(axs[0], cols_titles):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows_titles):
        ax.set_ylabel(row, rotation=90, size="large")

    # Unit ID labels to print
    uids = [
        "L_{0}.AH_{0}.D_{0}",
        "L_{0}.AH_{0}.D_{15}",
        "L_{0}.AH_{1}.D_{0}",
        "L_{0}.AH_{1}.D_{15}",
        "L_{0}.AH_{2}.D_{0}",
        "L_{0}.AH_{2}.D_{15}",
        "L_{0}.AH_{3}.D_{0}",
        "L_{0}.AH_{3}.D_{15}",
        "L_{0}.M_{0}",
        "L_{0}.M_{64}",
        "L_{1}.AH_{0}.D_{0}",
        "L_{1}.AH_{0}.D_{15}",
        "L_{1}.AH_{1}.D_{0}",
        "L_{1}.AH_{1}.D_{15}",
        "L_{1}.AH_{2}.D_{0}",
        "L_{1}.AH_{2}.D_{15}",
        "L_{1}.AH_{3}.D_{0}",
        "L_{1}.AH_{3}.D_{15}",
        "L_{1}.M_{0}",
        "L_{1}.M_{64}",
    ]
    uid_inds = [ind for ind, id in enumerate(unit_ids) if id in uids]

    # Plot on each axis
    for c in range(n_cols):
        cids = ids[c]
        cx = np.arange(len(cids))
        for r in range(n_rows):
            cy = factors[c][:, r]
            if bars[c]:
                axs[r, c].bar(cx, cy, color=colors[r])
            else:
                axs[r, c].plot(cx, cy, color=colors[r])

        # Set xticks
        if c in [1, 2]:
            axs[r, c].set_xticks(cx)
            axs[r, c].set_xticklabels([str(id) for id in ids[c]])
        if c == 3:
            axs[r, c].set_xlim(-1, len(cy) + 1)
            axs[r, c].set_xticks(uid_inds)
            axs[r, c].set_xticklabels([f"${s}$" for s in uids], rotation=45)
            axs[r, c].tick_params(axis="x", direction="in", pad=-40)
        axs[r, c].set_xlabel(xlabels[c])

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(
        left=0.05,
        bottom=0.1,
        right=0.95,
        top=0.95,
        wspace=0.1,
        hspace=0.1,
    )

    # Show plot
    plt.show()

    return


def plot_losses(losses, accs):
    splits = ["train", "val"]
    n_rows = 2
    n_cols = 1

    # Create figure and axes using gridspec
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharey=False,
        sharex=True,
    )
    accs = {k: 100 * v for k, v in accs.items()}
    measures = [losses, accs]
    labels = ["Loss", "Accuracy[%]"]
    for r in range(n_rows):
        for inds, split in enumerate(splits):
            y = measures[r]
            axs[r].plot(y[split], label=f"{split}")
            axs[r].set_ylabel(labels[r])
            axs[r].set_xlabel("Epochs")
        plt.legend()
    plt.tight_layout()
    plt.show()


def analyse(config):
    # Load trained models
    tr_mem = load_training_memory(config)
    dataset, vocabulary, info = get_dataset(config["control"]["dataset_path"])

    # Compute model activations
    if config["control"]["compute_activations"]:
        pats, inp_ids, unit_ids = exctract_actv_patterns(tr_mem, vocabulary, info)
        save_activations(config, pats, inp_ids, unit_ids)
    else:
        pats, inp_ids, unit_ids = load_activations(config)

    # Compute Canonical Polyadic Decomposition
    if config["control"]["compute_cpd"]:
        factors, error = compute_cpd(
            pats.cpu().numpy(),
            n_facts=info["n"]["numbers"] + info["n"]["operators"],
            disentangle_language=config["cpd"]["disentangle_lang"],
            normalize_pats=config["cpd"]["normalize_pats"],
            sweep_n_factors=False,
            max_nfacts=50,
            inp_ids=inp_ids,
            info=info,
        )
        save_cpd(config, factors, error)
    else:
        factors, error = load_cpd(config)

    # Plot results
    plot_cpd(
        factors,
        tr_mem["checkpoint_epochs"],
        list(range(info["n"]["numbers"])) + info["operators"],
        info["languages"],
        unit_ids,
        sel_nf=5,
    )

    plot_losses(losses=tr_mem["losses"], accs=tr_mem["accs"])
    print(tr_mem.keys())
    return


def main():
    parser = argparse.ArgumentParser(
        description="Perform analyses on the trained Hooked Transformer Model"
    )
    args = parse_arguments(parser)
    config = read_config(args["config_file"])
    analyse(config)
    return


if __name__ == "__main__":
    main()
