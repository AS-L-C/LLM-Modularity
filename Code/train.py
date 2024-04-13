import argparse
import copy
import datetime
import os

import torch
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig

from numbers_dataset.dataset_utils import get_dataloaders
from utils.utils_fcns import read_config

# from numbers_dataset import get_dataloaders
# from transf_lens_wrappers.setup_tr_lens import *


def get_device(device):
    device = torch.device(
        "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    return device


def get_model(info, config):
    device = info["device"]
    hp_m = config["hp_model"]
    cfg = HookedTransformerConfig(
        n_layers=hp_m["n_layers"],
        n_heads=hp_m["n_heads"],
        d_model=hp_m["d_model"],
        d_head=hp_m["d_head"],
        d_mlp=hp_m["d_mlp"],
        act_fn="relu",
        normalization_type=None,
        d_vocab=info["inp_voc"],
        d_vocab_out=info["out_voc"],
        n_ctx=info["seq_len"],
        init_weights=True,
        device=device,
        seed=999,
    )
    model = HookedTransformer(cfg)
    return model


def get_optimizer(model, config):
    hp_tr = config["hp_training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp_tr["learning_rate"],
        weight_decay=hp_tr["weight_decay"],
        betas=hp_tr["betas"],
    )
    return optimizer


def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


def train_epoch(model, dl, device, optimizer, scheduler):
    return run_epoch(
        model,
        dl,
        device,
        mode="train",
        optimizer=optimizer,
        scheduler=scheduler,
    )


def eval_epoch(model, dl, device):
    with torch.inference_mode():
        return run_epoch(model, dl, device, mode="test")


def run_epoch(model, dl, device, mode, optimizer=None, scheduler=None):
    ep_loss = 0
    ep_acc = 0
    for ind_batch, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        ep_loss += loss.item()
        ep_acc += torch.sum(logits[:, -1, :].argmax(dim=-1) == y).item()
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

    n_batches = ind_batch + 1
    ep_loss /= n_batches
    ep_acc /= len(dl.dataset)
    return ep_loss, ep_acc


def train_model(model, dataloaders, info, optimizer, config, scheduler):
    device = info["device"]
    splits = ["train", "val"]
    n_eps = config["hp_training"]["n_epochs"]
    losses = {split: torch.zeros(n_eps) for split in splits}
    accs = {split: torch.zeros(n_eps) for split in splits}
    checkpoint_epochs = [
        epoch
        for epoch in range(n_eps)
        if ((epoch + 1) % config["log"]["save_freq"]) == 0
    ]
    model_checkpoints = [None] * len(checkpoint_epochs)
    ind_ckpt = 0
    for epoch in range(n_eps):
        # Train model
        losses["train"][epoch], accs["train"][epoch] = train_epoch(
            model,
            dataloaders["train"],
            device,
            optimizer,
            scheduler,
        )

        # Evaluate model
        losses["val"][epoch], accs["val"][epoch] = eval_epoch(
            model,
            dataloaders["val"],
            device,
        )

        # Print current performance
        print(
            ("Epoch: ") + (f"{epoch:>3}/{n_eps}"),
            end="  |  ",
        )
        for split in splits:
            print(
                (f"{split}_loss: {losses[split][epoch]:>7.3f}"),
                end="  |  ",
            )
            print(
                (f"{split}_acc: {100*accs[split][epoch]:>7.3f}%"),
                end=("  |  " if split == "train" else "\n"),
            )
        # Cache model
        if config["log"]["save_ckpts"]:
            if epoch in checkpoint_epochs:
                model_checkpoints[ind_ckpt] = copy.deepcopy(model.state_dict())
                ind_ckpt += 1

    # Test model
    losses["test"], accs["test"] = eval_epoch(
        model,
        dataloaders["test"],
        device,
    )
    print(
        f'Training ended with test_loss: {losses["test"]:.3f} and test_acc: {100*accs["test"]:.3f}%'
    )
    return losses, accs, model_checkpoints, checkpoint_epochs


def init_fld(model_path):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fld = model_path + "/" + current_time
    os.mkdir(fld)
    return fld


def save_model(
    model,
    losses,
    accs,
    model_checkpoints,
    checkpoint_epochs,
    info,
    vocab,
    fld,
):
    torch.save(
        {
            "model": model,
            "final_ckpt": model.state_dict(),
            "config": model.cfg,
            "checkpoints": model_checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
            "losses": losses,
            "accs": accs,
            "info": info,
            "vocab": vocab,
        },
        fld + "/training_results.pth",
    )
    return


def get_scheduler(optimizer, config):
    if config["hp_training"]["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["hp_training"]["learning_rate"],
            steps_per_epoch=config["hp_training"]["steps_per_epoch"],
            epochs=config["hp_training"]["n_epochs"],
        )
    else:
        scheduler = None
    return scheduler


def train_htransformer(
    config_file="../Configs/train_config.yml",
):
    # Read configuration file
    config = read_config(config_file)

    # Retrieve dataloaders
    dataloaders, vocabulary, info = get_dataloaders(
        config["control"]["create_dataset"],
        config["control"]["dataset_path"],
        config["control"]["use_equals"],
    )

    # Add info to config dict
    config["hp_training"]["steps_per_epoch"] = len(dataloaders["train"])

    # Select device
    info["device"] = get_device(config["hp_training"]["device"])

    # Initialize model
    model = get_model(info, config)

    # Initialize optimizer
    optimizer = get_optimizer(model, config)

    # Initialize scheduler
    scheduler = get_scheduler(optimizer, config)

    # Train the model
    losses, accs, model_checkpoints, checkpoint_epochs = train_model(
        model,
        dataloaders,
        info,
        optimizer,
        config,
        scheduler,
    )

    # Store trained model and supporting info
    if config["log"]["store_results"]:
        fld = init_fld(config["log"]["model_path"])
        save_model(
            model,
            losses,
            accs,
            model_checkpoints,
            checkpoint_epochs,
            info,
            vocabulary,
            fld,
        )
    return


def parse_arguments(parser):
    """Parses input arguments"""
    parser.add_argument(
        "--config_file",
        default="../Configs/train_config.yml",
        help="The model training configuration file",
    )

    args = parser.parse_args()
    return vars(args)


def main():
    parser = argparse.ArgumentParser(
        description="Trains a hooked transformer model on the numbers dataset"
    )
    args = parse_arguments(parser)
    train_htransformer(**args)
    return


if __name__ == "__main__":
    main()
