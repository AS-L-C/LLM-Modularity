import argparse
import random

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_vocabulary(n_nums=11, n_langs=5, use_equals=True):
    # Define operators and languages
    info = dict()
    info["languages"] = ["math", "eng", "spa", "ita", "ger"]
    info["operators"] = ["+", "*"]
    info["oper_map"] = {"+": torch.add, "*": torch.multiply}
    info["n"] = {
        "languages": len(info["languages"]),
        "operators": len(info["operators"]),
        "numbers": n_nums,
    }
    info["use_equals"] = use_equals
    vocabulary = {}

    # Numbers
    for lan in range(n_langs):
        offs = (lan) * (n_nums)
        vocabulary.update({offs + i: i for i in range(n_nums)})
    offs += n_nums

    # Operators in language
    for lan in range(n_langs):
        vocabulary.update(
            {offs + i: info["operators"][i] for i in range(info["n"]["operators"])}
        )
        offs += info["n"]["operators"]

    # Indices
    info["indices"] = {}
    info["indices"]["nums"] = [0, (n_nums) * (n_langs) - 1]
    info["indices"]["operators"] = [
        (n_nums) * (n_langs),
        (n_nums) * (n_langs) + (info["n"]["operators"]) * (n_langs) - 1,
    ]

    info["seq_len"] = 3
    info["inp_voc"] = (info["n"]["numbers"] + info["n"]["operators"]) * info["n"][
        "languages"
    ]
    info["out_voc"] = (info["n"]["numbers"] - 1) ** 2 + 1

    if use_equals:
        vocabulary.update({offs: "="})
        info["indices"]["equals"] = info["indices"]["operators"][-1] + 1
        info["seq_len"] += 1
        info["inp_voc"] += 1

    print(vocabulary)
    print(info["indices"])
    return vocabulary, info


def translate(x, vocabulary):
    return [vocabulary[xi.item()] for xi in x]


def perform_operation(x, vocabulary, info):
    x = translate(x, vocabulary)
    return info["oper_map"][x[1]](x[0], x[2])


def compute_result(x, vocabulary, info):
    y = torch.zeros_like(x[:, 0])
    for r, xr in enumerate(x):
        y[r] = perform_operation(xr, vocabulary, info)
    return y


def create_tensor_dataset(vocabulary, info):
    info["n"]["examples"] = (
        (info["n"]["languages"] ** 3)
        * (info["n"]["numbers"] ** 2)
        * (info["n"]["operators"])
    )
    info["n"]["repeats"] = info["n"]["examples"] // (
        info["n"]["numbers"] * info["n"]["languages"]
    )
    x = torch.full((info["n"]["examples"], info["seq_len"]), torch.nan).to(torch.int64)

    # First number
    x[:, 0] = torch.arange(info["n"]["numbers"] * info["n"]["languages"]).repeat(
        info["n"]["repeats"]
    )

    # Operator
    x1_temp = torch.arange(
        info["indices"]["operators"][0], info["indices"]["operators"][1] + 1
    ).repeat_interleave(info["n"]["languages"] * info["n"]["numbers"])

    x[:, 1] = x1_temp.repeat(info["n"]["numbers"] * info["n"]["languages"])

    # Second number
    x[:, 2] = torch.arange(
        info["n"]["numbers"] * info["n"]["languages"]
    ).repeat_interleave(info["n"]["repeats"])

    # Equal sign
    if info["use_equals"]:
        x[:, 3] = info["indices"]["equals"]

    assert x.isnan().any().item() is False, "There are nan values in the x tensor"
    y = compute_result(x, vocabulary, info).to(torch.int64)
    plt.scatter(torch.arange(info["n"]["examples"]), y)
    plt.xlabel("Example")
    plt.ylabel("Result")
    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset, info


def split_dataset(dataset, info, f_train=0.6, f_val=0.2, f_test=0.2):
    f = [f_train, f_val, f_test]
    assert sum(f) == 1, "Fractions don't add up to 1"
    datasets = {split: None for split in ["train", "val", "test"]}
    datasets["train"], datasets["val"], datasets["test"] = (
        torch.utils.data.random_split(dataset, f)
    )
    for split in datasets:
        info["n"][split] = len(datasets[split])
    assert (
        sum([info["n"][split] for split in datasets]) == info["n"]["examples"]
    ), "Splits examples don't add up to n_examples"
    return datasets, info


def create_dataloaders(
    dataset,
    g,
    seed_worker,
    shuffle=True,
    batch_size=1024,
):
    dataloaders = {}
    for split in dataset:
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
        )
    return dataloaders


def save_dataset(dataset, vocabulary, info, dataset_path):
    torch.save(
        {"dataset": dataset, "vocabulary": vocabulary, "info": info}, dataset_path
    )
    return


def get_dataset(dataset_path):
    torch_dict = torch.load(dataset_path)
    dataset, vocabulary, info = [
        torch_dict[k] for k in ["dataset", "vocabulary", "info"]
    ]
    return dataset, vocabulary, info


def get_dataloaders(
    create_dataset=True,
    dataset_path="../Data/dataset.pt",
    use_equals=True,
):
    g, worker_seed = set_seeds()
    if create_dataset:
        # device = get_device(device)
        vocabulary, info = create_vocabulary(use_equals=use_equals)
        dataset, info = create_tensor_dataset(vocabulary, info)
        dataset, info = split_dataset(dataset, info)
        save_dataset(dataset, vocabulary, info, dataset_path)
    else:
        dataset, vocabulary, info = get_dataset(dataset_path)

    dataloaders = create_dataloaders(dataset, g, worker_seed)
    return dataloaders, vocabulary, info


def set_seeds(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    worker_seed = torch.initial_seed() % 2**32
    g = torch.Generator()
    g.manual_seed(seed)
    return g, worker_seed


def parse_arguments(parser):
    """Parses input arguments"""
    parser.add_argument(
        "--create_dataset",
        default="False",
        choices=["True", "False"],
        help="Set to True to create the dataset, or False to load it",
    )
    parser.add_argument(
        "--dataset_path",
        default="../Data/dataset.pt",
        help="The path to store tha dataset at, or to load it from",
    )
    parser.add_argument(
        "--use_equals",
        default="True",
        choices=["True", "False"],
        help="Set to True to add an equals sign at the end of the input sequence",
    )
    args = parser.parse_args()
    args = vars(args)
    for k, v in args:
        if v in ["True", "False"]:
            args[k] = True if v == "True" else False
    return args


def main():
    parser = argparse.ArgumentParser(description="Creates the numbers dataset")
    args = parse_arguments(parser)
    dataloaders, vocabulary, info = get_dataloaders(**args)


if __name__ == "__main__":
    main()
