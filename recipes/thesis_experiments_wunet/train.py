import argparse
import os
import sys
import toml
import json5
import numpy as np
import torch
from torch.utils.data import DataLoader

import torch.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

from audio_zen.utils import initialize_module


def main(rank, config, resume, only_validation):
    torch.manual_seed(config["seed"])  # for both CPU and GPU
    np.random.seed(config["seed"])

    train_dataloader = DataLoader(
        #dataset=initialize_config(config["train_dataset"]),
        dataset=initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"]),
        # batch_size=config["train_dataloader"]["batch_size"],
        # num_workers=config["train_dataloader"]["num_workers"],
        # shuffle=config["train_dataloader"]["shuffle"],
        # pin_memory=config["train_dataloader"]["pin_memory"]
        **config["train_dataset"]["dataloader"],
    )

    valid_dataloader = DataLoader(
        #dataset=initialize_config(config["validation_dataset"]),
        dataset=initialize_module(config["validation_dataset"]["path"], args=config["validation_dataset"]["args"]),
        num_workers=1,
        batch_size=1,
        pin_memory=False,
    )

    #model = initialize_config(config["model"])
    model = initialize_module(config["model"]["path"], args=config["model"]["args"])
    

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])

    #trainer_class = initialize_config(config["trainer"], pass_args=False)
    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)

    trainer = trainer_class(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train(validation_only=only_validation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wave-U-Net for Speech Enhancement")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.toml or *.json).")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    parser.add_argument("-V", "--only_validation", action="store_true", help="Only run validation. It is used for debugging validation.")
    args = parser.parse_args()

    if args.configuration.endswith(".json"):
        configuration = json5.load(open(args.configuration))
    elif args.configuration.endswith(".toml"):
        configuration = toml.load(args.configuration)
    else:
        print ("wrong config format")
        sys.exit (0)

    # configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration)) #TODO needed? doesn't it break everything?
    if configuration.get("config_path", None) == None:
        configuration["config_path"] = args.configuration

    

    mp.spawn (
        main,
        nprocs=1,
        args=(configuration, args.resume, args.only_validation),
        join=True
        )
    #main(configuration, resume=args.resume)
