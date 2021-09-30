import argparse
import os
import sys

import toml

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from audio_zen.utils import initialize_module


def main(config, checkpoint_path, output_dir):
    inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    inferencer = inferencer_class(
        config,
        checkpoint_path,
        output_dir
    )
    inferencer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("-C", "--configuration", type=str, required=True, help="Config file.")
    parser.add_argument("-M", "--model_checkpoint_path", type=str, required=False, help="The path of the model's checkpoint.")
    parser.add_argument("-O", "--output_dir", type=str, required=False, help="The path for saving enhanced speeches.")
    parser.add_argument("-I", "--input_file", type=str, required=False, help="The path for input file indicating raw data paths.")
    args = parser.parse_args()

    configuration = toml.load(args.configuration)

    config_replacement = configuration.get("config_replacement", None)

    if config_replacement == None:
        checkpoint_path = args.model_checkpoint_path
        output_dir = args.output_dir
    else:
        checkpoint_path = config_replacement["m_flag"]
        output_dir = config_replacement["o_flag"]

    if args.input_file:
        configuration["dataset"]["args"]["dataset"] = args.input_file

    main(configuration, checkpoint_path, output_dir)
