import argparse
import json
import os
import pathlib

import torch.nn as nn
import yamlargparse


class network_wrapper(nn.Module):
    """
    A wrapper class for loading different neural network models for tasks such as 
    speech enhancement (SE), speech separation (SS), and target speaker extraction (TSE).
    It manages argument parsing, model configuration loading, and model instantiation 
    based on the task and model name.
    """

    def __init__(self):
        """
        Initializes the network wrapper without any predefined model or arguments.
        """
        super(network_wrapper, self).__init__()
        self.args = None  # Placeholder for command-line arguments
        self.config_path = None  # Path to the YAML configuration file
        self.model_name = None  # Model name to be loaded based on the task
        self.root_path = pathlib.Path(__file__).parent.resolve()

    def load_args_se(self):
        """
        Loads the arguments for the speech enhancement task using a YAML config file.
        Sets the configuration path and parses all the required parameters such as 
        input/output paths, model settings, and FFT parameters.
        """
        self.config_path = os.path.join(self.root_path, "config", "inference", f"{self.model_name}.yaml")
        parser = yamlargparse.ArgumentParser("Settings")

        # General model and inference settings
        parser.add_argument('--config', help='Config file path', action=yamlargparse.ActionConfigFile)
        parser.add_argument('--mode', type=str, default='inference', help='Modes: train or inference')
        parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints/FRCRN_SE_16K', help='Checkpoint directory')
        parser.add_argument('--input-path', dest='input_path', type=str, help='Path for noisy audio input')
        parser.add_argument('--output-dir', dest='output_dir', type=str, help='Directory for enhanced audio output')
        parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='Enable CUDA (1=True, 0=False)')
        parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='Number of GPUs to use')

        # Model-specific settings
        parser.add_argument('--network', type=str, help='Select SE models: FRCRN_SE_16K, MossFormer2_SE_48K')
        parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000, help='Sampling rate')
        parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=float, default=60.0, help='Max segment length for one-pass decoding')
        parser.add_argument('--decode-window', dest='decode_window', type=float, default=1.0, help='Decoding chunk size')

        # FFT parameters for feature extraction
        parser.add_argument('--window-len', dest='win_len', type=int, default=400, help='Window length for framing')
        parser.add_argument('--window-inc', dest='win_inc', type=int, default=100, help='Window shift for framing')
        parser.add_argument('--fft-len', dest='fft_len', type=int, default=512, help='FFT length for feature extraction')
        parser.add_argument('--num-mels', dest='num_mels', type=int, default=60, help='Number of mel-spectrogram bins')
        parser.add_argument('--window-type', dest='win_type', type=str, default='hamming', help='Window type: hamming or hanning')

        # Parse arguments from the config file
        self.args = parser.parse_args(['--config', self.config_path])

    def load_config_json(self, config_json_path):
        with open(config_json_path, 'r') as file:
            return json.load(file)

    def combine_config_and_args(self, json_config, args):
        # Convert argparse.Namespace to a dictionary
        args_dict = vars(args)

        # Remove `config` key from args_dict (it's the path to the JSON file)
        args_dict.pop("config", None)

        # Combine JSON config and args_dict, prioritizing args_dict
        combined_config = {**json_config, **{k: v for k, v in args_dict.items() if v is not None}}
        return combined_config

    def load_args_sr(self):
        """
        Loads the arguments for the speech super-resolution task using a YAML config file.
        Sets the configuration path and parses all the required parameters such as
        input/output paths, model settings, and FFT parameters.
        """
        self.config_path = os.path.join(self.root_path, "config", "inference", f"{self.model_name}.yaml")
        config_json = os.path.join(self.root_path, "config", "inference", f"{self.model_name}.json")
        parser = yamlargparse.ArgumentParser("Settings")

        # General model and inference settings
        parser.add_argument('--config', help='Config file path', action=yamlargparse.ActionConfigFile)
        parser.add_argument('--config_json', type=str, help='Path to the config.json file')
        parser.add_argument('--mode', type=str, default='inference', help='Modes: train or inference')
        parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints/FRCRN_SE_16K', help='Checkpoint directory')
        parser.add_argument('--input-path', dest='input_path', type=str, help='Path for noisy audio input')
        parser.add_argument('--output-dir', dest='output_dir', type=str, help='Directory for enhanced audio output')
        parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='Enable CUDA (1=True, 0=False)')
        parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='Number of GPUs to use')

        # Model-specific settings
        parser.add_argument('--network', type=str, help='Select SE models: FRCRN_SE_16K, MossFormer2_SE_48K')
        parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000, help='Sampling rate')
        parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=float, default=60.0, help='Max segment length for one-pass decoding')
        parser.add_argument('--decode-window', dest='decode_window', type=float, default=1.0, help='Decoding chunk size')

        # Parse arguments from the config file
        self.args = parser.parse_args(['--config', self.config_path])
        json_config = self.load_config_json(config_json)
        self.args = self.combine_config_and_args(json_config, self.args)
        self.args = argparse.Namespace(**self.args)

    def __call__(self, task, model_name):
        """
        Calls the appropriate argument-loading function based on the task type 
        (e.g., 'speech_enhancement', 'speech_separation', or 'target_speaker_extraction').
        It then loads the corresponding model based on the selected task and model name.

        Args:
        - task (str): The task type ('speech_enhancement', 'speech_separation', 'target_speaker_extraction').
        - model_name (str): The name of the model to load (e.g., 'FRCRN_SE_16K').

        Returns:
        - self.network: The instantiated neural network model.
        """

        self.model_name = model_name  # Set the model name based on user input

        # Load arguments specific to the task
        if task == 'speech_enhancement':
            self.load_args_se()  # Load arguments for speech enhancement
        elif task == 'speech_super_resolution':
            self.load_args_sr()  # load aurguments for speech super-resolution
        else:
            print(f'{task} is not supported, please select from: speech_enhancement, speech_separation, or target_speaker_extraction')
            return
        self.args.task = task
        self.args.network = self.model_name  # Set the network name to the model name

        if self.args.network == 'MossFormer2_SE_48K':
            from clearervoice48k.networks import CLS_MossFormer2_SE_48K
            self.network = CLS_MossFormer2_SE_48K(self.args)  # Load MossFormer2_SE model
        elif self.args.network == 'MossFormer2_SR_48K':
            from clearervoice48k.networks import CLS_MossFormer2_SR_48K
            self.network = CLS_MossFormer2_SR_48K(self.args)  # Load MossFormer2_SR model
        else:
            print("No network found!")
            return

        return self.network  # Return the instantiated network model
