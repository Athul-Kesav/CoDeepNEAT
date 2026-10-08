from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Tuple

# Keep the virtualenv version ahead of the system copy on installations where
# pydantic_core imports typing_extensions during module initialization.
venv_site_packages = os.path.join(
    sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'
)
if os.path.isdir(venv_site_packages):
    sys.path.insert(0, venv_site_packages)
    import typing_extensions
    sys.path.remove(venv_site_packages)

import torch
from torch import nn

dir_path = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(dir_path))
sys.path.extend([
    project_root,
    os.path.join(project_root, 'src'),
    os.path.join(project_root, 'runs'),
    os.path.join(project_root, 'configuration'),
])

import src.main.singleton as singleton
from configuration import config
from configuration.config_loader import load_simple_config
from runs.runs_manager import get_fully_train_folder_path
from src.analysis.run import get_run
from src.phenotype.neural_network.evaluator.data_loader import get_data_shape
from src.phenotype.neural_network.feature_multiplication import get_model_of_target_size
from src.phenotype.neural_network.neural_network import Network


def parse_model_name(model_name: str) -> Tuple[int, int]:
    match = re.fullmatch(r'bp-(\d+)_fm-(\d+)(?:-best-\d+)?\.model', model_name)
    if match is None:
        raise ValueError(
            'Model name must match bp-<blueprint>_fm-<multiplier>[-best-<rank>].model'
        )
    return int(match.group(1)), int(match.group(2))


def find_blueprint(run, blueprint_id: int):
    for generation in run.generations:
        for blueprint in generation.blueprint_population:
            if blueprint.id == blueprint_id:
                return blueprint, generation.generation_number
    raise ValueError(f'Blueprint {blueprint_id} was not found in the saved generations')


def reconstruct_model(run_name: str, model_name: str) -> Network:
    blueprint_id, feature_multiplier = parse_model_name(model_name)
    run = get_run(run_name)
    blueprint, generation_number = find_blueprint(run, blueprint_id)
    singleton.instance = run.generations[generation_number]
    input_shape = get_data_shape()

    model = Network(
        blueprint,
        input_shape,
        sample_map=blueprint.best_module_sample_map,
        allow_module_map_ignores=False,
        feature_multiplier=1,
        target_feature_multiplier=feature_multiplier,
    ).to(config.get_device())

    if feature_multiplier != 1:
        model = get_model_of_target_size(
            blueprint,
            model.sample_map,
            model.size(),
            input_shape,
            target_size=model.size() * feature_multiplier,
        ).to(config.get_device())
        model.target_feature_multiplier = feature_multiplier

    checkpoint_path = os.path.join(get_fully_train_folder_path(run_name), model_name)
    checkpoint = torch.load(checkpoint_path, map_location=config.get_device())
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def sparsity_manifest(model: nn.Module) -> Dict[str, object]:
    layers = []
    eligible_parameters = 0
    total_parameters = 0

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        weight = module.weight.detach()
        parameter_count = weight.numel()
        total_parameters += parameter_count
        eligible = parameter_count % 4 == 0
        if eligible:
            eligible_parameters += parameter_count
        layers.append({
            'name': name,
            'type': type(module).__name__,
            'shape': list(weight.shape),
            'parameters': parameter_count,
            'eligible_for_flattened_2:4': eligible,
        })

    return {
        'total_conv_linear_parameters': total_parameters,
        'eligible_conv_linear_parameters': eligible_parameters,
        'eligible_fraction': eligible_parameters / total_parameters if total_parameters else 0,
        'layers': layers,
    }


def main():
    parser = argparse.ArgumentParser(description='Reconstruct a trained CoDeepNEAT model for sparsity analysis')
    parser.add_argument('-c', '--config', required=True, help='Run config, e.g. cifar100')
    parser.add_argument('-m', '--model', required=True, help='Checkpoint filename in fully_trained_models')
    parser.add_argument('--manifest', help='Optional JSON path for the layer eligibility report')
    args = parser.parse_args()

    # Fine-tuning config loading disables W&B initialization while preserving
    # the saved run configuration needed to rebuild the phenotype.
    load_simple_config(args.config, lambda: None, lambda: None, use_wandb_override=False)
    model_path = os.path.join(get_fully_train_folder_path(config.run_name), args.model)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    model = reconstruct_model(config.run_name, args.model)
    manifest = sparsity_manifest(model)
    manifest.update({
        'run_name': config.run_name,
        'model': args.model,
        'device': str(config.get_device()),
        'state_dict_verified': True,
    })

    output = json.dumps(manifest, indent=2)
    print(output)
    if args.manifest:
        with open(args.manifest, 'w') as manifest_file:
            manifest_file.write(output + '\n')


if __name__ == '__main__':
    main()
