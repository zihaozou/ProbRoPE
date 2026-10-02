"""Load a training configuration with command-line overrides."""
import argparse
import json
import sys
from pathlib import Path


def parse_training_args(parser, argv=None):
    """Validate JSON options using the training parser; explicit CLI values win."""
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--config', type=Path)
    selected, remaining = selector.parse_known_args(argv)
    if selected.config is None:
        return parser.parse_args(remaining)
    with selected.config.open() as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        parser.error('Training config must be a JSON object.')
    actions = {a.dest: a for a in parser._actions}
    tokens = []
    for key, value in config.items():
        action = actions.get(key)
        if action is None or key == 'help':
            parser.error(f'Unknown training option in config: {key}')
        option = next((x for x in action.option_strings if x.startswith('--')), None)
        if option is None:
            parser.error(f'Option cannot be configured: {key}')
        if value is None:
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                parser.error(f'{key} must be a boolean.')
            tokens.append(option if value else '--no-' + option[2:])
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                parser.error(f'{key} must be a boolean.')
            if value == action.const:
                tokens.append(option)
            elif value != action.default:
                parser.error(f'{key} cannot be set to {value}.')
        elif isinstance(value, list):
            if action.nargs in ('+', '*') or isinstance(action.nargs, int):
                tokens.extend([option, *map(str, value)])
            elif key == 'model_paths':
                tokens.extend([option, json.dumps(value)])
            else:
                parser.error(f'{key} does not accept a list.')
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            tokens.extend([option, str(value)])
        else:
            parser.error(f'Unsupported value for {key}.')
    args = parser.parse_args(tokens + remaining)
    return args
