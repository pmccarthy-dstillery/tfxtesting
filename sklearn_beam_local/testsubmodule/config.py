import os
import yaml

CONFIG_FILE = '/home/pmccarthy/tfxtesting/sklearn_beam_local/testconfig.yaml'

def load():
    with open(CONFIG_FILE, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config