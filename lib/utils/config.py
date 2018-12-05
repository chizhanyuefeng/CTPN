import yaml
from easydict import EasyDict as edict

with open("config.yml", "r") as f:
    cfg = edict(yaml.load(f))