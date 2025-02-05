import yaml
import os

class Config:
    def __init__(self, model_name, dataset, model_dir="models"):
        self.model_name = model_name
        self.dataset = dataset
        self.model_dir = model_dir
        self.config = self.load_config()

    def load_config(self):
        config_path = os.path.join(self.model_dir, f"{self.model_name}.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f).get(self.dataset, {})

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value