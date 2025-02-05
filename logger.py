import os
import datetime
import json
from lib.utils import print_log, CustomJSONEncoder


class Logger:
    def __init__(self, model_name, dataset):
        self.model_name = model_name
        self.dataset = dataset
        self.log_file = self._setup_log_file()

    def _setup_log_file(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_path = "logs"
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, f"{self.model_name}-{self.dataset}-{now}.log")
        return log_file

    def log_base_info(self, model, config):
        with open(self.log_file, "w") as log:
            print_log(self.dataset, log=log)
            print_log(f"--------- {self.model_name} ---------", log=log)
            print_log(json.dumps(config, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log)

    def log_model_summary(self, model, batch_size, in_steps, num_nodes, feature_size):
        from torchinfo import summary
        summary_str = summary(model, [batch_size, in_steps, num_nodes, feature_size], verbose=0)
        with open(self.log_file, "a") as log:
            print_log(summary_str, log=log)

    def log(self, message):
        with open(self.log_file, "a") as log:
            print_log(message, log=log)
