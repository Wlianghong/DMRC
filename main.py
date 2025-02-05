import os
import torch
import datetime
from config import Config
from logger import Logger
from trainer import Trainer
from lib.utils import seed_everything, set_cpu_num
from models.DMRCMLP import DMRCMLP
from models.DMRCFormer import DMRCFormer

def main_train(config: Config):
    # Set random seed and CPU settings
    seed_everything(config.get("seed"))
    set_cpu_num(1)

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.get('gpu')))

    # Load configuration and model
    model = globals()[config.model_name](**config.get("model_args"))

    # Prepare data
    trainer = Trainer(model, config.config, config.dataset)
    train_loader, val_loader, test_loader, scaler = trainer.prepare_data()

    # Setup logger
    logger = Logger(config.model_name, config.dataset)
    logger.log_base_info(model, config.config)
    logger.log_model_summary(
        model,
        config.get("batch_size"),
        config.get("in_steps"),
        config.get("num_nodes"),
        next(iter(train_loader))[0].shape[-1],
    )
    logger.log(f"Loss: {trainer.criterion._get_name()}")

    # Save model
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = "saved_models"
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{config.model_name}-{config.dataset}-{now}.pt")
    logger.log(f"Saved Model: {save_file}")

    # Train model
    model = trainer.train(train_loader, val_loader, scaler, log=logger.log_file, save=save_file if config.get("save") else None)
    trainer.test(test_loader, scaler, log=logger.log_file)


if __name__ == "__main__":
    config = Config("DMRCMLP", "PEMS08")
    main_train(config)
