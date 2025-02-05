import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import time
import datetime
import numpy as np
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from lib.utils import print_log
import matplotlib.pyplot as plt
from models.loss import LossFusion
from tqdm import tqdm

class Trainer:
    def __init__(self, model, config, dataset):
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.DataParallel(model).to(self.device)
        self.criterion = self._get_loss_function()
        self.optimizer, self.scheduler = self._get_optimizer_and_scheduler()

    def _get_loss_function(self):
        if self.dataset in ("METRLA", "PEMSBAY"):
            return LossFusion("masked_mae")
        elif self.dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
            return LossFusion()
        else:
            raise ValueError("Unsupported dataset.")

    def _get_optimizer_and_scheduler(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get("weight_decay", 0),
            eps=self.config.get("eps", 1e-8),
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config["milestones"],
            gamma=self.config.get("lr_decay_rate", 0.1),
            verbose=False,
        )
        return optimizer, scheduler

    def prepare_data(self, data_dir="data"):
        data_path = os.path.join(data_dir, self.dataset)
        train_loader, val_loader, test_loader, scaler = get_dataloaders_from_index_data(
            data_path,
            tod=self.config.get("time_of_day"),
            dow=self.config.get("day_of_week"),
            batch_size=self.config.get("batch_size", 64),
        )
        return train_loader, val_loader, test_loader, scaler

    def _train_one_epoch(self, train_loader, scaler, clip_grad=0):
        self.model.train()
        batch_y_loss_list = []
        batch_x_loss_list = []
        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            result = self.model(x_batch, is_train=True)
            y_out = result["y"]
            x_hat = result["x_hat"]
            mask_labels = result["mask_labels"]

            y_out = scaler.inverse_transform(y_out)
            if x_hat is not None:
                x_hat = scaler.inverse_transform(x_hat)
                mask_labels = scaler.inverse_transform(mask_labels)

            loss, y_loss, x_loss = self.criterion(y_out, y_batch, x_hat, mask_labels)
            batch_y_loss_list.append(y_loss.item())
            batch_x_loss_list.append(x_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.optimizer.step()

        epoch_y_loss = np.mean(batch_y_loss_list)
        epoch_x_loss = np.mean(batch_x_loss_list)
        self.scheduler.step()

        return epoch_y_loss, epoch_x_loss

    @torch.no_grad()
    def _eval_model(self, val_loader, scaler):
        self.model.eval()
        batch_loss_list = []
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out_batch = self.model(x_batch)["y"]
            out_batch = scaler.inverse_transform(out_batch)
            loss = self.criterion(out_batch, y_batch)[0]

            batch_loss_list.append(loss.item())

        return np.mean(batch_loss_list)

    @torch.no_grad()
    def _predict(self, loader, scaler):
        self.model.eval()
        y_true = []
        y_pred = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out_batch = self.model(x_batch)["y"]
            out_batch = scaler.inverse_transform(out_batch)

            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            y_pred.append(out_batch)
            y_true.append(y_batch)

        y_true = np.vstack(y_true).squeeze()
        y_pred = np.vstack(y_pred).squeeze()

        return y_true, y_pred

    def train(self, train_loader, val_loader, scaler, verbose=1, plot=False, save=None, log=None):
        wait = 0
        min_val_loss = np.inf

        train_loss_list = []
        val_loss_list = []

        wait_for_mask = 0

        for epoch in range(self.config.get("max_epochs", 300)):
            # Train for one epoch
            train_y_loss, train_x_loss = self._train_one_epoch(train_loader, scaler, clip_grad=self.config.get("clip_grad", 0))
            train_loss_list.append(train_y_loss)

            # Validate the model
            val_loss = self._eval_model(val_loader, scaler)
            val_loss_list.append(val_loss)

            # Logging
            if (epoch + 1) % verbose == 0:
                print_log(
                    datetime.datetime.now(),
                    f"Epoch {epoch + 1}, Train Y Loss = {train_y_loss:.5f},  Train X Loss = {train_x_loss:.5f}, Val Loss = {val_loss:.5f}",
                    log=log
                )

            # Early stopping and model saving
            if val_loss < min_val_loss:
                wait = 0
                wait_for_mask = 0
                min_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                wait += 1
                if wait >= self.config.get("early_stop", 30):
                    break

                if self.config.get("adaptive_mask", True):
                    wait_for_mask += 1
                    if wait_for_mask >= self.config.get("change_mask_ratio", 10):
                        self.model.module.mask_ratio *= self.config.get("ratio_decay", 0.5)
                        self.criterion.x_loss_weight *= self.config.get("ratio_decay", 0.5)
                        if self.model.module.mask_ratio < self.config.get("ratio_threshold", 0.02):
                            self.model.module.mask_ratio = 0.
                            self.criterion.x_loss_weight = 0.
                        wait_for_mask = 0
                        print_log(f"Change mask ratio: {self.model.module.mask_ratio}", log=log)

        # Restore best model and evaluate
        self.model.load_state_dict(best_state_dict)
        train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*self._predict(train_loader, scaler))
        val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*self._predict(val_loader, scaler))

        # Log final results
        out_str = f"Early stopping at epoch: {epoch + 1}\n"
        out_str += f"Best at epoch {best_epoch + 1}:\n"
        out_str += f"Train Loss = {train_loss_list[best_epoch]:.5f}\n"
        out_str += f"Train RMSE = {train_rmse:.5f}, MAE = {train_mae:.5f}, MAPE = {train_mape:.5f}\n"
        out_str += f"Val Loss = {val_loss_list[best_epoch]:.5f}\n"
        out_str += f"Val RMSE = {val_rmse:.5f}, MAE = {val_mae:.5f}, MAPE = {val_mape:.5f}"
        print_log(out_str, log=log)

        # Plot loss curves
        if plot:
            plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
            plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
            plt.title("Epoch-Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        # Save the best model
        if save:
            torch.save(best_state_dict, save)

        return self.model

    def test(self, test_loader, scaler, log=None):
        print_log("--------- Test ---------", log=log)

        start = time.time()
        y_true, y_pred = self._predict(test_loader, scaler)
        end = time.time()

        rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
        out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            rmse_all,
            mae_all,
            mape_all,
        )
        out_steps = y_pred.shape[1]
        for i in range(out_steps):
            rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
            out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
                i + 1,
                rmse,
                mae,
                mape,
            )

        print_log(out_str, log=log, end="")
        print_log("Inference time: %.2f s" % (end - start), log=log)
