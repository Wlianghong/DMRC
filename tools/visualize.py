import sys
import os
import torch
sys.path.append("..")

from config import Config
from trainer import Trainer
from lib.utils import seed_everything, set_cpu_num
from models.DMRCMLP import DMRCMLP
from visualize_util import visualize

# load config
config = Config("DMRCMLP", "PEMS08", model_dir="../models")

# Set random seed and CPU settings
seed_everything(config.get("seed"))
set_cpu_num(1)

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.get('gpu')))

# Load configuration and model
model = globals()[config.model_name](**config.get("model_args"))
model.mask_ratio = 0.

# # print structure
for name, module in model.named_modules():
    print(f"Layer Name: {name}, Layer Type: {module.__class__.__name__}")
# print(model)

# Prepare data
trainer = Trainer(model, config.config, config.dataset)
train_loader, val_loader, test_loader, scaler = trainer.prepare_data("../data")

# load model
model = trainer.model
model.load_state_dict(torch.load(os.path.join("../saved_models", "DMRCMLP-PEMS08-2025-01-16-15-23-20.pt")))

# test
trainer.test(test_loader, scaler, log=None)

# ############################################ register hook ############################################
v_type = "tsne"
v_dim = 2

def y_predictor_hook(module, input, output):
    ####################### input #######################
    out = output.cpu() # x (batch_size, in_steps, num_node, feature)
    spatial_data = out[1, :, :, :]
    spatial_data = spatial_data.reshape(-1, spatial_data.shape[-1])
    visualize(spatial_data, v_type, v_dim, "(c) Last Pred. MLP Layer Output")

    # temporal_data = out[0, :, :10, :]
    # temporal_data = temporal_data.reshape(-1, temporal_data.shape[-1])
    # visualize(temporal_data, v_type, v_dim)

y_hook = model.module.y_predictor.mlp_layers[-1].register_forward_hook(y_predictor_hook)

def x_predictor_hook(module, input, output):
    # ####################### input #######################
    # x = input[0].cpu() # x (batch_size, in_steps, num_node, feature)
    # spatial_data = x[0, 0, :, :]
    # spatial_data = spatial_data.reshape(-1, spatial_data.shape[-1])
    # visualize(spatial_data, v_type, v_dim)

    # temporal_data = x[0, :, :10, :]
    # temporal_data = temporal_data.reshape(-1, temporal_data.shape[-1])
    # visualize(temporal_data, v_type, v_dim)

    ####################### output #######################
    out = output.cpu() # x (batch_size, in_steps, num_node, feature)
    spatial_data = out[1, :, :, :]
    spatial_data = spatial_data.reshape(-1, spatial_data.shape[-1])
    visualize(spatial_data, v_type, v_dim, "(d) Last Rec. MLP Layer Output")

    # temporal_data = out[0, :, :10, :]
    # temporal_data = temporal_data.reshape(-1, temporal_data.shape[-1])
    # visualize(temporal_data, v_type, v_dim)

x_hook = model.module.x_predictor.mlp_layers[-1].register_forward_hook(x_predictor_hook)

def sharing_out_hook(module, input, output):
    ####################### output #######################
    out = output.cpu() # x (batch_size, in_steps, num_node, feature)
    spatial_data = out[1, :, :, :]
    spatial_data = spatial_data.reshape(-1, spatial_data.shape[-1])
    visualize(spatial_data, v_type, v_dim, "(b) Last Shared Layer Output")

    # temporal_data = out[0, :, :10, :]
    # temporal_data = temporal_data.reshape(-1, temporal_data.shape[-1])
    # visualize(temporal_data, v_type, v_dim)

s_out_hook = model.module.shared_attn_layers[-1].register_forward_hook(sharing_out_hook)

def sharing_in_hook(module, input, output):
    ####################### input #######################
    x = input[0].cpu() # x (batch_size, in_steps, num_node, feature)
    spatial_data = x[1, :, :, :]
    spatial_data = spatial_data.reshape(-1, spatial_data.shape[-1])
    visualize(spatial_data, v_type, v_dim, "(a) First Shared Layer Input")

    # temporal_data = x[0, :, :10, :]
    # temporal_data = temporal_data.reshape(-1, temporal_data.shape[-1])
    # visualize(temporal_data, "umap", dim=2)

s_in_hook = model.module.shared_attn_layers[0].register_forward_hook(sharing_in_hook)

# test
model.eval()
model.module.mask_ratio = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(trainer.device)
        model(x_batch, is_train=True)
        break