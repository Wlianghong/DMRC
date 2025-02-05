from main import main_train
from config import Config

# import random
# seeds = [random.randint(10**7, 10**8 - 1) for _ in range(5)]
# print(seeds)

seeds = [51970489, 34110658, 74418540, 20436023, 93048409]

config = Config("DMRCMLP", "PEMS04")
mask_ratio = config.config["model_args"]["mask_ratio"]
config.set("save", False)

for seed in seeds:
    config.set("seed", seed)
    # DMRCFormer
    config.config["model_args"]["use_recon"] = True
    config.set("adaptive_mask", True)
    main_train(config)

for seed in seeds:
    config.set("seed", seed)
    # with reconstruction, and with fixed masking
    config.config["model_args"]["use_recon"] = True
    config.set("adaptive_mask", False)
    main_train(config)

for seed in seeds:
    config.set("seed", seed)
    # without reconstruction, but with diminishing masking
    config.config["model_args"]["use_recon"] = False
    config.set("adaptive_mask", True)
    main_train(config)

for seed in seeds:
    config.set("seed", seed)
    # without reconstruction, and without masking
    config.config["model_args"]["use_recon"] = False
    config.set("adaptive_mask", False)
    config.config["model_args"]["mask_ratio"] = 0.
    main_train(config)

    # reset mask ratio
    config.config["model_args"]["mask_ratio"] = mask_ratio


