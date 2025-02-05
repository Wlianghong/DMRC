# DMRC
**Diminishingly Masked Reconstruction-Constrained Transformer (DMRC)** is an end-to-end model that integrates a prediction branch for future traffic flow and a reconstruction branch for restoring masked historical data. It uses a gradually decaying fine-grained mask, with a higher masking ratio early in training to distinguish effective features from noise, and a lower ratio later to refine prediction accuracy.

## File Tree
* `DMRC` The root directory, containing code for training and testing, configuration file reading, logging, etc.
  * `data` Datasets.
  * `libs` Some  widgets.
  * `logs` The logs for training and testing.
  * `model_ref` The source code of the reference model STAEFormer.
  * `models` The code of DMRC.
  * `save_models` The stored trained DMRC model files.
  * `tools` Code for visualization and analysis.