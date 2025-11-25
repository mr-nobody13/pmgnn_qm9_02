# PAMNet-QM9

This is a simplified version of the **Physics‑aware Multiplex Graph Neural Network (PAMNet)** repository that focuses exclusively on small‑molecule property prediction on the QM9 dataset.  All code and data related to RNA and protein tasks have been removed to reduce complexity and potential dependency issues.  In addition, the code has been updated to be compatible with more recent versions of PyTorch, PyTorch Geometric and other dependencies.

## Installation

Install the required packages using the provided `requirements.txt` file.  It is recommended to create a fresh Python virtual environment.

```bash
pip install -r requirements.txt
```

The script will automatically download and preprocess the QM9 dataset on first use.

## Usage

To train and evaluate PAMNet on the QM9 dataset run:

```bash
python main_qm9.py --dataset QM9 --model PAMNet --target 7 --epochs 900 --batch_size 32 --dim 128 --n_layer 6 --lr 1e-4
```

This will train the model on approximately 110k molecules and report MAE on a validation and test split.  Checkpoints will be saved under `./save/QM9`.

## Notes

* Only the QM9 dataset is supported.  Code paths and dependencies for PDBbind and RNA have been removed.
* The code has been updated to use `torch_geometric.loader.DataLoader` to avoid deprecated APIs in recent versions of PyTorch Geometric.
* If you encounter issues related to RDKit, ensure that the RDKit version specified in `requirements.txt` is installed.

