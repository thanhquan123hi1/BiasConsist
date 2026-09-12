# BiasConsist

This repository is a research project for the **`BiasConsist`** deepfake detector framework.
It provides a DeepfakeBench-style training and evaluation pipeline focused on parameter-efficient CLIP bias-tuning and consistency learning.

## Models

The project includes several detector variants:

- **`BiasConsistency`**: Weak-to-strong consistency learning with bias-only parameter adaptation.
- **`BiasArtifactConsistency`**: Artifact-preserving consistency learning.
- **`BiasEMATeacherConsistency`**: EMA-teacher consistency learning.
- **`BiasLoraAsy`**: CLIP ViT backbone with LoRA adapters on the last attention blocks, trainable backbone biases, and asymmetric supervised contrastive loss.

Main files:

- `training/detectors/BiasConsistency.py`
- `training/detectors/BiasArtifactConsistency.py`
- `training/detectors/BiasEMATeacherConsistency.py`
- `training/detectors/BiasLoraAsy.py`
- `training/config/detector/`
- `training/train.py`
- `training/test.py`

## Setup

Install the Python dependencies used by the training pipeline:

```bash
pip install -r requirements.txt
```

Place frame data and dataset JSON files wherever you prefer, then update:

- `rgb_dir`
- `dataset_json_folder`
- `log_dir`

in `training/config/train_config.yaml` and `training/config/test_config.yaml`.

The dataset loader expects DeepfakeBench-style JSON metadata, for example:

```text
preprocessing/dataset_json/FaceForensics++.json
datasets/rgb/<dataset>/.../frames/*.png
```

## Training

```bash
python training/train.py \
  --detector_path training/config/detector/BiasLoraAsy.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v2"
```

Train the bias-only CLIP detector with weak-to-strong consistency:

```bash
python training/train.py \
  --detector_path training/config/detector/BiasConsistency.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v2" "FaceShifter" "DeeperForensics-1.0"
```

`BiasConsistency` trains only CLIP vision-backbone parameters whose names end
in `.bias`, plus the binary classifier. Its objective is the mean
cross-entropy of weak/strong views plus a scheduled KL divergence from the
detached weak-view prediction to the strong-view prediction. Evaluation uses a
single image view.

Train the artifact-preserving variant:

```bash
python training/train.py \
  --detector_path training/config/detector/BiasArtifactConsistency.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v2" "FaceShifter" "DeeperForensics-1.0"
```

`BiasArtifactConsistency` keeps bias-only CLIP adaptation, but restricts KL to
confident weak predictions that agree with the ground-truth label. It averages
KL over the selected samples like `BiasConsistency`, gives the
artifact-preserving weak view more CE weight, and uses a milder strong
augmentation policy.

Train the EMA-teacher variant:

```bash
python training/train.py \
  --detector_path training/config/detector/BiasEMATeacherConsistency.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "FaceForensics++"
```

`BiasEMATeacherConsistency` keeps the artifact-preserving objective, but the
weak-view teacher is a frozen exponential moving average of the student model
instead of the current student weak branch.

Optional fine-tuning from a checkpoint:

```bash
python training/train.py \
  --detector_path training/config/detector/BiasLoraAsy.yaml \
  --weights_path path/to/checkpoint.pth
```

## Evaluation

```bash
python training/test.py \
  --detector_path training/config/detector/BiasLoraAsy.yaml \
  --test_dataset "Celeb-DF-v2" \
  --weights_path path/to/checkpoint.pth
```

Use `--save_feat --feat_out_dir <dir>` to dump feature pickles for t-SNE or
other analysis.

## Git & Repository
```bash
git clone https://github.com/thanhquan123hi1/BiasConsist.git
cd BiasConsist
```

If connecting an existing local clone:
```bash
git remote set-url origin https://github.com/thanhquan123hi1/BiasConsist.git
git push -u origin main
```

