# DLGNet
DLGNet: A dual-branch lesion-aware network with the supervised Gaussian Mixture model for colon lesions classification in colonoscopy images

Our paper has been accepted by Medical Image Analysis.

## Training the Model

python train_test.py

train_dataset-root: Folder to which you downloaded and extracted the training data

val_datapath-root: Folder to which you downloaded and extracted the val data

record_path: The path where the training results are stored

model_path = The path where the model is stored

best_path = The path where the model with the best result on the validation set is stored

First go into the train_test and adapt all the paths to match your file system and the download locations of training and test sets.

Then python train_test.py to train your dataset.

## Citation

If you find the code useful for your research, please cite our paper.

Wang, Kai-Ni, et al. "DLGNet: A dual-branch lesion-aware network with the supervised Gaussian Mixture model for colon lesions classification in colonoscopy images." Medical Image Analysis 87 (2023): 102832.
