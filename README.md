This repository contains the code used in paper "CLEAR: Ranked Multi-Positive Contrastive Representation Learning for Robust Trajectory Similarity Computation"
# Requirements
- Ubuntu OS
- Python 3.9.13 (tested)
- PyTorch 1.13.0 (tested)

 # Preprocessing
 We mainly follow [t2vec](https://github.com/boathit/t2vec#readme) to preprocess the datasets but reproduce all Julia scripts in Python.
 We support two trajectory datasets of different moving objects. They're [**Porto**](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i) and [**GeoLife**](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/). Taking **Porto** as an example, our preprocessing includes several steps:
 1. Unify the datasets in different formats:  
    ```python preprocess/preprocess.py -dataset_name "porto"```  
    Then you'll get a .pkl file called "porto.pkl" in "data/porto".  
 2. Data augmentation.  
    ```python preprocess/augmentation.py```  
    Then you'll get a .pkl file named such as "porto_distort_rate_0.2.pkl" in "data/porto/augmentation". Feel free to use multiprocessing :-)
3. Token generation.  
   ```python preprocess/grid_partitioning.py -dataset_name "porto"```  
   Then you'll get a series of .pkl files named such as "porot_distort_rate_0.2_token.pkl" in "data/porto/token/cell-100_minfreq-50". Again, feel free to use multiprocessing. Meanwhile, the node2vec for the partitioned space will also be done here. You can find the node embedding file "../data/porto/porto_size-256_cellsize-100_minfreq-50_node2vec.pkl"

# Training
You can train CLEAR with the following settings.   
```python main.py -dataset_name "porto" -combination "multi" -loss "pos-rank-out-all" -model_name "clear-DualRNN" -pretrain_mode "pf" -pretrain_method "node2vec" -batch_size 64 -cell_size 100 -minfreq 50 -aug1_name "distort" -aug1_rate 0.4 -aug2_name "downsampling" -aug2_rate 0.4```  
The trained model will be saved in "{}_checkpoint.pt" and "{}_best.pt". To facilicate the ablation study, they'll be named such as "clear-DualRNN_grid_cell-100_minfreq-50_multi-downsampling-distort-246_pos-rank-out-all_batch-64_pretrain-node2vec-pf_porto_checkpoint.pt" and saved in "data/porto". To reproduce the results of the other variants mentioned in our paper, you can modify the parameters such as combination, loss, batch_size, cell_size, and minfreq to corresponding values.  

# Evaluation
We support three types of evaluation metrics, i.e., "self-similarity", "cross-similarity" and "knn". Taking "self-similarity" as an example, you can follow the next steps to reproduce the results.  
1. Prepare the experimental dataset.  
   ```python ./experiment/experiment.py -mode data -dataset_name "porto" -exp_list "self-similarity" -partition_method "grid" -cell_size 100 -minfreq 50```  
   Then you'll get the experimental dataset in "experiment/self-similarity/porto/cellsize-100-minfreq-50".  
2. Encode.  
   ```python ./experiment/experiment.py -mode encode -dataset_name "porto" -exp_list "self-similarity" -partition_method "grid" -cell_size 100 -minfreq 50 -combination "multi" -loss "pos-rank-out-all" -batch_size 64 -aug1_name "distort" -aug1_rate 0.4 -aug2_name "downsampling" -aug2_rate 0.4 -model_name "clear-DualRNN" -pretrain_mode "pf" -pretrain_method "node2vec"```  
   Then you'll get the encoded vector for self-similarity experimental set named with a suffix corresponding to your model, in "experiment/self-similarity/porto/cellsize-100-minfreq-50"
3. Experiment.
   ```python experiment.py -mode encode -dataset_name "porto" -exp_list "self-similarity" -spatial_type "grid" -cell_size 100 -minfreq 50 -combination "single" -loss "pos-rank-out-all" -batch_size 64 -aug1_name "distort" -aug1_rate 0.4 -aug2_name "downsampling" -aug2_rate 0.4 -model_name "clear-DualRNN" -pretrain_mode "pf" -pretrain_method "node2vec"```
   Then you'll get the experimental results (.csv file) in "experiment".

We put the unified data file and our trained model of Porto in [data](https://drive.google.com/drive/folders/1WoLxTSLKfbSblL0tfTHBUQQb9mfhJ0BC?usp=sharing).

# Reference
[CLEAR](https://ieeexplore.ieee.org/abstract/document/10591640) has been accepted by [IEEE MDM 2024](https://mdm2024.github.io/) and selected as the [Best paper runner-up](https://mdm2024.github.io/awards.html).

If you find our work useful and inspiring, please consider citing our paper:
```
@inproceedings{li2024clear,
  title={CLEAR: Ranked Multi-Positive Contrastive Representation Learning for Robust Trajectory Similarity Computation},
  author={Li, Jialiang and Liu, Tiantian and Lu, Hua},
  booktitle={2024 25th IEEE International Conference on Mobile Data Management (MDM)},
  pages={21--30},
  year={2024},
  organization={IEEE}
}
```
 
