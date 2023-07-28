This repository contains the code used in paper "CLEAR: Ranked Multi-Positive Contrastive Representation Learning for Robust Trajectory Similarity Computation"
# Requirements
- Ubuntu OS
- Python = 3.9.13 (tested)
- PyTorch 1.13.0 (tested)

 # Preprocessing
 We mainly follow [t2vec](https://github.com/boathit/t2vec#readme) to preprocess the datasets but reproduce all Julia scripts in Python.
 We suppport three trajectory datasets of different moving objects. They're "[porto](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)", "[geolife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)" and "[aisus](https://marinecadastre.gov/ais/)". Taking "porto" as example, our preprocessing includes several steps:
 1. Unify the datasets in diffrent formats:  
    ```python unify/run_unify.py -dataset_name "porto"```  
    Then you'll get a .csv file called "porto_filter.csv" and a .h5 file called "porto_filter.csv" in "../data/porto".  
 2. Data augmentation.  
    ```python augmentation/run_augmentation.py -dataset_name "porto"```  
    Then you'll get a series of .h5 file named such as "porto_distort_rate_0.2.h5" in "../data/porto/augmentation". Feel free to use multiprocessing :-)
3. Token generation.  
   ```python token_generation/run_token.py -dataset_name "porto" -cell_size 100 -minfreq 50```  
   Then you'll get a series of .h5 file named such as "porot_distort_rate_0.2_seq.h5" in "../data/porto/token/cell-100_minfreq-50". Again, feel free to use multiprocessing.

# Training
You can train CLEAR with the following settings.   
```python main.py -dataset_name "porto" -combination "single" -loss "pos-rank-out-all" -batch_size 64 -spatial_type "grid" -cell_size 100 -minfreq 50 -aug1_name "distort" -aug1_rate 0.4 -aug2_name "downsampling" -aug2_rate 0.4```  
The trained model will be saved in "{}_checkpoint.pt" and "{}_best.pt". To facilicate the ablation study, they'll be named such as "clear-S_grid_cell-100_minfreq-50_multi-single-downsampling-distort-246_pos-rank-out-all_batch-64_porto_checkpoint.pt". To reproduce the results of the other variants mentioned in our paper, you can modify the parameters such as combination, loss, batch_size, cell_size and minfreq to corresponding values.  

# Evaluation
We support three types of evaluation metrics, i.e., "self-similarity", "cross-similarity" and "knn". Taking "self-similarity" as an example, you can follow the next steps to reproduce the results.  
1. Prepare experimental dataset.  
   ```python experiment.py -mode data -dataset_name "porto" -exp_list "self-similarity" -spatial_type "grid" -cell_size 100 -minfreq 50```  
   Then you'll get the experimental dataset in "../experiment/self-similarity/porto/cell-100-minfreq-50".  
2. Encode and evaluate.  
   ```python experiment.py -mode data -dataset_name "porto" -exp_list "self-similarity" -spatial_type "grid" -cell_size 100 -minfreq 50```  
   Then you'll get the experimental results (.csv file) in "../experiment/".

   
 
