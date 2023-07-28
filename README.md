This repository contains the code used in paper "Clear: Ranked Multi-Positive Contrastive Representation Learning for Robust Trajectory Similarity Computation"
## Requirements
- Ubuntu OS
- Python >= 3.8
- PyTorch 1.13.0 (tested)

 ## Preprocessing
 We mainly follow [t2vec](https://github.com/boathit/t2vec#readme) to preprocess the datasets but reproduce all Julia scripts in Python.
 We suppport three trajectory datasets of different moving objects. They're "[porto](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)", "[geolife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)" and "[aisus](https://marinecadastre.gov/ais/)". Taking "porto" as example, our preprocessing includes several steps:
 1. Unify the datasets in diffrent formats:
    '
    python execute.py -dataset_name "porto"
    '
    
 
