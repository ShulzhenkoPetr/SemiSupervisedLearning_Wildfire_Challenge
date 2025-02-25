# SemiSupervisedLearning_Wildfire_Challenge

For a more detailed review of the project and results, please refer to the file Project_report.pdf

This project is based on a more complex version of the [Wildfire Prediction Dataset (Satellite Images) Challenge](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/data). The focus is on **binary classification** of satellite images into **wildfire** and **no-wildfire** categories.  

## Key Challenges & Objectives  
- **No access to original dataset annotations**, encouraging exploration of **transfer learning, active learning, semi-supervised, and unsupervised methods**.  
- **Independent implementation**: Except for one pre-trained model, all methods, pipelines, and utilities are implemented in **PyTorch** without third-party libraries.

## Usage

1. Install required dependencies (Hydra is used for config management in the training pipeline)
2. Downlaod the data with data/download_data.sh script; analyze with data_exploration.ipynb
3. For training, testing refer to the scipts train.py and test.py
4. For semi-supervised learning via pseudo-labeling refer to preuso_labels_training.py
