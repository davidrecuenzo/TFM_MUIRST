# TFM_MUIRST

## 1. Introduction

This repository consists of the code necessary to develop the Master's Thesis of the Master's Degree in Telematic Networks and Services. It is a repository that uses Federated Learning techniques to train and test a Deep Learning model of an Intrusion Detection System (IDS) based on anomaly detection.

Author: David Recuenzo Bermejo.

## 2. Installation

Required software:

- Python 3.7
- EMQX
- Anaconda

To follow the installation of the product you must follow the steps below:

```bash
conda create -n tfm python=3.7
conda activate tfm
pip install torch torchvision
pip install -r requirements.txt 
cd FedML
pip install -r requirements.txt
cd ../
```

## 3. Data Preparation

Run `sh download.sh` under `data/UCI-MLR` to download dataset.
You must execute `python3 min_max.sh` under `experiments/distributed` to obtain min and max datas from N-BaIoT dataset.
If you want to know how many samples there are for each CSV of the dataset, you can execute the notebook `FeatureDataset.ipynb` under `experiments/distributed`.


## 4. Code Structure of repository

- `FedML` and `FedML-IoT`: a repository where you can find the necessary scripts of the FedML python framework required for the operation of the architecture.

- `data`: provide data downloading scripts `sh download.sh` and store the downloaded datasets. It is necessary to unzip the compressed folders of the attacks.

- `data_preprocessing`: data loaders, partition methods and utility functions.

- `model`: IoT models: AE, VAE and MLP.

- `training`: provide script to train models. In this folder when training the model, a file called model.ckpt will be saved, which is necessary to evaluate the performance.

- `experiments/distributed`: Script to evaluate the performance. Run `python3 fl_test_*.py` to obtain metrics.

- `experiments/Raspberry Pi`: 
1. It is the code designed for the implementation on the Raspberry Pi 4B.
2. It contains two blocks, `main_uci_rp_*.py` should be implemented on the edge device and `app_*.py` should be implemented on the server.

## 5. Run the repository

In Federated Server, you must run to train model:

```bash
emqx start
python3 app_*.py
```

In Federated client:

```bash
conda activate tfm
python3 main_uci_rp_*.py --server_ip http://<server_ip>:5000 --client_uuid <uuid_of_client>
```

## 6. Results

Use the `fl_test_*` to evaluate the performance. You must run `python3 fl_test_*.sh` under `experiments/distributed`