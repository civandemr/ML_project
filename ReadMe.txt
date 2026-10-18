'''
ACL 2022 long paper
https://aclanthology.org/2022.acl-long.52/
First author: HongSeok Choi (hongking9@gist.ac.kr)
Corresponding author: Hyunju Lee (hyunjulee@gist.ac.kr)
Code for Early Stopping Based on Unlabeled Samples in Text Classification.
Free for non-commercial use.
'''

## We used Anaconda, and follows the below commands to implement the virtual enviroment.
conda create --name env_name python=3.6
conda activate env_name 
conda install -c anaconda tensorflow-gpu=2.2.0
conda install -c conda-forge transformers=3.3.1
conda install -c anaconda scikit-learn
conda install -c conda-forge dataclasses=0.7
conda install -c conda-forge sentencepiece=0.1.91

## First, you need pre-trained parameters before running 'main.py'. 
## You can download the pre-trained parameters (e.g. BERT-base-uncased) by running "download_bert_params.py".
## The file name of the pre-trained parameters should be "pytorch_model.bin".
## Also, if you already have tensorflow checkpoint ('~.ckpt'),then you can convert it to '~.bin' file by using transformers-cli from "https://huggingface.co/transformers/converting_tensorflow_models.html".

## You can run cell-by-cell by using "run-cell-by-cell.ipynb" in jupyter notebook.
## (This ipynb script will help easy undertanding about our algorithm.)

## "setup_experiments.py" can generate five seeds balanced and imbalanced settings (K=50) of SST-2 dataset. (See the paper for detail)

## For example, if you want to use gpu1 and experiment with the balanced classification, then the command is as below. 
for t in {0..4}; do CUDA_VISIBLE_DEVICES=1 python main.py --task SST-2 --data_path bal/$t --seed $t --log_prefix bal_ --word_freeze True; done

## If you want to use gpu2 and experiment with the imbalanced classification, then the command is as below.
for t in {0..4}; do CUDA_VISIBLE_DEVICES=2 python main.py --task SST-2 --data_path imbal/$t --seed $t --log_prefix imbal_ --word_freeze True; done

## In script "main.py", all the hyperparameters are described and the default values also are set.

## You can check the results in 'logs' folder.

## New versions can be updated continuously at "https://github.com/DMCB-GIST/BUS-stop/"