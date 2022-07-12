# CS4248 - Natural Language Processing
## Grammar Error Correction with Bidirectional Encoder Representations from Transformers (GECwBERT)
Authors: Moon Geonsik, Rui Zhi, Surthi, Tian Yong, Xingquan

GECwBERT is a model which consists of two main tasks 1) Grammar Error Detection(GED) and 2) Grammar Error Correction(GEC). For this model, GED will be used to fine-tune the pretrained model and GEC will be used to correct the sentences.

### Grammar Error Detection (GED)
GED is used to fine-tune the pre-trained model to detect grammatical error. Here, we have fine-tuned the pretrain model using four different corpora (NUCLE, CoLA, FCE, W&I- LOCNESS) individually.

### Grammar Error Correction (GEC)
GEC only depends on the Mask Language Model(MLM). In other words, there is no annotated training being used. The sentences will be reformulated through the help of MLM and the four fine-tuned models done by GED. 

### Fine-tuned Corpora
  - [NUCLE](https://sterling8.d2.comp.nus.edu.sg/nucle_download/nucle.php)
  - [FCE](https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz)
  - [GECwBERT](https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz) 
  - [CoLA](https://nyu-mll.github.io/CoLA/cola_public_1.1.zip)

### Downloads
- Install miniconda
- Download **pre-processed data** & **fine-tune models**
  pre-processed data: (Google Drive)
    - url: https://drive.google.com/drive/folders/1EhVFTZ3t6WrPjeABvgAvlRDg5HqjAcWL?usp=sharing
  pre-trained model: (Google Drive) 
    - url: https://drive.google.com/drive/folders/1XnKulXvX2zJOMaM3WQU5zYJvoJrho2zU?usp=sharing

### Dependencies
- PyTorch version >= 1.9.0
- Python version >= 3.8.0
- pytorch_pretrained_bert >= 0.6.2
- pandas >= 1.3.4
- sklearn >= 0.0
- numpy >= 1.21.2
- keras >= 2.7.0
- tqdm >= 4.62.3
- spacy >= 2.2.4
- hunspell >= 2.0.2
```
conda env create -f environment.yml
source ~/.bashrc
conda activate myenv
python -m spacy download en_core_web_sm
```

### Preprocess Data
Files required:
1. nucle.train.gold.bea19.m2 (From NUCLE)
1. fce.train.gold.bea19.m2 (From FCE)
1. ABC.train.gold.bea19.m2 (From W&I+LOCNESS)
```
python3 m2preprocess.py
```

### Fine-tune Model using GED
Files required:
1. data_nucle.csv
1. data_ABC.csv (from W&I+Locness)
1. data_fce.csv
1. in_domain_train.tsv (From CoLA)
1. out_domain_train.tsv (From CoLA)
- Produce four fine-tuned models by executing the first four files(1-4) as inputs
- Recommended number of epochs : 4

```
python3 ged_train.py [input_file.csv/ input_file.tsv] [output_model.pth] [number of epoch]
```

### Predict using GEC

Files required:
1. model fine-tuned by CoLA (e.g. cola-GED-4epochs.pth)
1. model fine-tuned by NUCLE (e.g. nucle-GED-4epochs.pth)
1. model fine-tuned by W&I Locness (e.g. ABC-GED-4epochs.pth)
1. model fine-tuned by FCE (e.g. fce-GED-4epochs.pth)
1. en_GB-large.aff
1. en_GB-large.dic
1. test_file.m2 (e.g. offcial-2014.combined.m2)
```
python3 gec_modified.py [cola-GED-4epochs.pth] [nucle-GED-4epochs.pth] [ABC-GED-4epochs.pth] [fce-GED-4epochs.pth] [offcial-2014.combined.m2]
```

### Evaluation (F0.5-score)
Files required:
1. input_sentences.txt (modified_4_model_input.txt from GEC)
1. output_sentences.txt (modified_4_model_output.txt from GEC)

Setting up the environment for evaluation:
```
python3 -m venv errant_env
source errant_env/bin/activate
pip3 install -U pip setuptools wheel
pip3 install errant
python3 -m spacy download en
```

Merge the **input_sentences.txt** and **output_sentences.txt** into .m2 format:
```
errant_parallel -orig [input_sentences.txt] -cor [output_sentences.txt] -out [out.m2]
```

Calculate F0.5-Score:
```
errant_compare -hyp [output.m2] -ref [test_file.m2]
```

### Acknowledgments
Our code was modified from [GECwBERT](https://github.com/sunilchomal/GECwBERT) codebase.
