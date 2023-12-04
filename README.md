# RDE

## Requirements

### Packages

- python == 3.6.12
- numpy == 1.16.2
- pytorch == 1.6.0
- scipy == 1.5.2
- tqdm == 4.54.0
- logzero == 1.6.2
- scikit-learn == 0.23.2
- pyxclib (available in https://github.com/kunaldahiya/pyxclib)

### Hardware requirements

For datasets *Eurlex-4K* and *Wiki10-31K*, the ensemble models can be distributed to 1-4 GPUs, and a total of 16G GPU RAM is needed. For dataset *Amazon-670K*, 4 GPUs, each with 24G GPU RAM are needed.

## Data Preparation

#### Parabel, Bonsai, FastXML, PfastreXML

For baseline methods *Parabel*, *Bonsai*, *FastXML* and *PfastreXML*, we use Bag-of-Words (BoW) data, which can be downloaded from http://manikvarma.org/downloads/XC/XMLRepository.html. Please put files in the same location as below:

```
data
├── eurlex
│   ├── train.txt
│   └── test.txt
├── wiki10
│   ├── train.txt
│   └── test.txt
└── amazon670k
    ├── train.txt
    └── test.txt
```

#### X-Transformer

For baseline method *X-Transformer*, we use the same data as https://github.com/OctoberChang/X-Transformer. You can download the data from https://drive.google.com/drive/folders/1KUGFuJq6kBoFw_zhVPU0NndftTKD2hZk. Please put files in the same location as below:

```
data
├── eurlex-xtransformer
│   ├── X.trn.npz
│   ├── X.tst.npz
│   ├── Y.trn.npz
│   └── Y.tst.npz
└── wiki10-xtransformer
    ├── X.trn.npz
    ├── X.tst.npz
    ├── Y.trn.npz
    └── Y.tst.npz
```

## Running

### Pretrain baseline model

#### Parabel, Bonsai, FastXML, PfastreXML

For baseline methods *Parabel*, *Bonsai*, *FastXML* and *PfastreXML*, you should first convert train and test data:

```bash
python convert_data.py eurlex
```

Then, makefile for baseline methods:

```bash
make -C baseline/parabel/
```

Finally, run baseline methods:

```bash
sh baseline/run_parabel.sh eurlex
```

#### X-Transformer

For baseline method *X-Transformer*, we follow https://github.com/OctoberChang/X-Transformer to get the predicted results on training and test data. You can download the results from https://drive.google.com/drive/folders/1KUGFuJq6kBoFw_zhVPU0NndftTKD2hZk. Please put files in the same location as below:

```
data
├── eurlex-xtransformer
│   └── xtransformer
│       ├── trn.pred.npz
│       └── tst.pred.npz
└── wiki10-xtransformer
    └── xtransformer
        ├── trn.pred.npz
        └── tst.pred.npz
```

### Test baseline results

```bash
python main.py -d eurlex -b parabel --test-baseline
```

### Train RDE

```bash
python main.py -d eurlex -b parabel --train
```

