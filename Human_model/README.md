# This file shows how to use the pipeline module to reproduce the results of the paper.

# 1. directory structure
```text
├── bam
├── modelData
│   └── empty
│       ├── cleaned
│       └── origin
├── results
│   ├── 2_FeatureSelection
│   ├── 3_FeatureReduction
│   └── 4_Classification
└── script 
```
You should put bam files in bam directory, and the module data in the modelData directory. The results will be saved in the results directory.

# 2. run the pipeline
```bash
# Build 10k cpm data
./script/step1.sh

# gc pipeline
# 1. Feature selection from whole genome features
./script/step2_gc.sh gc
# 2. Feature reduction and train model
python ./script/step3_gc_train.py gc `pwd`

## for the independent test set
python ./script/step3_gc_test.py gc `pwd` test 
python ./script/step3_gc_test.py gc `pwd` ind
python ./script/step3_gc_test.py gc `pwd` ind2
``` 
Feature selection results will be saved in the 2_FeatureSelection directory, feature reduction results will be saved in the 3_FeatureReduction directory, and classification results will be saved in the 4_Classification directory.