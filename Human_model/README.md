# This file shows how to use the pipeline module to reproduce the results of the paper.

# 1. directory structure
```text
├── fq
├── bams
├── modelData
│   ├── empty
│   │   ├── cleaned
│   │   └── origin
│   └── trim_q30_gcc_10k_cpm
│       ├── cleaned
│       └── origin
├── results
│   ├── 2_FeatureSelection
│   ├── 3_FeatureReduction
│   └── 4_Classification
└── script 
```
You should put bam files in bam directory, and the module data in the modelData directory. The results will be saved in the "results" directory.

# 2. run the pipeline
```bash
# Build 10k cpm data
./script/step1.sh gc

# gc pipeline
# Hardware requirement: At least 48 cores and 96GB RAM
# 1. Nested cross-validation evaluation
./script/run_full_cv_test.sh gc

# 2. Feature selection from whole-genome features
./script/step2.sh gc

# 3. Feature reduction and model training
python ./script/step3.py gc `pwd`

# 4. For the independent test sets
TYPE="gc"
## Build feature data for each independent test set
./script/build_feature_data.sh modelData/"$TYPE".ind1.ids.txt  "$TYPE"
./script/build_feature_data.sh modelData/"$TYPE".ind2_sd.ids.txt  "$TYPE"
./script/build_feature_data.sh modelData/"$TYPE".ind3_ay.ids.txt  "$TYPE"
## Merge all test sets into final tab file
./script/make_all_tab.sh "$TYPE" "$TYPE/all.$TYPE.tab"
python ./script/step3_test.py gc `pwd` test 
python ./script/step3_test.py gc `pwd` ind1
python ./script/step3_test.py gc `pwd` ind2_sd
python ./script/step3_test.py gc `pwd` ind3_ay
``` 

Feature selection results will be saved in the 2_FeatureSelection directory, feature reduction results will be saved in the 3_FeatureReduction directory, and classification results will be saved in the 4_Classification directory.


# 3. Prediction for additional cancer types and blood sample storage duration experiments
```bash
## for additional cancer types 
./script/build_feature_data.sh modelData/"$TYPE".otherca.ids.txt  "$TYPE"
./script/make_all_tab.sh "$TYPE" "$TYPE/all.$TYPE.tab"
python ./script/step3_test.py gc `pwd` otherca

## for blood sample storage duration experiments
./script/build_feature_data.sh modelData/"$TYPE".storagetime.ids.txt  "$TYPE"
./script/make_all_tab.sh "$TYPE" "$TYPE/all.$TYPE.tab"
python ./script/step3_test.py gc `pwd` storagetime
``` 


# 4. Performance evaluation of simplified models using selected original rbcDNA regions (for response)
(1) Feature reduction based on discriminatory score ranking
```bash
python script/step3_top_origin_feature.py gc `pwd`
```
(2) Feature reduction based on PCA-derived contribution ranking
```bash
python ./script/step3_ori_topn_feature.py gc `pwd`
```
