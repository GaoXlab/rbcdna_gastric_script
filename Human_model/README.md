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
# 1. Feature selection from whole-genome features
# Hardware requirement: At least 48 cores and 96GB RAM
./script/step2.sh gc

# 2. Feature reduction and model training
python ./script/step3.py gc `pwd`

# 3. For the independent test sets
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

# 4. Permutation test

Note: Scripts with _rnd suffix are used for permutation testing. Before running, first execute shuffle_sample_info.py to generate the necessary configuration files.

Then follow the steps described in the previous section, replacing step2.sh with step2_rnd.sh

```bash
# You should copy gc.p100.ids.txt to gc_rnd.p100.ids.txt
cp modelData/{gc,gc_rnd}.p100.ids.txt

./script/step2_rnd.sh gc_rnd
python ./script/step3_rnd.py gc_rnd `pwd`
python ./script/step3_test_rnd.py gc_rnd `pwd` test
```

# 5. Nested cross-validation evaluation (In response to reviewer comments)
```bash
./script/run_full_cv_test.sh gc
```

# 6. Performance evaluation of simplified models using selected original rbcDNA regions
(1) Feature reduction based on discriminatory score ranking
```bash
python script/step3_top_origin_feature.py gc `pwd`
```
(2) Feature reduction based on PCA-derived contribution ranking
```bash
python ./script/step3_ori_topn_feature.py gc `pwd`
```
