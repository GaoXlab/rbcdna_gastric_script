suppressPackageStartupMessages({
  library(dplyr)
  library(readxl)
  library(readr)
})
excel_file <- "./Figures/Supplementary_Tables.xlsx"

sampleinfo <- read_excel(excel_file, sheet = "Supplementary Table 1")

save(sampleinfo, file = "./Figures/sampleinfo.RData")

## predictive scores
pred_trn   <- read_csv('Human_model/results/4_Classification/gc_prediction_trncv.csv', show_col_types = FALSE)
pred_test  <- read_csv('Human_model/results/4_Classification/gc_prediction_test.csv', show_col_types = FALSE)
pred_zhejiang    <- read_csv('Human_model/results/4_Classification/gc_prediction_ind_zj.csv', show_col_types = FALSE)
pred_shandong    <- read_csv('Human_model/results/4_Classification/gc_prediction_ind2_sd.csv', show_col_types = FALSE)
pred_anyang <- read_csv('Human_model/results/4_Classification/gc_prediction_ind3_ay.csv', show_col_types = FALSE)

align_prediction <- function(pred_df, info_df) {
  pred_df %>%
    rename(any_of(c(Sample = "seqID"))) %>%
    rename(any_of(c(final_prob = "0"))) %>%
    select(Sample, final_prob, lr, cb, source_key) %>%
    inner_join(info_df %>% select(Sample, Group), by = "Sample") %>%
    mutate(Target = ifelse(Group %in% c("GC"), 1, 0)) %>%
    # 【关键修改】：剔除 Group，只输出纯粹的三列
    select(Sample, Target, final_prob, lr, cb, source_key)
}

GC_trncv_pred      <- align_prediction(pred_trn, sampleinfo[, c('Sample', 'Group')])
GC_test1_pred     <- align_prediction(pred_test, sampleinfo[, c('Sample', 'Group')])
GC_ind1_pred       <- align_prediction(pred_zhejiang, sampleinfo[, c('Sample', 'Group')])
GC_ind2_pred       <- align_prediction(pred_shandong, sampleinfo[, c('Sample', 'Group')])
GC_ind3_pred <- align_prediction(pred_anyang, sampleinfo[, c('Sample', 'Group')])

save(GC_trncv_pred, GC_test1_pred, 
     GC_ind1_pred, GC_ind2_pred, GC_ind3_pred, file = "./Figures/prediction.RData")