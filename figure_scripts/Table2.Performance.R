args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(pROC)
  library(stringr)
  library(ggplot2)
  library(cowplot)
  library(dplyr)
  library(ggplotify)
  library(patchwork)
  library(openxlsx)
  library(clinfun)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)

## sampleinfo:
load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

## predictive scores
GC_ind1_pred$source_key  = 'ind-Zhejiang'
GC_ind2_pred$source_key  = 'ind-Shandong'
GC_ind3_pred$source_key  = 'ind-Henan'
GC_ind_pred <- as.data.frame(rbind(GC_ind1_pred, GC_ind2_pred, GC_ind3_pred))
GC_test2_pred = GC_ind_pred[GC_ind_pred$source_key=='ind-Zhejiang',]
GC_test3_pred_s1 = GC_ind_pred[GC_ind_pred$source_key=='ind-Henan',]
GC_test3_pred_s2 = GC_ind_pred[GC_ind_pred$source_key=='ind-Shandong',]

## cutoff determined
cutoff_spe90 = Cutoff( 0.90, GC_trncv_pred)
cutoff_spe95 = Cutoff( 0.95, GC_trncv_pred)

trn_hd_90 = nrow(GC_trncv_pred[(GC_trncv_pred$Target==0)&(GC_trncv_pred$final_prob<cutoff_spe90),])
trn_hd_95 = nrow(GC_trncv_pred[(GC_trncv_pred$Target==0)&(GC_trncv_pred$final_prob<cutoff_spe95),])
trn_hd = nrow(GC_trncv_pred[GC_trncv_pred$Target==0,])
cutoff <- setNames(
  list(
    round((trn_hd_90/trn_hd)*100,2),
    round((trn_hd_95/trn_hd)*100,2)
  ),
  c('> 90% specificity', '> 95% specificity')
)
cutoff_df <- data.frame(
  Label = names(cutoff),
  Value = unlist(cutoff),
  Cutoff = c(cutoff_spe90, cutoff_spe95)
)

## sensitivity at 90% specificity
all_re = c()
for(cutoff in c(0.9, 0.95)){ # 0.85, 0.9, 0.95
      cutoff_at_spe = Cutoff(cutoff, GC_trncv_pred)

      GC_trncv_pred2 = getinfo(GC_trncv_pred)
      trn_tmp = get_sensitivity_inxspe(GC_trncv_pred2, cutoff_at_spe)
      trn_tmp = trn_tmp[which((trn_tmp$Var1!='trn')), ]
      trn_tmp_spe = get_HDspecificity_inxspe(GC_trncv_pred2, cutoff_at_spe)
      trn_tmp_spe = trn_tmp_spe[which(trn_tmp_spe$Var1=='Total'), ]

      GC_test1_pred2 = getinfo(GC_test1_pred)
      test1_tmp = get_sensitivity_inxspe(GC_test1_pred2, cutoff_at_spe)
      test1_tmp = test1_tmp[which((test1_tmp$Var1!='test')), ]
      test1_tmp_spe = get_HDspecificity_inxspe(GC_test1_pred2, cutoff_at_spe)
      test1_tmp_spe = test1_tmp_spe[which(test1_tmp_spe$Var1=='Total'), ]

      GC_ind1_pred2 = getinfo(GC_test2_pred)
      ind1_tmp = get_sensitivity_inxspe(GC_ind1_pred2, cutoff_at_spe)
      ind1_tmp = ind1_tmp[which((ind1_tmp$Var1!='ind-Zhejiang') & (ind1_tmp$Var1!='ind-Henan') & (ind1_tmp$Var1!='ind-Shandong')), ]
      ind1_tmp_spe = get_HDspecificity_inxspe(GC_ind1_pred2, cutoff_at_spe)
      ind1_tmp_spe = ind1_tmp_spe[which(ind1_tmp_spe$Var1=='ZHEJIANG'), ]

      GC_ind2_pred2 = getinfo(GC_test3_pred_s1)
      ind2_tmp = get_sensitivity_inxspe(GC_ind2_pred2, cutoff_at_spe)
      ind2_tmp = ind2_tmp[which((ind2_tmp$Var1!='ind-Zhejiang') & (ind2_tmp$Var1!='ind-Henan') & (ind2_tmp$Var1!='ind-Shandong')), ]
      ind2_tmp_spe = get_HDspecificity_inxspe(GC_ind2_pred2, cutoff_at_spe)
      ind2_tmp_spe = ind2_tmp_spe[which(ind2_tmp_spe$Var1=='ANYANG'), ]

      GC_ind3_pred2 = getinfo(GC_test3_pred_s2)
      ind3_tmp = get_sensitivity_inxspe(GC_ind3_pred2, cutoff_at_spe)
      ind3_tmp = ind3_tmp[which((ind3_tmp$Var1!='ind-Zhejiang') & (ind3_tmp$Var1!='ind-Henan') & (ind3_tmp$Var1!='ind-Shandong')), ]
      ind3_tmp_spe = get_HDspecificity_inxspe(GC_ind3_pred2, cutoff_at_spe)
      ind3_tmp_spe = ind3_tmp_spe[which(ind3_tmp_spe$Var1=='SHANDONG'), ]

      GC_ind_total = getinfo(rbind(GC_test2_pred, GC_test3_pred_s1,GC_test3_pred_s2))
      ind_total_tmp = get_sensitivity_inxspe(GC_ind_total, cutoff_at_spe)
      ind_total_tmp = ind_total_tmp[which((ind_total_tmp$Var1!='ind-Zhejiang') & (ind_total_tmp$Var1!='ind-Henan') & (ind_total_tmp$Var1!='ind-Shandong')), ]
      ind_total_tmp_spe = get_HDspecificity_inxspe(GC_ind_total, cutoff_at_spe)
      ind_total_tmp_spe = ind_total_tmp_spe[which(ind_total_tmp_spe$Var1=='Total'), ]

      trn_tmp2 <- trn_tmp %>% select(Var1, Freq, Detected, SEN_95Ci) %>% rename(Discovery_Freq = Freq, Discovery_Detected = Detected, Discovery_SEN_95Ci = SEN_95Ci)
      test1_tmp2 <- test1_tmp %>% select(Var1, Freq, Detected, SEN_95Ci) %>% rename(Test_Freq = Freq, Test_Detected = Detected, Test_SEN_95Ci = SEN_95Ci)
      ind1_tmp2 <- ind1_tmp %>% select(Var1, Freq, Detected, SEN_95Ci) %>% rename(IND1_Freq = Freq, IND1_Detected = Detected, IND1_SEN_95Ci = SEN_95Ci)
      ind2_tmp2 <- ind2_tmp %>% select(Var1, Freq, Detected, SEN_95Ci) %>% rename(IND2_Freq = Freq, IND2_Detected = Detected, IND2_SEN_95Ci = SEN_95Ci)
      ind3_tmp2 <- ind3_tmp %>% select(Var1, Freq, Detected, SEN_95Ci) %>% rename(IND3_Freq = Freq, IND3_Detected = Detected, IND3_SEN_95Ci = SEN_95Ci)
      ind_total_tmp2 <- ind_total_tmp %>% select(Var1, Freq, Detected, SEN_95Ci) %>% rename(IND_total_Freq = Freq, IND_total_Detected = Detected, IND_total_SEN_95Ci = SEN_95Ci)
      all <- purrr::reduce(list(trn_tmp2, test1_tmp2, ind1_tmp2, ind2_tmp2, ind3_tmp2, ind_total_tmp2), dplyr::full_join, by = "Var1")
      all$Var1=as.character(all$Var1); all[which(all$Var1=='Total'), 'Var1'] <- 'GC'
      all$Var1 <- factor(all$Var1, levels = c("GC", "earlyGC", "advGC", "II", "III", "Intestinal", "Diffuse", "Mix", "Missing"))
      all <- all[which((ind3_tmp$Var1!='ZHEJIANG') & (ind3_tmp$Var1!='ANYANG') & (ind3_tmp$Var1!='SHANDONG')),  ]
      all$cutoff <- str_c("at ", round(cutoff * 100, 2), " % specificity: ", cutoff_at_spe)
      colnames(all) = c('Var1', rep(c('N', 'Detected', 'Performance (95% CI)'), 6), "cutoff")

      trn_tmp2_spe <- trn_tmp_spe %>% select(Var1, Freq, Detected, SPE_95Ci) %>% rename(Discovery_Freq = Freq, Discovery_Detected = Detected, Discovery_SPE_95Ci = SPE_95Ci)
      test1_tmp2_spe <- test1_tmp_spe %>% select(Var1, Freq, Detected, SPE_95Ci) %>% rename(Test_Freq = Freq, Test_Detected = Detected, Test_SPE_95Ci = SPE_95Ci)
      ind1_tmp2_spe <- ind1_tmp_spe %>% select(Var1, Freq, Detected, SPE_95Ci) %>% rename(IND1_Freq = Freq, IND1_Detected = Detected, IND1_SPE_95Ci = SPE_95Ci)
      ind2_tmp2_spe <- ind2_tmp_spe %>% select(Var1, Freq, Detected, SPE_95Ci) %>% rename(IND2_Freq = Freq, IND2_Detected = Detected, IND2_SPE_95Ci = SPE_95Ci)
      ind3_tmp2_spe <- ind3_tmp_spe %>% select(Var1, Freq, Detected, SPE_95Ci) %>% rename(IND3_Freq = Freq, IND3_Detected = Detected, IND3_SPE_95Ci = SPE_95Ci)
      ind_total_tmp2_spe <- ind_total_tmp_spe %>% select(Var1, Freq, Detected, SPE_95Ci) %>% rename(IND_total_Freq = Freq, IND_total_Detected = Detected, IND_total_SPE_95Ci = SPE_95Ci)
      all_spe <- purrr::reduce(list(trn_tmp2_spe, test1_tmp2_spe, ind1_tmp2_spe, ind2_tmp2_spe, ind3_tmp2_spe, ind_total_tmp2_spe), dplyr::full_join, by = "Var1")
      all_spe <- all_spe[(all_spe$Var1), ] 
      all_spe$Var1=as.character(all_spe$Var1); all_spe[which(all_spe$Var1=='Total'), 'Var1'] <- 'Non-GC'
      all_spe$cutoff <- str_c("Specificity at ", cutoff_at_spe)
      colnames(all_spe) = c('Var1', rep(c('N', 'Detected', 'Performance (95% CI)'), 6), "cutoff")

      all = rbind(all_spe, all)
      label_row <- as.data.frame(matrix("", nrow = 1, ncol = ncol(all)))
      colnames(label_row) <- colnames(all)
      label_row[1, ] <- c("Classification", "Discovery", "", "", "Test", "", "", "IND 1", "", "", "IND 2", "", "", "IND 3", "", "", "IND total", "", "", "cutoff")

      all <- rbind(label_row, all)
      all_re = rbind(all_re, all)
}
all_re = as.data.frame(all_re)
old_colnames <- colnames(all_re)

colnames(all_re) <- as.character(all_re[1, ])
all_re <- all_re[-1, ]
all_re <- rbind(old_colnames, all_re)
rownames(all_re) <- NULL

write.xlsx(list(cutoff_df, all_re), file.path(out_dir, 'Table2_performance.xlsx'))







