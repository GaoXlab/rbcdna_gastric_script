args <- commandArgs(trailingOnly = TRUE)
working_dir <- if (length(args) >= 1) gsub("~\\+~", " ", args[1]) else getwd()
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_path <- gsub("~\\+~", " ", script_path)
script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))
if (!file.exists(file.path(script_dir, "plot_function.r"))) {
  script_dir <- file.path(working_dir, "figure_scripts")
}

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
  library(dplyr)
  library(ggpubr)
  library(openxlsx)
  library(tidyr)
  library(MatchIt)
  library(ggsci)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'or_function.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

GC_ind1_pred$source_key  = 'ind-Zhejiang'
GC_ind2_pred$source_key  = 'ind-Shandong'
GC_ind3_pred$source_key  = 'ind-Henan'
GC_ind_pred <- as.data.frame(rbind(GC_ind1_pred, GC_ind2_pred, GC_ind3_pred))
GC_prediction <- as.data.frame(GC_ind_pred)

## cutoff determined
cutoff_spe90 <- Cutoff(0.90, GC_trncv_pred)

## GC model performance
needed_cols <- c(
  'Sample', 'Group', 'Source', 'Age', 'Gender',
  'Smoking status', 'Alcohol status',
  'Lauren classification', 'Grade', 'Tumor location',
  'Helicobacter pylori', 'Tumor size group',
  'CEA (ng/mL)', 'CA19-9 (U/mL)', 'CA242 (U/mL)',
  'RBC (×10^12/L)', 'HGB (g/L)', 'WBC (×10^9/L)', 'PLT (×10^9/L)',
  'RDW (%)', 'MCV (fL)', 'MCHC (g/L)', 'MCH (pg)',
  'NEU (×10^9/L)', 'LYM (×10^9/L)',
  'NLR', 'PLR', 'SII', 'LMR'
)
missing_cols <- setdiff(needed_cols, colnames(sampleinfo))
if (length(missing_cols) > 0) {
  stop("Missing columns in sampleinfo: ", paste(missing_cols, collapse = ", "))
}
sampleinfo_tmp <- sampleinfo[, needed_cols]

sampleinfo_col_map <- c(
  'Source' = 'dataset_label',
  'Smoking status' = 'Smoking.state',
  'Alcohol status' = 'Alcohol.state',
  'Lauren classification' = 'Lauren.classification',
  'Tumor location' = 'Tumor.location',
  'Helicobacter pylori' = 'HP',
  'Tumor size group' = 'Tumor.size.group',
  'CEA (ng/mL)' = 'CEA',
  'CA19-9 (U/mL)' = 'CA199',
  'CA242 (U/mL)' = 'CA242',
  'RBC (×10^12/L)' = 'RBC',
  'HGB (g/L)' = 'Hb',
  'WBC (×10^9/L)' = 'WBC',
  'PLT (×10^9/L)' = 'PLT',
  'RDW (%)' = 'RDW',
  'MCV (fL)' = 'MCV',
  'MCHC (g/L)' = 'MCHC',
  'MCH (pg)' = 'MCH',
  'NEU (×10^9/L)' = 'NEU',
  'LYM (×10^9/L)' = 'LYM'
)
colnames(sampleinfo_tmp) <- ifelse(
  colnames(sampleinfo_tmp) %in% names(sampleinfo_col_map),
  unname(sampleinfo_col_map[colnames(sampleinfo_tmp)]),
  colnames(sampleinfo_tmp)
)
sampleinfo_tmp$dataset_label <- factor(sampleinfo_tmp$dataset_label, levels = c("ZHEJIANG", "ANYANG", "SHANDONG"))

GC_pred_m <- merge(GC_prediction, sampleinfo_tmp, by='Sample')

GC_pred_m$Group <- factor(GC_pred_m$Group, levels=c('Non-GC','GC'))

GC_pred_m$Age.group <- "≥ 60"
GC_pred_m$Age.group[GC_pred_m$Age < 60] <- '< 60'
GC_pred_m$Age.group <- factor(GC_pred_m$Age.group, levels=c('< 60','≥ 60'))

GC_pred_m$Smoking.state <- gsub(" smoker", "", GC_pred_m$Smoking.state)
GC_pred_m$Smoking.state <- factor(GC_pred_m$Smoking.state, levels=c('Current','Prior','Never','No record'))

GC_pred_m$Alcohol.state <- gsub(" consumed", "", GC_pred_m$Alcohol.state)
GC_pred_m$Alcohol.state <- factor(GC_pred_m$Alcohol.state, levels=c('Current','Prior','Never','No record'))

GC_pred_m$Grade <- factor(GC_pred_m$Grade, levels=c('Poor','Moderate/Poor','Moderate','Moderate/Well','Well','Missing'))

GC_pred_m$Tumor.location <- as.character(GC_pred_m$Tumor.location)
GC_pred_m$Tumor.location[GC_pred_m$Tumor.location == "Distal stomach"] <- "Distal"
GC_pred_m$Tumor.location[GC_pred_m$Tumor.location == "Proximal stomach"] <- "Proximal"
GC_pred_m$Tumor.location <- factor(GC_pred_m$Tumor.location, levels=c('Distal','Proximal','Total stomach'))

GC_pred_m$HP <- factor(GC_pred_m$HP, levels=c('No','Yes','Unknown'))
levels(GC_pred_m$HP)[levels(GC_pred_m$HP)=="Unknown"] <- "Missing"

GC_pred_m$Tumor.size.group <- factor(GC_pred_m$Tumor.size.group, levels=c('<3cm','3-5cm','>=5cm','Missing'))
levels(GC_pred_m$Tumor.size.group)[levels(GC_pred_m$Tumor.size.group)==">=5cm"] <- "≥ 5cm"

GC_pred_m$dataset_label <- factor(GC_pred_m$dataset_label, levels=c("ZHEJIANG", "ANYANG", "SHANDONG"))

y_max <- 1.03

theme_sig3 = theme_sig2 + theme(axis.text.x = element_text(angle = 45, hjust = 1))

p1_score_gender <- ggplot(data = GC_pred_m, aes(x = Gender, y = final_prob)) +
  geom_hline(yintercept=cutoff_spe90, color='red4', linetype='dashed', linewidth=0.2)+
  geom_boxplot(aes(color=dataset_label), outlier.colour = NA, position = position_dodge(width = 0.75), linewidth = 0.4)+
  geom_jitter(aes(color=dataset_label), shape=16, position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.75), size=0.5, alpha=0.7)+
  scale_color_manual(values=c('#293E90','#478AC9','#0097A6FF')) +
  stat_compare_means(aes(label = paste0(after_stat(method), ",\n", after_stat(p.signif))),
                     method = "wilcox.test", label.x.npc = 'center', label.y = 1, size = 5 / .pt, hjust = 0.5, lineheight = 0.65) +
  facet_grid(.~Group) + theme_sig2 + ylab('rbcDNA predictive scores') + xlab('Sex') + ylim(0, y_max)

p1_score_age <- ggplot(data = GC_pred_m, aes(x = Age.group, y = final_prob)) +
  geom_hline(yintercept=cutoff_spe90, color='red4', linetype='dashed', linewidth=0.2)+
  geom_boxplot(aes(color=dataset_label), outlier.colour = NA, position = position_dodge(width = 0.75), linewidth = 0.4)+
  geom_jitter(aes(color=dataset_label), shape=16, position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.75), size=0.5, alpha=0.7)+
  scale_color_manual(values=c('#293E90','#478AC9','#0097A6FF')) +
  stat_compare_means(aes(label = paste0(after_stat(method), ",\n", after_stat(p.signif))),
                     method = "wilcox.test", label.x.npc = 'center', label.y = 1, size = 5 / .pt, hjust = 0.5, lineheight = 0.65) +
  facet_grid(.~Group) + theme_sig2 + ylab('rbcDNA predictive scores') + xlab('Age (year)') + ylim(0, y_max)

shared_legend <- get_legend(
  ggplot(data = GC_pred_m, aes(x = Age.group, y = final_prob, color=dataset_label)) +
  geom_point(size = 3, shape=15) + scale_color_manual(values=c('#293E90','#478AC9','#0097A6FF')) +
  theme_sig2 + theme(legend.position='bottom', legend.title=element_blank(), legend.direction="horizontal",
      legend.background = element_rect(fill = NA, color = NA), legend.box.background = element_rect(fill = NA, color = NA),
      legend.text = element_text(size = 8, color = "black"), legend.key = element_blank(), legend.margin = margin(0, 0, 0, 0), legend.box.margin = margin(0, 0, 0, 0))
)

p2_alcohol <- plot_score_subgroup(GC_pred_m, "Alcohol.state", "History of alcohol consumption", facet_by_group = TRUE, label_sep = ",\n", rotate_x = TRUE)
p2_smoke <- plot_score_subgroup(GC_pred_m, "Smoking.state", "History of smoking", facet_by_group = TRUE, label_sep = ",\n")
p2_score_HP <- plot_score_subgroup(GC_pred_m, "HP", "H. pylori infection status", facet_by_group = TRUE)
p2_score_lauren <- plot_score_subgroup(GC_pred_m[which((GC_pred_m$Group=='GC') & (GC_pred_m$Lauren.classification!='Missing')), ], "Lauren.classification", "Lauren classification")
p2_score_size <- plot_score_subgroup(GC_pred_m[which((GC_pred_m$Group=='GC') & (GC_pred_m$Tumor.size.group!='Missing')), ], "Tumor.size.group", "Tumor size group")
p2_score_location <- plot_score_subgroup(GC_pred_m[which((GC_pred_m$Group=='GC') & (GC_pred_m$Tumor.location!='Stomach')), ], "Tumor.location", "Tumor location")
p2_score_grade <- plot_score_subgroup(GC_pred_m[which((GC_pred_m$Group=='GC') & (GC_pred_m$Grade!='Missing')), ], "Grade", "Grade")

#### sensitivity for anemia、tumor-marker negative、inflammation


GC_pred_m <- GC_pred_m %>%
  mutate(
    HGB_trans=case_when(
      Gender=="Male" & `Hb`<130 ~ 0, Gender=="Female" & `Hb`<120 ~ 0,
      Gender=="Male" & `Hb`>=130 ~ 200, Gender=="Female" & `Hb`>=120 ~ 200,
      TRUE ~ NA_real_))

all_sen = as.data.frame(rbind(
    get_performance_by_variable(GC_pred_m, "CEA", 5, cutoff_spe90, direction=">="),
    get_performance_by_variable(GC_pred_m, "CEA", 5, cutoff_spe90, direction="<"),
    get_performance_by_variable(GC_pred_m, "CA199", 37, cutoff_spe90, direction=">="),
    get_performance_by_variable(GC_pred_m, "CA199", 37, cutoff_spe90, direction="<"),
    get_performance_by_variable(GC_pred_m, "CA242", 20, cutoff_spe90, direction=">="),
    get_performance_by_variable(GC_pred_m, "CA242", 20, cutoff_spe90, direction="<"),
    get_performance_by_variable(GC_pred_m, "HGB_trans", 100, cutoff_spe90, direction="<"),
    get_performance_by_variable(GC_pred_m, "HGB_trans", 100, cutoff_spe90, direction=">=")
))

forest_data <- bind_rows(
  all_sen %>%
    transmute(Subgroup,Metric="Sensitivity",Estimate=Sensitivity,Lower=SEN.low,Upper=SEN.up,Lab=SEN_n),
  all_sen %>%
    transmute(Subgroup,Metric="Specificity",Estimate=Specificity,Lower=SPE.low,Upper=SPE.up,Lab=SPE_n)
) %>% mutate(
    Subgroup=case_when(
      Subgroup=="CEA >= 5" ~ "CEA positive\n(CEA ≥ 5)",
      Subgroup=="CEA < 5" ~ "CEA negative\n(CEA < 5)",
      Subgroup=="CA199 >= 37" ~ "CA19-9 positive\n(CA19-9 ≥ 37)",
      Subgroup=="CA199 < 37" ~ "CA19-9 negative\n(CA19-9 < 37)",
      Subgroup=="CA242 >= 20" ~ "CA24-2 positive\n(CA24-2 ≥ 20)",
      Subgroup=="CA242 < 20" ~ "CA24-2 negative\n(CA24-2 < 20)",
      Subgroup=="HGB_trans >= 100" ~ "No anemia\n(Male, Hb ≥ 130 g/L;\nFemale, Hb ≥ 120 g/L)",
      Subgroup=="HGB_trans < 100" ~ "Anemia\n(Male, Hb < 130 g/L;\nFemale, Hb < 120 g/L)",
      TRUE ~ Subgroup
    ),
    Subgroup=factor(
      Subgroup,
      levels=rev(c(
        "CEA positive\n(CEA ≥ 5)", "CEA negative\n(CEA < 5)",
        "CA19-9 positive\n(CA19-9 ≥ 37)", "CA19-9 negative\n(CA19-9 < 37)",
        "CA24-2 positive\n(CA24-2 ≥ 20)", "CA24-2 negative\n(CA24-2 < 20)",
        "No anemia\n(Male, Hb ≥ 130 g/L;\nFemale, Hb ≥ 120 g/L)",
        "Anemia\n(Male, Hb < 130 g/L;\nFemale, Hb < 120 g/L)",
        "Normal CRP levels\n(CRP < 10 mg/L)", "High CRP levels\n(CRP ≥ 10 mg/L)"
      ))
    ),
    Metric=factor(
      Metric,
      levels=c("Sensitivity","Specificity"),
      labels=c(
        "Sensitivity",
        "Specificity"
      )
    )
  )

pF <- ggplot(forest_data,aes(x=Estimate,y=Subgroup))+
  geom_errorbarh(aes(xmin=Lower,xmax=Upper),height=0.18,linewidth=0.4)+
  geom_point(size=2,shape=21,fill="#E64B35",color="#333333",stroke=0.3)+
  geom_text(aes(x=-4,label=Lab),hjust=0,size=6/.pt)+
  facet_grid(.~Metric)+ labs(x="Performance, % (95% CI)",y=NULL)+ theme_sig2 + 
  theme(strip.background=element_blank(),strip.text=element_text(size=8,face="plain"),
    axis.line.y=element_blank(), axis.ticks.y=element_blank(),
    panel.spacing.x=unit(0.2,"cm"))+
  coord_cartesian(clip="off") 

####
cbc_vars <- c(c('SII','NLR','PLR','WBC','PLT'), rev(c('RBC','Hb','MCV','MCHC','RDW')))
score_var <- "final_prob"
group_var <- "Group" 
group_levels <- c('Non-GC', 'GC')
group_labels <- c('Non-GC', 'GC')

cor_by_group <- GC_pred_m %>%
  filter(.data[[group_var]] %in% c('Non-GC', 'GC')) %>%
  group_by(.data[[group_var]]) %>% group_split() %>%
  purrr::map_dfr(function(dat) {
    g <- unique(dat[[group_var]])
    purrr::map_dfr(cbc_vars, ~ cor_test_one(dat, .x, score_var)) %>% mutate(group = g)
  })

cor_df <- cor_by_group %>%
  mutate(
    group = factor(group, levels = group_levels),
    variable = factor(variable, levels = cbc_vars),
    sig = case_when(is.na(pvalue) ~ "",pvalue < 0.001 ~ "***",pvalue < 0.01 ~ "**",pvalue < 0.05 ~ "*",TRUE ~ ""),
    label = ifelse(is.na(cor), "", paste0(sprintf("%.2f", cor), sig))
  )

pG<- ggplot(cor_df, aes(x = variable, y = group, fill = cor)) +
  geom_tile(color = "white", linewidth = 0.4) +
  annotate("rect", xmin = 0.5, xmax = 10.5, ymin = 0.5, ymax = 2.5, fill = NA, color = "grey", linewidth = 0.3) +
  geom_text(aes(label = label),size = 6/.pt, color = "black") +
  scale_fill_gradient2(low = "#2166AC",mid = "white",high = "#B2182B",midpoint = 0,limits = c(-1, 1),breaks = c(-1, -0.5, 0, 0.5, 1),name = "Correlation\ncoefficient") +
  scale_y_discrete(limits = rev(group_levels), labels = group_labels) +
  labs(title = "Spearman correlation between\nrbcDNA predictive scores and parameters") + coord_flip() +
  fig5_theme_common(base_size = 8, base_family = "", axis_line = FALSE, legend_position = "bottom",  plot_margin = margin(t = 5, r = 10, b = 12, l = 5)) +
  theme(axis.title = element_blank(),axis.ticks = element_blank(),legend.title = element_text(size = 6),legend.text = element_text(size = 6),
	  panel.border = element_blank(),plot.title = element_text(size = 8, color = "black", hjust = 0.5),
    plot.caption = element_text(size = 6, color = "black", hjust = 0),
     legend.key.height = grid::unit(2, "mm"),
     legend.spacing.x = grid::unit(1, "mm"),
     legend.spacing.y = grid::unit(0, "mm"),
     legend.box.spacing = grid::unit(0, "mm")
  )

### Univariate and multivariable-adjusted logistic regression
inds_df <- make_model_df(GC_pred_m)
analysis_vars <- c(
  "rbcDNA_score_z", "age", "sex", "Smoking", "Alcohol", "Hp",
  "Hb", "RBC", "WBC", "PLT", "RDW", "MCV", "MCHC", "MCH", "NEU", "LYM", "NLR", "PLR", "SII", "LMR")
univariate_table_ori <- run_univariate_set(inds_df, analysis_vars)
print(univariate_table_ori)

res_an <- fit_gc_model(inds_df, c("rbcDNA_score_z", "age", "sex"))
res_an2 <- fit_gc_model(inds_df, c("rbcDNA_score_z", "age", "sex", "Hb"))
res_an3 <- fit_gc_model(inds_df, c("rbcDNA_score_z", "age", "sex", "Hb", "RDW", "NLR"))
res_an3_without_rbcDNA <- fit_gc_model(inds_df, c("age", "sex", "Hb", "RDW", "NLR"))
res_an4 <- fit_gc_model(inds_df, c("rbcDNA_score_z", "age", "sex", "Smoking", "Alcohol", "Hp", "Hb", "RDW", "NLR"))

#### After PSM
set.seed(1234)
psm_data <- inds_df %>%
  filter(disease_status %in% c("GC", "Non-GC")) %>%
  drop_na(age, sex, disease_status) %>%
  mutate(age_group = cut(age, breaks = seq(0, 100, 10), right = FALSE))
m.out <- matchit(disease_status ~ age_group + sex, data = psm_data, method = "optimal", exact = "age_group", caliper = 0.5)
matched_data <- match.data(m.out)
summary(m.out)
table(matched_data$disease_status)
chisq.test(matched_data$sex, matched_data$disease_status)
compare_means(age ~ disease_status, data = matched_data)

univariate_table_psm <- run_univariate_set(matched_data, analysis_vars)
print(univariate_table_psm)

psm_model_vars <- c("rbcDNA_score_z", "Hb", "RDW", "NLR")
res_an3_psm <- fit_gc_model(matched_data, psm_model_vars)
res_an3_without_rbcDNA_psm <- fit_gc_model(matched_data, setdiff(psm_model_vars, "rbcDNA_score_z"))

univariate_export_cols <- c("term", "n", "event_n", "non_event_n", "OR_95CI", "P_value", "significance")
univariate_value_cols <- setdiff(univariate_export_cols, "term")

univariate_table_export <- full_join(
  univariate_table_ori %>%
    select(all_of(univariate_export_cols)) %>%
    rename_with(~ paste0("Original__", .x), all_of(univariate_value_cols)),
  univariate_table_psm %>%
    select(all_of(univariate_export_cols)) %>%
    rename_with(~ paste0("After_PSM__", .x), all_of(univariate_value_cols)),
  by = "term"
) %>%
  mutate(
    term = display_term(term)
)

univariate_header <- c("term", univariate_value_cols, univariate_value_cols)
univariate_group_header <- c(
  "",
  "Original independent validation sets", rep("", length(univariate_value_cols) - 1),
  "After PSM", rep("", length(univariate_value_cols) - 1)
)

wb_univariate <- createWorkbook()
addWorksheet(wb_univariate, "Univariate")
writeData(wb_univariate, "Univariate", t(as.data.frame(univariate_group_header)), startRow = 1, colNames = FALSE)
writeData(wb_univariate, "Univariate", t(as.data.frame(univariate_header)), startRow = 2, colNames = FALSE)
writeData(wb_univariate, "Univariate", univariate_table_export, startRow = 3, colNames = FALSE)
mergeCells(wb_univariate, "Univariate", cols = 2:(length(univariate_value_cols) + 1), rows = 1)
mergeCells(wb_univariate, "Univariate", cols = (length(univariate_value_cols) + 2):(length(univariate_value_cols) * 2 + 1), rows = 1)
saveWorkbook(wb_univariate, file = file.path(out_dir, "univariate_table.xlsx"), overwrite = TRUE)


model_list <- list(
  "Model1" = res_an, "Model2" = res_an2, "Model3" = res_an3, "Model4_full_adjusted" = res_an4,
  "Model3_without_rbcDNA" = res_an3_without_rbcDNA,
  "Model3_PSM" = res_an3_psm,
  "Model3_PSM_without_rbcDNA" = res_an3_without_rbcDNA_psm
)

model_export_cols <- c("term", "statistic", "OR_95CI", "P_value", "significance")

model_labels <- c(
  "Model1" = "Model1", "Model2" = "Model2", "Model3" = "Model3", "Model4_full_adjusted" = "Model4 full-adjusted",
  "Model3_without_rbcDNA" = "Model3 (without rbcDNA scores)",
  "Model3_PSM" = "After PSM, Model3",
  "Model3_PSM_without_rbcDNA" = "After PSM, Model3 (without rbcDNA score)"
)

main_model_keys <- c("Model1", "Model2", "Model3", "Model4_full_adjusted")
sensitivity_model_keys <- c("Model3_without_rbcDNA", "Model3_PSM", "Model3_PSM_without_rbcDNA")

blank_model_row <- tibble(
  Model_formula = NA_character_, term = NA_character_, statistic = NA_real_,
  OR_95CI = NA_character_, P_value = NA_real_, significance = NA_character_)

multivariable_or_export <- bind_rows(
  lapply(main_model_keys, function(model_key) {format_model_section(model_key, model_list[[model_key]])}),
  list(blank_model_row),
  lapply(sensitivity_model_keys, function(model_key) {format_model_section(model_key, model_list[[model_key]])})
)

write.xlsx(as.data.frame(multivariable_or_export),file = file.path(out_dir, "Supplementary_multivariable_OR.xlsx"),overwrite = TRUE)

p_all <- make_or_forest(
  res_an3$or_table,
  "Multivariable logistic regression\nDisease ~ rbcDNA score + covariates",
  rbc_only = TRUE)
p_all_psm <- make_or_forest(res_an3_psm$or_table, "Age- and sex-matched analysis")
pH <- plot_grid(p_all, p_all_psm, ncol=1, rel_heights=c(0.6, 1), align="v")

aligned_row1 <- cowplot::align_plots(
    p1_score_gender, p1_score_age, p2_alcohol, p2_smoke,align = "hv",axis = "tblr")

row1_plots <- plot_grid(aligned_row1[[1]], aligned_row1[[2]], aligned_row1[[3]], aligned_row1[[4]],
    ncol = 4, rel_widths = c(1.15, 1.15, 1, 1), labels = c("A", "B", "C", ""),
    label_size = 12,label_fontface = "bold",label_x = 0.01,label_y = 0.96,hjust = 0,vjust = 0,align = "hv",axis = "tblr")

row1_with_leg <- ggdraw() +
  draw_plot(row1_plots,x = 0,y = 0.03,width = 1,height = 0.93) +
  draw_grob(shared_legend,x = 0.08,y = 0.04,width = 0.42,height = 0.10)

aligned_score_row <- cowplot::align_plots(
    p2_score_HP, p2_score_lauren, p2_score_size, p2_score_location, p2_score_grade, align = "hv",axis = "tblr")

row2_plots <- plot_grid(plotlist = aligned_score_row,ncol = 5,
                        rel_widths = c(1.5, 1, 1, 1, 1.15), 
                        labels=c('D','E','',''), label_size = 12, label_fontface = "bold", label_x = 0.01,label_y = 1,hjust = 0,vjust = 0) +
                        theme(plot.margin = margin(t = 5, unit = "pt"))

row3_plots = plot_grid(pF, pG, pH, ncol=3, rel_widths=c(1.8,1,1.2),
              labels=c('F','G','H'), label_size = 12, label_fontface = "bold", label_x = 0.01,label_y = 1,hjust = 0,vjust = 0)+
                        theme(plot.margin = margin(t = 5, unit = "pt"))

Fig5_Final <- plot_grid(row1_with_leg, row2_plots, row3_plots, ncol=1, rel_heights=c(1,1,1.2))

ggsave(file.path(out_dir, 'Figure5.pdf'), Fig5_Final, width=8, height=8)

model3_adjusted_pred <- make_adjusted_pred(inds_df, res_an3, res_an3_without_rbcDNA)
psm_adjusted_pred <- make_adjusted_pred(matched_data, res_an3_psm, res_an3_without_rbcDNA_psm)

write.table(model3_adjusted_pred, file.path(out_dir, "model3_adjusted_pred.log"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(psm_adjusted_pred, file.path(out_dir, "psm_adjusted_pred.log"), sep = "\t", row.names = FALSE, quote = FALSE)
