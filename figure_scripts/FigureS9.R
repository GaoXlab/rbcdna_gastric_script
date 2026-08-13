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
  library(pROC)
  library(stringr)
  library(ggplot2)
  library(cowplot)
  library(dplyr)
  library(tidyr)
  library(ggsci)
  library(openxlsx)
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
  'Sample', 'Group', 'Age', 'Gender',
  'HGB (g/L)', 'RDW (%)', 'NLR'
)
missing_cols <- setdiff(needed_cols, colnames(sampleinfo))
if (length(missing_cols) > 0) {
  stop("Missing required columns in sampleinfo: ", paste(missing_cols, collapse = ", "))
}
sampleinfo_tmp <- sampleinfo[, needed_cols]
sampleinfo_col_map <- c(
  'HGB (g/L)' = 'Hb',
  'RDW (%)' = 'RDW'
)
colnames(sampleinfo_tmp) <- ifelse(
  colnames(sampleinfo_tmp) %in% names(sampleinfo_col_map),
  sampleinfo_col_map[colnames(sampleinfo_tmp)],
  colnames(sampleinfo_tmp)
)

### test:
GC_pred_test <- merge(GC_test1_pred, sampleinfo_tmp, by='Sample')
GC_pred_test$Group <- factor(GC_pred_test$Group, levels=c('Non-GC','GC'))

df_test <- make_model_df(GC_pred_test)
test_model_vars <- c("rbcDNA_score_z", "age", "sex", "Hb", "RDW", "NLR")
test_res_an3 <- fit_gc_model(df_test, test_model_vars)
test_res_an3_without_rbcDNA <- fit_gc_model(df_test, setdiff(test_model_vars, "rbcDNA_score_z"))

p_test <- make_or_forest(test_res_an3$or_table, "Multivariable-adjusted logistic regression\nDisease ~ rbcDNA score + covariates")

a_test <- make_adjusted_pred(df_test, test_res_an3, test_res_an3_without_rbcDNA)
roc1_test = pROC::roc(a_test$Target,a_test$pred_with_rbcDNA, percent = TRUE) # 0.9884
roc2_test = pROC::roc(a_test$Target,a_test$pred_without_rbcDNA, percent = TRUE) # 0.9739

p1_sig_test <- p_value_to_text(pROC::roc.test(roc1_test, roc2_test, method="delong")$p.value)
test_curves <- list(
  get_roc_curve_info(roc2_test, "without rbcDNA", pal_material("grey")(10)[9], pal_material("grey", alpha=0.2)(10)[9]),
  get_roc_curve_info(roc1_test, "with rbcDNA", pal_material("deep-purple")(10)[10], pal_material("deep-purple", alpha=0.2)(10)[10])
)
p_test_auc <- plot_auc_panel(test_curves, title = "Test cohort\n", label_y = c(22, 8, 12), test_text_y = 4, test_text = p1_sig_test)

### independent validation sets:
model3_adjusted_pred = read.table(file.path(out_dir, "model3_adjusted_pred.log"), sep = "\t", head=TRUE)
psm_adjusted_pred = read.table(file.path(out_dir, "psm_adjusted_pred.log"), sep = "\t", head=TRUE)

roc1_psm = pROC::roc(psm_adjusted_pred$Target,psm_adjusted_pred$pred_with_rbcDNA, percent = TRUE) # 0.9884
roc2_psm = pROC::roc(psm_adjusted_pred$Target,psm_adjusted_pred$pred_without_rbcDNA, percent = TRUE) # 0.9739

p2_sig_test <- p_value_to_text(pROC::roc.test(roc1_psm, roc2_psm, method="delong")$p.value)

ind_curves <- list(
  get_roc_curve_info(roc2_psm, "(After PSM) without rbcDNA", pal_material("grey")(10)[9], pal_material("grey", alpha=0.2)(10)[9]),
  get_roc_curve_info(roc1_psm, "(After PSM) with rbcDNA", pal_material("light-blue")(10)[9], pal_material("light-blue", alpha=0.2)(10)[9])
)
p_ind <- plot_auc_panel(ind_curves, title = "Independent validation sets\n", label_y = c(22, 8, 12), test_text_y = 4, test_text = p2_sig_test)

make_performance_row <- function(dat, cohort, model, pred_col = "pred", best_method = "youden") {
  dat_eval <- dat %>%
    transmute(
      Target = as.numeric(.data[["Target"]]),
      pred = as.numeric(.data[[pred_col]])
    ) %>%
    filter(!is.na(Target), !is.na(pred))

  roc_obj <- pROC::roc(dat_eval$Target, dat_eval$pred, percent = TRUE, quiet = TRUE)
  best_coords <- pROC::coords(
    roc_obj,
    x = "best",
    best.method = best_method,
    ret = c("threshold", "sensitivity", "specificity"),
    transpose = FALSE
  )
  cutoff <- as.numeric(best_coords[["threshold"]][1])
  pred_class <- ifelse(dat_eval$pred >= cutoff, 1, 0)
  tp <- sum(pred_class == 1 & dat_eval$Target == 1)
  fn <- sum(pred_class == 0 & dat_eval$Target == 1)
  tn <- sum(pred_class == 0 & dat_eval$Target == 0)
  fp <- sum(pred_class == 1 & dat_eval$Target == 0)
  sens_ci <- binom.test(tp, tp + fn)$conf.int * 100
  spec_ci <- binom.test(tn, tn + fp)$conf.int * 100

  auc_ci <- pROC::ci.auc(roc_obj)

  data.frame(
    Cohort = cohort,
    Model = model,
    N = nrow(dat_eval),
    GC_n = sum(dat_eval$Target == 1),
    Non_GC_n = sum(dat_eval$Target == 0),
    Cutoff_method = "Youden",
    Cutoff = round(cutoff, 4),
    Sensitivity = round(100 * tp / (tp + fn), 2),
    Sensitivity_95CI = sprintf("%.2f-%.2f", sens_ci[1], sens_ci[2]),
    Specificity = round(100 * tn / (tn + fp), 2),
    Specificity_95CI = sprintf("%.2f-%.2f", spec_ci[1], spec_ci[2]),
    AUC = round(as.numeric(pROC::auc(roc_obj)), 2),
    AUC_95CI = sprintf("%.2f-%.2f", auc_ci[1], auc_ci[3]),
    `AUC (95% CI)` = sprintf("%.1f (%.1f-%.1f)", as.numeric(pROC::auc(roc_obj)), auc_ci[1], auc_ci[3]),
    `Sensitivity (95% CI)` = sprintf("%.1f (%.1f-%.1f)", 100 * tp / (tp + fn), sens_ci[1], sens_ci[2]),
    `Specificity (95% CI)` = sprintf("%.1f (%.1f-%.1f)", 100 * tn / (tn + fp), spec_ci[1], spec_ci[2]),
    check.names = FALSE,
    row.names = NULL
  )
}

make_model_performance_row <- function(model_res, cohort, model, best_method = "youden") {
  make_performance_row(
    model_res$data %>%
      transmute(Target = disease_numeric, pred = pred),
    cohort = cohort,
    model = model,
    pred_col = "pred",
    best_method = best_method
  )
}

performance_summary <- bind_rows(
  make_model_performance_row(test_res_an3, "Test cohort", "with rbcDNA"),
  make_model_performance_row(test_res_an3_without_rbcDNA, "Test cohort", "without rbcDNA"),
  make_performance_row(model3_adjusted_pred, "Independent validation sets", "with rbcDNA", "pred_with_rbcDNA"),
  make_performance_row(model3_adjusted_pred, "Independent validation sets", "without rbcDNA", "pred_without_rbcDNA"),
  make_performance_row(psm_adjusted_pred, "Independent validation sets after PSM", "with rbcDNA", "pred_with_rbcDNA"),
  make_performance_row(psm_adjusted_pred, "Independent validation sets after PSM", "without rbcDNA", "pred_without_rbcDNA")
)

write.xlsx(performance_summary, file.path(out_dir, "FigureS9_adjusted_model_performance.xlsx"), overwrite = TRUE)
write.table(performance_summary, file.path(out_dir, "FigureS9_adjusted_model_performance.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

performance_table <- performance_summary %>%
  mutate(
    Cohort = recode(Cohort,
                    "Independent validation sets after PSM" = "Independent validation\nsets after PSM",
                    "Independent validation sets" = "Independent validation\nsets"),
    Model = recode(Model,
                   "with rbcDNA" = "With rbcDNA",
                   "without rbcDNA" = "Without rbcDNA")
  ) %>%
  select(Cohort, Model, Cutoff, `AUC (95% CI)`, `Sensitivity (95% CI)`, `Specificity (95% CI)`)

make_performance_table_plot <- function(dat) {
  header <- colnames(dat)
  header[4:6] <- c("AUC\n(95% CI)", "Sensitivity\n(95% CI)", "Specificity\n(95% CI)")
  dat_plot <- dat
  colnames(dat_plot) <- header

  x_pos <- c(0.6, 1.55, 2.25, 3.2, 4.35, 5.5)
  col_width <- c(1.1, 0.8, 0.55, 1.1, 1.25, 1.25)
  body_cells <- expand.grid(row = seq_len(nrow(dat_plot)), col = seq_along(header))
  body_cells$label <- as.vector(as.matrix(dat_plot[, header]))
  body_cells$x <- x_pos[body_cells$col]
  body_cells$y <- body_cells$row + 1
  body_cells$fill <- ifelse(body_cells$row %% 2 == 0, "#F4F4F4", "white")

  header_cells <- data.frame(
    row = 0,
    col = seq_along(header),
    label = header,
    x = x_pos,
    y = 1,
    fill = "#EAEAEA"
  )

  table_cells <- bind_rows(header_cells, body_cells)

  ggplot(table_cells, aes(x = x, y = y)) +
    geom_tile(aes(width = col_width[col], fill = fill), height = 0.9, color = "grey70", linewidth = 0.25) +
    geom_text(aes(label = label, fontface = ifelse(row == 0, "bold", "plain")),
              size = 5.6 / .pt, lineheight = 0.85) +
    scale_fill_identity() +
    scale_y_reverse(expand = expansion(mult = c(0.02, 0.02))) +
    scale_x_continuous(limits = c(0, 6.15), expand = c(0, 0)) +
    coord_cartesian(clip = "off") +
    theme_void() +
    theme(plot.margin = margin(t = 2, r = 4, b = 2, l = 4))
}

p_table <- make_performance_table_plot(performance_table)

row_abc <- plot_grid(p_test, p_test_auc, p_ind, ncol=3, rel_widths=c(0.9,1,1), labels = c("A", "B", "C"),
         label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 0.98, hjust = 0, vjust = 0.5)
row_d <- plot_grid(p_table, ncol = 1, labels = c("D"),
                   label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 0.98, hjust = 0, vjust = 1)
FigS9 <- plot_grid(row_abc, row_d, ncol = 1, rel_heights = c(1, 0.56))

ggsave(file.path(out_dir, "FigureS9.pdf"), FigS9, width = 8, height = 4.6)
