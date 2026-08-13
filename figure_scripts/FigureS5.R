args <- commandArgs(trailingOnly = TRUE)
working_dir <- if (length(args) >= 1) gsub("~\\+~", " ", args[1]) else getwd()
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_path <- gsub("~\\+~", " ", script_path)
script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))
if (!file.exists(file.path(script_dir, "or_function.r"))) {
  script_dir <- file.path(working_dir, "figure_scripts")
}

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(pROC)
  library(ggplot2)
  library(cowplot)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'or_function.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')


GC_prediction <- as.data.frame(rbind(GC_trncv_pred, GC_test1_pred))

cols <- c('Sample', 'Group', 'Stage', 'Atrophic', 'IntestinalMetaplasia', 'CEA (ng/mL)', 'CA19-9 (U/mL)', 'CA242 (U/mL)')
sampleinfo_tmp <- sampleinfo[, intersect(colnames(sampleinfo), cols)]
colnames(sampleinfo_tmp)[grep('CEA', colnames(sampleinfo_tmp))] <- 'CEA'
colnames(sampleinfo_tmp)[grep('CA19-9', colnames(sampleinfo_tmp))] <- 'CA199'
colnames(sampleinfo_tmp)[grep('CA242', colnames(sampleinfo_tmp))] <- 'CA242'

GC_pred_m <- merge(GC_prediction, sampleinfo_tmp, by='Sample')
GC_pred_m <- GC_pred_m[GC_pred_m$source_key == 'test', ]

GC_pred_m$CEA <- as.numeric(GC_pred_m$CEA)
GC_pred_m$CA199 <- as.numeric(GC_pred_m$CA199)
GC_pred_m$CA242 <- as.numeric(GC_pred_m$CA242)

generate_roc_panel <- function(df, neg_condition, pos_condition, title_label) {
  df_neg <- df[neg_condition, ]
  df_neg$Target <- 0

  df_pos <- df[pos_condition, ]
  df_pos$Target <- 1

  df_plot <- rbind(df_neg, df_pos)

  df_plot$Target <- as.factor(df_plot$Target)
  df_plot$final_prob <- as.numeric(df_plot$final_prob)
  df_plot$CEA <- as.numeric(df_plot$CEA)
  df_plot$CA199 <- as.numeric(df_plot$CA199)
  df_plot$CA242 <- as.numeric(df_plot$CA242)

  roc_rbcDNA <- pROC::roc(df_plot$Target, df_plot$final_prob, levels = c(0, 1), percent = TRUE)
  roc_cea <- pROC::roc(df_plot$Target, df_plot$CEA, levels = c(0, 1), percent = TRUE)
  roc_ca199 <- pROC::roc(df_plot$Target, df_plot$CA199, levels = c(0, 1), percent = TRUE)
  roc_ca242 <- pROC::roc(df_plot$Target, df_plot$CA242, levels = c(0, 1), percent = TRUE)
  set.seed(1234)
  roc_random <- pROC::roc(sample(df_plot$Target), df_plot$final_prob, levels = c(0, 1), percent = TRUE)

  get_roc_line_info <- function(roc_obj, label, color) {
    ci_auc <- pROC::ci.auc(roc_obj)
    list(
      roc_df = data.frame(Sp_inv = 100 - roc_obj$specificities, Sens = roc_obj$sensitivities),
      poly_df = data.frame(x = numeric(), y = numeric()),
      auc_text = paste0(label, ": ", round(ci_auc[2], 0), " (", round(ci_auc[1], 0), "-", round(ci_auc[3], 0), ")"),
      color = color,
      fill_color = NA
    )
  }

  random_curve_df <- function(roc_obj) {
    data.frame(Sp_inv = 100 - roc_obj$specificities, Sens = roc_obj$sensitivities)
  }

  color_cea <- ggsci::pal_material("brown")(10)[7]
  color_ca199 <- ggsci::pal_material("brown")(10)[5]
  color_ca242 <- ggsci::pal_material("brown")(10)[4]

  p <- plot_auc_panel(
    list(
      get_roc_line_info(roc_cea, "CEA", color_cea),
      get_roc_line_info(roc_ca199, "CA19-9", color_ca199),
      get_roc_line_info(roc_ca242, "CA242", color_ca242),
      get_roc_curve_info(roc_rbcDNA, "rbcDNA", "#512CA7", rgb(81, 44, 167, 30, maxColorValue = 255))
    ),
    title = title_label,
    label_y = c(34, 28, 22, 16, 10),
    test_text_y = 40,
    test_text = "Test cohort:"
  ) +
    geom_path(data = random_curve_df(roc_random), aes(x = Sp_inv, y = Sens), color = rgb(128, 128, 128, 120, maxColorValue = 255), linewidth = 0.4) +
    annotate("text", x = 100, y = 4, label = "Random Classifiers", color = rgb(128, 128, 128, 180, maxColorValue = 255), size = 6 / .pt, hjust = 1)

  return(p)
}

pA <- generate_roc_panel(GC_pred_m, GC_pred_m$Group == 'Non-GC', GC_pred_m$Group == 'GC' & GC_pred_m$Stage == 'I', "Non-GC vs stage I GC")
pB <- generate_roc_panel(GC_pred_m, GC_pred_m$Group == 'Non-GC', GC_pred_m$Group == 'GC' & GC_pred_m$Stage %in% c('II', 'III'), "Non-GC vs stage II-III GC")
pC <- generate_roc_panel(GC_pred_m, GC_pred_m$Group == 'Non-GC' & GC_pred_m$Atrophic == 'Yes', GC_pred_m$Group == 'GC' & GC_pred_m$Stage == 'I', "CAG vs stage I GC")
pD <- generate_roc_panel(GC_pred_m, GC_pred_m$Group == 'Non-GC' & GC_pred_m$Atrophic == 'Yes', GC_pred_m$Group == 'GC' & GC_pred_m$Stage %in% c('II', 'III'), "CAG vs stage II-III GC")
pE <- generate_roc_panel(GC_pred_m, GC_pred_m$Group == 'Non-GC' & GC_pred_m$IntestinalMetaplasia == 'Yes', GC_pred_m$Group == 'GC' & GC_pred_m$Stage == 'I', "IM vs stage I GC")
pF <- generate_roc_panel(GC_pred_m, GC_pred_m$Group == 'Non-GC' & GC_pred_m$IntestinalMetaplasia == 'Yes', GC_pred_m$Group == 'GC' & GC_pred_m$Stage %in% c('II', 'III'), "IM vs stage II-III GC")

FigS5 <- plot_grid(pA, pC, pE, pB, pD, pF, ncol = 3, nrow = 2, labels = c("A", "C", "", "B", "D", ""),
                    label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 1.02, hjust = 0, vjust = 1.5)
ggsave(file.path(out_dir, 'FigureS5.pdf'), FigS5, width = 8, height = 5.6)
