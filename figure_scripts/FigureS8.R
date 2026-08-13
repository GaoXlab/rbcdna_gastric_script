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
  library(ggplot2)
  library(cowplot)
  library(dplyr)
  library(ggpubr)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'or_function.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

GC_trncv_pred$source_key  = 'Discovery\ncohort'
GC_test1_pred$source_key  = 'Test\ncohort'
GC_ind1_pred$source_key  = 'Independent\nvalidation sets'
GC_ind2_pred$source_key  = 'Independent\nvalidation sets'
GC_ind3_pred$source_key  = 'Independent\nvalidation sets'
GC_ind_pred <- as.data.frame(rbind(GC_trncv_pred, GC_test1_pred, GC_ind1_pred, GC_ind2_pred, GC_ind3_pred))
GC_prediction <- as.data.frame(GC_ind_pred)

## cutoff determined
cutoff_spe90 <- Cutoff(0.90, GC_trncv_pred)

## GC model performance
needed_cols <- c(
  'Sample', 'Group',
  'RBC (×10^12/L)', 'HGB (g/L)', 'WBC (×10^9/L)', 'PLT (×10^9/L)',
  'CEA (ng/mL)', 'CA19-9 (U/mL)', 'CA242 (U/mL)',
  'RDW (%)', 'MCV (fL)', 'MCHC (g/L)',
  'NLR', 'PLR', 'SII',
  'IL-6 (pg/mL)', 'CRP (mg/L)'
)
missing_cols <- setdiff(needed_cols, colnames(sampleinfo))
if (length(missing_cols) > 0) {
  stop("Missing required columns in sampleinfo: ", paste(missing_cols, collapse = ", "))
}
sampleinfo_tmp <- sampleinfo[, needed_cols]

sampleinfo_col_map <- c(
  'RBC (×10^12/L)' = 'RBC',
  'HGB (g/L)' = 'Hb',
  'WBC (×10^9/L)' = 'WBC',
  'PLT (×10^9/L)' = 'PLT',
  'CEA (ng/mL)' = 'CEA',
  'CA19-9 (U/mL)' = 'CA199',
  'CA242 (U/mL)' = 'CA242',
  'RDW (%)' = 'RDW',
  'MCV (fL)' = 'MCV',
  'MCHC (g/L)' = 'MCHC',
  'IL-6 (pg/mL)' = 'IL.6',
  'CRP (mg/L)' = 'CRP'
)
colnames(sampleinfo_tmp) <- ifelse(
  colnames(sampleinfo_tmp) %in% names(sampleinfo_col_map),
  sampleinfo_col_map[colnames(sampleinfo_tmp)],
  colnames(sampleinfo_tmp)
)


GC_pred_m <- merge(GC_prediction, sampleinfo_tmp, by='Sample')
GC_pred_m$Group <- factor(GC_pred_m$Group, levels=c('Non-GC','GC'))
GC_pred_m$source_key <- factor(GC_pred_m$source_key, levels=(c('Discovery\ncohort', 'Test\ncohort', 'Independent\nvalidation sets')))

color_mapping <- c("Non-GC" = ggsci::pal_material("blue-grey", alpha = 0.7)(10)[5], "GC" = "#9F1A1ACC")

pA_logCEA <- plot_tm_box("CEA", 5, "Log-transformed CEA (ng/mL)")
pA_logCA199 <- plot_tm_box("CA199", 37, "Log-transformed CA 19-9 (U/mL)")
pA_logCA242 <- plot_tm_box("CA242", 20, "Log-transformed CA 242 (U/mL)")
pB_logCRP <- plot_tm_box("CRP", 10, "Log-transformed CRP (mg/L)")
pB_logIL6 <- plot_tm_box("IL.6", 7, "Log-transformed IL-6 (pg/mL)")

## Panel A-B: Tumor markers and their correlation heatmap
cor_vars <- c("RBC", "Hb", "MCV", "MCHC", "RDW", "WBC", "PLT", "SII", "NLR", "PLR", "CEA", "CA199", "CA242")
group_levels <- c("Non-GC", "GC")
source_levels <- c("Discovery\ncohort","Test\ncohort","Independent\nvalidation sets")
cor_group_labels <- c("Non-GC_Discovery\ncohort" = "Non-GC\nDiscovery", "GC_Discovery\ncohort" = "GC\nDiscovery",
                      "Non-GC_Test\ncohort" = "Non-GC\nTest","GC_Test\ncohort" = "GC\nTest",
                      "Non-GC_Independent\nvalidation sets" = "Non-GC\nIndependent","GC_Independent\nvalidation sets" = "GC\nIndependent")
cor_by_group <- make_cor_by_group(GC_pred_m, cor_vars)

tumor_marker_vars <- c("CEA", "CA199", "CA242")
tumor_cor_df <- format_cor_df(cor_by_group, tumor_marker_vars, labels = rev(cor_group_labels)) %>% filter(variable %in% tumor_marker_vars)

pC_cor <- plot_cor_heatmap(
  tumor_cor_df, rev(tumor_marker_vars),
  "Spearman correlations between\nrbcDNA predictive scores and tumor markers",
  plot_margin = margin(t = 10, r = 5, b = 20, l = 5)
) + coord_flip()

row1 <- plot_grid(pA_logCEA, pA_logCA199, pA_logCA242, pC_cor, ncol = 4, rel_widths = c(1, 1, 1, 1.5),
        labels = c("A", "", "", "B"), label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 1, hjust = 0, vjust = 1)

## Panel B: Blood count correlations
cbc_vars <- c("RBC", "Hb", "MCV", "MCHC", "RDW", "WBC", "PLT", "SII", "NLR", "PLR")
cor_group_labels <- c("Non-GC_Discovery\ncohort" = "Discovery\nNon-GC", "GC_Discovery\ncohort" = "Discovery\nGC",
                      "Non-GC_Test\ncohort" = "Test\nNon-GC","GC_Test\ncohort" = "Test\nGC",
                      "Non-GC_Independent\nvalidation sets" = "Independent\nNon-GC","GC_Independent\nvalidation sets" = "Independent\nGC")
cbc_cor_df <- format_cor_df(cor_by_group, cbc_vars, labels = cor_group_labels) %>% filter(variable %in% cbc_vars)

pD_cor <- plot_cor_heatmap(cbc_cor_df, cbc_vars, "Spearman correlations between rbcDNA predictive scores and hematologic/inflammatory indices", plot_margin = margin(t = 5, r = 5, b = 20, l = 10))

row2 <- plot_grid(pD_cor, ncol = 1, labels = c("C"),
  label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 1, hjust = 0, vjust = 1)

## Panel D-E: CRP and IL-6 correlations
GC_pred_m$CRP_label = ''
GC_pred_m[which(GC_pred_m$CRP >= 10), 'CRP_label'] = '≥ 10 mg/L'
GC_pred_m[which(GC_pred_m$CRP < 10), 'CRP_label'] = '< 10 mg/L'
GC_pred_m$source_key <- factor(GC_pred_m$source_key, levels=(c('Discovery\ncohort', 'Test\ncohort', 'Independent\nvalidation sets')))
pD_1 <- ggplot(data = GC_pred_m[!is.na(GC_pred_m$CRP), ], aes(x =CRP_label, y = final_prob)) +
  geom_boxplot(aes(fill=Group), outlier.colour = NA, width = 0.5) +
  geom_jitter(shape = 16, position = position_jitter(0.2), size = 0.3, color = "darkgrey") +
  stat_compare_means(aes(label = paste0(after_stat(method), ", ", after_stat(p.signif))), method = "wilcox.test", label.x.npc = 'center', label.y.npc = 0.99, size= 6 / .pt, hjust = 0.5, vjust = 1) +
   facet_grid(source_key ~ Group) +
  theme_sig2 +
  scale_fill_manual(values = color_mapping) + 
  labs(x = "CRP (mg/L)", y = "rbcDNA predictive scores") +
  ylim(0, 1.05) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), strip.background = element_blank(), panel.spacing = unit(0.1, "lines"))

GC_pred_m$IL6_label = ''
GC_pred_m[which(GC_pred_m$IL.6 >= 7), 'IL6_label'] = '≥ 7 pg/mL'
GC_pred_m[which(GC_pred_m$IL.6 < 7), 'IL6_label'] = '< 7 pg/mL'
GC_pred_m$source_key <- factor(GC_pred_m$source_key, levels=(c('Discovery\ncohort', 'Test\ncohort', 'Independent\nvalidation sets')))
pE_1 <- ggplot(data = GC_pred_m[!is.na(GC_pred_m$IL.6), ], aes(x =IL6_label, y = final_prob)) +
  geom_boxplot(aes(fill=Group), outlier.colour = NA, width = 0.5) +
  geom_jitter(shape = 16, position = position_jitter(0.2), size = 0.3, color = "darkgrey") +
  stat_compare_means(aes(label = paste0(after_stat(method), ", ", after_stat(p.signif))), method = "wilcox.test", label.x.npc = 'center', label.y.npc = 0.99, size= 6 / .pt, hjust = 0.5, vjust = 1) +
  facet_grid(source_key ~ Group) +
  scale_fill_manual(values = color_mapping) + 
  theme_sig2 +
  labs(x = "IL-6 (pg/mL)", y = "rbcDNA predictive scores") +
  ylim(0, 1.05) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), strip.background = element_blank(), panel.spacing = unit(0.1, "lines"))

row3_plots <- cowplot::align_plots(pB_logCRP, pD_1, pB_logIL6, pE_1, align = "hv", axis = "tb")
row3 = plot_grid(plotlist = row3_plots, ncol=4, rel_widths= c(0.6, 0.8, 0.6, 0.8), labels = c("D"),
       label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 1, hjust = 0, vjust = 1)

FigS8 <- plot_grid(row1, row2, row3, ncol = 1, rel_heights = c(1, 0.7, 1))

ggsave(file.path(out_dir, "FigureS8.pdf"), FigS8, width = 8, height = 8)
