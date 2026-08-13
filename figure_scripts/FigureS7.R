args <- commandArgs(trailingOnly = TRUE)
working_dir <- if (length(args) >= 1) gsub("~\\+~", " ", args[1]) else getwd()
setwd(working_dir)

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
  library(ggpubr)
})

script_path <- sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])
script_path <- gsub("~\\+~", " ", script_path)
script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))
if (!file.exists(file.path(script_dir, "plot_function.r"))) {
  script_dir <- file.path(working_dir, "figure_scripts")
}
source(file.path(script_dir, "plot_function.r"), chdir = TRUE)
font_normal <- 8

load("./Figures/prediction.RData")
load("./Figures/sampleinfo.RData")

## Data preparation
GC_prediction <- as.data.frame(rbind(GC_trncv_pred, GC_test1_pred))
cutoff_spe90 <- Cutoff(0.90, GC_trncv_pred)

cols <- c(
  "Sample", "Group", "Age", "Gender", "Smoking status", "Alcohol status",
  "Helicobacter pylori"
)

sampleinfo_tmp <- sampleinfo[, cols]
colnames(sampleinfo_tmp) <- c(
  "Sample", "Group", "Age", "Gender", "Smoking.state", "Alcohol.state", "HP"
)

GC_pred_m <- merge(GC_prediction, sampleinfo_tmp, by = "Sample")
GC_pred_m$Group <- factor(GC_pred_m$Group, levels = c("Non-GC", "GC"))
GC_pred_m$source_key <- factor(GC_pred_m$source_key, levels = c("TRAIN_CV", "test"), labels = c("Discovery", "Test"))
GC_pred_m$Smoking.state <- factor(gsub(" smoker", "", GC_pred_m$Smoking.state), levels = c("Current", "Prior", "Never", "No record"))
GC_pred_m$Alcohol.state <- factor(gsub(" consumed", "", GC_pred_m$Alcohol.state), levels = c("Current", "Prior", "Never", "No record"))

## Plot settings
color_mapping <- c("Discovery" = "#962C28", "Test" = "#8791B2")
source_levels <- names(color_mapping)

## Panel A: Sex
pA <- ggplot(data = GC_pred_m, aes(x = Gender, y = final_prob)) +
  geom_boxplot(aes(color = source_key), outlier.shape = NA, fill = "white", width = 0.6, position = position_dodge(width = 0.75)) +
  geom_point(aes(color = source_key), position = position_jitterdodge(jitter.width = 0.5, dodge.width = 0.75), size = 0.5) +
  scale_color_manual(values = color_mapping) +
  stat_compare_means(aes(label = paste0(after_stat(method), ", ", after_stat(p.signif))), method = "wilcox.test", label.x.npc = "center", label.y.npc = 1, size = 6 / .pt) +
  facet_grid(. ~ Group) +
  theme_sig2 +
  labs(x = "Sex", y = "rbcDNA predictive scores")

## Panel B: Age
pB_Age_nonGC <- make_age_cor(GC_pred_m[GC_pred_m$Target == 0, ], "top", TRUE, "Non-GC")
pB_Age_GC <- make_age_cor(GC_pred_m[GC_pred_m$Target == 1, ], "bottom", FALSE, "GC")
pB <- plot_grid(pB_Age_nonGC, pB_Age_GC, ncol = 2, rel_widths = c(1, 1))

## Panel C: Alcohol and smoking history
pC_1 <- ggplot(data = GC_pred_m[!is.na(GC_pred_m$Alcohol.state), ], aes(x = Alcohol.state, y = final_prob)) +
  geom_boxplot(outlier.colour = NA, width = 0.5) +
  geom_jitter(shape = 16, position = position_jitter(0.2), size = 0.3, color = "darkgrey") +
  stat_compare_means(aes(label = paste0(after_stat(method), ", ", after_stat(p.signif))), method = "kruskal.test", label.x.npc = "center", label.y = 0.99, size = 5 / .pt, hjust = 0.5, lineheight = 0.65) +
  facet_grid(. ~ Group) +
  theme_sig2 +
  labs(x = "History of alcohol consumption", y = "rbcDNA predictive scores") +
  ylim(0, 1.05) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), strip.background = element_blank())

pC_2 <- ggplot(data = GC_pred_m[!is.na(GC_pred_m$Smoking.state), ], aes(x = Smoking.state, y = final_prob)) +
  geom_boxplot(outlier.colour = NA, width = 0.5) +
  geom_jitter(shape = 16, position = position_jitter(0.2), size = 0.3, color = "darkgrey") +
  stat_compare_means(aes(label = paste0(after_stat(method), ", ", after_stat(p.signif))), method = "kruskal.test", label.x.npc = "center", label.y = 0.99, size = 5 / .pt, hjust = 0.5, lineheight = 0.65) +
  facet_grid(. ~ Group) +
  theme_sig2 +
  labs(x = "History of smoking", y = "rbcDNA predictive scores") +
  ylim(0, 1.05) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), strip.background = element_blank(), axis.title.y = element_blank())

## Panel D: H. pylori sensitivity and specificity
df_hp <- GC_pred_m[GC_pred_m$HP %in% c("Yes", "No"), ]
df_hp$HP <- factor(df_hp$HP, levels = c("No", "Yes"))

hp_sens <- make_hp_perf_data(df_hp, "Sensitivity")
hp_spec <- make_hp_perf_data(df_hp, "Specificity")

pD_Sens <- ggplot(hp_sens, aes(x = HP, y = Rate, fill = Source)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.9, alpha = 0.9) +
  geom_errorbar(aes(ymin = Low, ymax = Up), position = position_dodge(width = 0.8), width = 0.2) +
  geom_text(aes(label = Label), position = position_dodge(width = 0.8), vjust = 1.2, color = "white", size = 6/.pt) +
  scale_fill_manual(values = color_mapping) +
  theme_bar +
  ylab("Sensitivity\nat 90% specificity (%)") +
  xlab("H. pylori infection status") +
  ylim(0, 102) +
  theme(plot.margin = margin(t = 14, b = 20, unit = "pt"))

pD_Spec <- ggplot(hp_spec, aes(x = HP, y = Rate, fill = Source)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.9, alpha = 0.9) +
  geom_errorbar(aes(ymin = Low, ymax = Up), position = position_dodge(width = 0.8), width = 0.2) +
  geom_text(aes(label = Label), position = position_dodge(width = 0.8), vjust = 1.2, color = "white", size = 6/.pt) +
  scale_fill_manual(values = color_mapping) +
  theme_bar +
  ylab("Specificity (%)") +
  xlab("H. pylori infection status") +
  ylim(0, 102) +
  theme(plot.margin = margin(t = 14, b = 20, unit = "pt"))

pD <- plot_grid(pD_Sens, pD_Spec, ncol = 2)

## Assemble figure
row1 <- plot_grid(
  pA, pB, ncol = 2, rel_widths = c(1, 1.3), labels = c("A", "B"),
  label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 1, hjust = 0, vjust = 1)

row2 <- plot_grid(
  pC_1, pC_2, pD, ncol = 3, rel_widths = c(1, 1, 1.5), labels = c("C", "", "D"),
  label_size = 12, label_fontface = "bold", 
  label_x = 0.01, label_y = 1, hjust = 0, vjust = 1)

legend_plot <- ggplot(data = GC_pred_m, aes(x = Age, y = final_prob, color = source_key)) +
  geom_point() +
  scale_color_manual(values = color_mapping) +
  theme_sig2 +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.direction = "horizontal",
    legend.background = element_rect(fill = NA, color = NA),
    legend.box.background = element_rect(fill = NA, color = NA)
  )

shared_legend <- get_legend(legend_plot)

FigS7 <- plot_grid(row1, shared_legend, row2, ncol = 1, rel_heights = c(1, 0.1, 1.1))

ggsave(file.path(out_dir, "FigureS7.pdf"), FigS7, width = 8, height = 5)
