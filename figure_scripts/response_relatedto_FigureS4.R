args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
	library(stringr)
	library(ggplot2)
	library(grid)
	library(cowplot)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)

# Figure S4A: 特征选择与 AUC (Top n features vs AUC)
auc = read.table('./Human_model/results/3_FeatureReduction/gc_trncv_detail.csv', sep = ',', head = TRUE)
auc = unique(auc[(auc$n_pcas <= 60) & (auc$top_feature <= 1000), 1:7])
colnames(auc) = c("type", "auc2", "auc", "topn", "n_pca", "cb_AUC", "lr_AUC")
auc$topn = as.numeric(auc$topn)
auc$n_pca = as.numeric(auc$n_pca)

auc_2 = read.table('./Human_model/results/3_FeatureReduction/gc_trncv_ori_topn_detail.csv', sep=',', head=TRUE)[, 1:7]
colnames(auc_2) = c("type", "auc2", "auc", "topn", "n_pca", "cb_AUC", "lr_AUC")
auc_2 = auc_2[auc_2$topn %in% c(10, seq(100, 1000, 100)), ]
auc_2 = auc_2[which(auc_2$topn %in% auc[which(auc$n_pca == 0),'topn']), ]
auc_2$topn = as.numeric(auc_2$topn)
auc_2$n_pca = as.numeric(auc_2$n_pca)

feature_score = read.table('./Human_model/gc/all.gc.bed.out', head=FALSE, sep='\t')
feature_score$X = str_c('chr', feature_score$V1, ':', feature_score$V2, '-', feature_score$V3)
feature_score$score = feature_score$V4/344

contri = read.table('./Human_model/results/3_FeatureReduction/gc_loadings_cumsum.csv', sep=',', head=TRUE)
contri$rank_2 = 1:nrow(contri)

feature_score_top1000 = merge(feature_score[, c('X', 'score')], contri[, c('X','contribution_percent','cumulative_contribution_percent')], by='X')
feature_score_top1000 = feature_score_top1000[order(feature_score_top1000$contribution_percent, decreasing = TRUE), ]
feature_score_top1000$rank_2 = 1:nrow(feature_score_top1000)
x_lim <- c(0, max(feature_score_top1000$rank_2, na.rm = TRUE))
feature_score_top1000$contribution_percent_pct <- feature_score_top1000$contribution_percent * 100
point_size_lim <- range(feature_score_top1000$score, na.rm = TRUE)
feature_score_top1000$point_size <- 0.2 + (feature_score_top1000$score - point_size_lim[1]) /
  diff(point_size_lim) * (1.6 - 0.2)
pA_1 =  ggplot(feature_score_top1000, aes(x = rank_2)) +
  geom_point(aes(y = contribution_percent_pct, color = score, size = point_size)) +
  scale_x_continuous(limits = x_lim) +
  scale_y_continuous(name = "Contribution (%)") +
  scale_size_identity(guide = "none") +
  scale_color_continuous(name = "Features with the\nhighest discriminative score",
                         guide = guide_colorbar(direction = "horizontal",
                                                title.position = "top",
                                                title.hjust = 0.5,
                                                barheight = grid::unit(0.25, "cm"),
                                                barwidth = grid::unit(1.6, "cm"))) +
  theme_bw()+ theme_cor +
  labs(x = 'Number of selected features') +
  theme(legend.position = "inside",
        legend.position.inside = c(0.98, 0.98),
        legend.justification = c(1, 1),
        legend.direction = "horizontal",
        legend.background = element_blank(),
        legend.text = element_text(size = 6),
        legend.title = element_text(size = 6),
        plot.margin = margin(t = 5, r = 5, b = 17, l = 5, unit = "pt"))
auc_compare = rbind(
	data.frame(topn = auc[which(auc$n_pca == 0), "topn"], AUC = auc[which(auc$n_pca == 0), "auc"] * 100, model = "Combined"),
	data.frame(topn = auc[which(auc$n_pca == 0), "topn"], AUC = auc[which(auc$n_pca == 0), "cb_AUC"] * 100, model = "CatBoost"),
	data.frame(topn = auc[which(auc$n_pca == 0), "topn"], AUC = auc[which(auc$n_pca == 0), "lr_AUC"] * 100, model = "Logistic regression")
)
auc_compare_labels <- tapply(auc_compare$AUC, auc_compare$model, function(x) {
	sprintf("%.0f-%.0f%% AUC", min(x, na.rm = TRUE), max(x, na.rm = TRUE))
})
auc_compare_labels <- setNames(paste0(names(auc_compare_labels), ": ", auc_compare_labels),
                               names(auc_compare_labels))
auc_compare_labels <- auc_compare_labels[c("Combined", "CatBoost", "Logistic regression")]
pA <- ggplot(auc_compare, aes(x = topn, y = AUC, color = model, group = model)) +
	geom_line(linewidth = 1) +
	geom_point(shape = 16, size = 1.5) +
	theme_cor + 
	theme(legend.background = element_blank(),
	      legend.position = "inside",
	      legend.position.inside = c(0.98, 0.02),
	      legend.justification = c(1, 0),
	      legend.key = element_blank(),
	      legend.text = element_text(size = 5),
	      legend.title = element_blank(),
	      legend.key.size = unit(0.3, "cm"),
	      legend.spacing.y = unit(0.05, "cm"),
		  plot.margin = margin(t = 5, r = 5, b = 17, l = 5, unit = "pt")) +
	scale_color_manual(name = "Model", values = c("Combined" = "#374E55", "CatBoost" = "#DF8F44", "Logistic regression" = "#00A1D5"),
	                   breaks = c("Combined", "CatBoost", "Logistic regression"),
	                   labels = auc_compare_labels) +
	ylim(50, 88) + labs(title = 'Top n features ranked by the highest discriminative scores', x = 'Number of selected features', y = 'AUC from 5-fold CV\nin the discovery cohort (%)')

auc_2_compare = rbind(
	data.frame(topn = auc_2[which(auc_2$n_pca == 0), "topn"], AUC = auc_2[which(auc_2$n_pca == 0), "auc"] * 100, model = "Combined"),
	data.frame(topn = auc_2[which(auc_2$n_pca == 0), "topn"], AUC = auc_2[which(auc_2$n_pca == 0), "cb_AUC"] * 100, model = "CatBoost"),
	data.frame(topn = auc_2[which(auc_2$n_pca == 0), "topn"], AUC = auc_2[which(auc_2$n_pca == 0), "lr_AUC"] * 100, model = "Logistic regression")
)
auc_2_compare_labels <- tapply(auc_2_compare$AUC, auc_2_compare$model, function(x) {
	sprintf("%.0f-%.0f%% AUC", min(x, na.rm = TRUE), max(x, na.rm = TRUE))
})
auc_2_compare_labels <- setNames(paste0(names(auc_2_compare_labels), ": ", auc_2_compare_labels),
                                 names(auc_2_compare_labels))
auc_2_compare_labels <- auc_2_compare_labels[c("Combined", "CatBoost", "Logistic regression")]
pB <- ggplot(auc_2_compare, aes(x = topn, y = AUC, color = model, group = model)) +
	geom_line(linewidth = 1) +
	geom_point(shape = 16, size = 1.5) +
	theme_cor + 
	theme(legend.background = element_blank(),
	      legend.position = "inside",
	      legend.position.inside = c(0.98, 0.02),
	      legend.justification = c(1, 0),
	      legend.key = element_blank(),
	      legend.text = element_text(size = 5),
	      legend.title = element_blank(),
	      legend.key.size = unit(0.3, "cm"),
	      legend.spacing.y = unit(0.05, "cm"),
		  plot.margin = margin(t = 5, r = 5, b = 17, l = 5, unit = "pt")) +
	scale_color_manual(name = "Model", values = c("Combined" = "#374E55", "CatBoost" = "#DF8F44", "Logistic regression" = "#00A1D5"),
	                   breaks = c("Combined", "CatBoost", "Logistic regression"),
	                   labels = auc_2_compare_labels) +
	ylim(50, 88) + labs(title = 'Top n features ranked by contribution (%)', x = 'Number of selected features', y = 'AUC from 5-fold CV\nin the discovery cohort (%)')
g = plot_grid(pA_1, pA, pB, ncol=3, rel_widths=c(0.5,0.5,0.5))
ggsave(file.path(out_dir, "response_figureS4.pdf"), g, width = 8, height = 2.5)
