args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_path <- gsub("~\\+~", " ", script_path)
script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
	library(pROC)
	library(stringr)
	library(ggplot2)
	library(cowplot)
	library(dplyr)
	library(ggpubr)
	library(ggsci)
	library(reportROC)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'or_function.r'), chdir = TRUE)

## 加载基础数据
load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

cutoff_spe90 = Cutoff(0.90, GC_trncv_pred)

# Figure S4A: 特征选择与 AUC (Top n features vs AUC)
auc = read.table('./Human_model/results/3_FeatureReduction/gc_trncv_detail.csv', sep = ',', head = TRUE)
auc = auc[auc$top_feature!=10, ]
auc = unique(auc[(auc$n_pcas <= 60) & (auc$top_feature <= 1000), 1:5])
colnames(auc) = c("type", "auc2", "auc", "topn", "n_pca")
auc$topn = factor(auc$topn)
auc$n_pca = factor(auc$n_pca)

reversed_colors <- rev(pal_jama(alpha = 0.8)(7)[1:6])
pA <- ggplot(auc, aes(x = topn, y = auc * 100)) +
	geom_rect(aes(xmin = 9.5, xmax = 10.5, ymin = 80, ymax = 94), fill = '#F0F0F0', alpha = 0.2) +
	geom_line(data = auc[which(auc$n_pca == 0),], aes(group = n_pca, color = n_pca), linewidth = 1) +
	geom_boxplot(outlier.color = NA) +
	geom_jitter(aes(color = n_pca), shape = 16, position = position_jitter(0.2), size = 1.5) +
	theme_cor + 
	theme(legend.background = element_blank(),
	      legend.position = "inside",
	      legend.position.inside = c(0.2, 0.65),
	      legend.key = element_blank(),
	      legend.text = element_text(size = 5),
	      legend.title = element_text(size = 7),
	      legend.key.size = unit(0.3, "cm"),
	      legend.spacing.y = unit(0.05, "cm"),
		  plot.margin = margin(t = 5, r = 5, b = 17, l = 5, unit = "pt")) +
	scale_color_manual(name = "Parameters: n_pca", values = c("grey", reversed_colors), labels = c('original features', '10', '20', '30', '40', '50', '60')) +
	ylim(80, 94) + labs(title = 'Top n features with the highest discriminative scores', x = 'Number of selected features', y = 'AUC from 5-fold CV\nin the discovery cohort (%)')

# Figure S4B: 5-fold cross validation score
GC_trncv_pred_fold <- read_and_merge(str_c('./Human_model/results/4_Classification/gc_prediction_trncv_fold.csv'), sampleinfo)
GC_trncv_pred_fold$source_key <- factor(GC_trncv_pred_fold$source_key, levels = c(0, 1, 2, 3, 4), labels = c('Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'))
GC_trncv_pred_fold$Group = factor(GC_trncv_pred_fold$Group, levels = c('Non-GC', 'GC'))
pB <- ggplot(data = GC_trncv_pred_fold, aes(x = Group, y = final_prob, fill = Group)) +
	geom_boxplot(outlier.shape = NA, outlier.color = NA, linewidth = 0.3) +
	geom_jitter(width = 0.3, size = 0.1) +
	scale_fill_manual(values = c(ggsci::pal_material("blue-grey", alpha = 0.8)(10)[5], "#9F1A1AFF")) +
	theme_sig2 + theme(
		legend.position = 'bottom',
		legend.title = element_blank(),
		legend.key.size = unit(0.3, "cm"),
		legend.text = element_text(color = "black", size = 8),
		plot.title = element_text(color = "black", size = 6, hjust = 0.5),
		axis.text.x = element_blank(),
		axis.ticks.x = element_blank(),
		axis.title.x = element_blank(),
		strip.placement = "outside",
		strip.text = element_text(color = "black", size = 6, margin = margin(t = 0)),
		strip.switch.pad.grid = unit(0.15, "cm")) + ylim(0, 1) +
	facet_grid(. ~ source_key)+labs(x=NULL, y='rbcDNA predictive scores',title="Discovery cohort, 5-fold cross-validation")
	
# Figure S4C: Randomized labels model
rnd_label = read.table('./Human_model/modelData/sampleinfo.gc_shuffled.txt', sep = '\t', head = TRUE)
colnames(rnd_label)[1] = 'Sample'
rnd_label[which(rnd_label$target == 0), 'Group'] = 'Non-GC'
rnd_label[which(rnd_label$target == 1), 'Group'] = 'GC'

GC_trncv_rnd_pred <- read_and_merge('./Human_model/results/4_Classification/gc_rnd_prediction_trncv.csv', rnd_label)
GC_test1_rnd_pred <- read_and_merge('./Human_model/results/4_Classification/gc_rnd_prediction_test.csv', sampleinfo)
roc_rnd_trn <- pROC::roc(GC_trncv_rnd_pred$Target, GC_trncv_rnd_pred$final_prob, levels = c(0, 1), percent = TRUE)
roc_rnd_test <- pROC::roc(GC_test1_rnd_pred$Target, GC_test1_rnd_pred$final_prob, levels = c(0, 1), percent = TRUE)
set.seed(1234)
roc_rnd_trn_random <- pROC::roc(sample(GC_trncv_rnd_pred$Target), GC_trncv_rnd_pred$final_prob, levels = c(0, 1), percent = TRUE)
roc_rnd_test_random <- pROC::roc(sample(GC_test1_rnd_pred$Target), GC_test1_rnd_pred$final_prob, levels = c(0, 1), percent = TRUE)
random_curve_df <- function(roc_obj) {
  data.frame(Sp_inv = 100 - roc_obj$specificities, Sens = roc_obj$sensitivities)
}
gC <- plot_auc_panel(
  list(
    get_roc_curve_info(roc_rnd_trn, "Discovery cohort", "#A50F15", rgb(165, 15, 21, 20, maxColorValue = 255)),
    get_roc_curve_info(roc_rnd_test, "Test cohort", "#FCBBA1", rgb(253, 176, 99, 20, maxColorValue = 255))
  ),
  title = NULL, label_y = c(32, 20, 12)) +
  geom_path(data = random_curve_df(roc_rnd_trn_random), aes(x = Sp_inv, y = Sens), color = rgb(128, 128, 128, 120, maxColorValue = 255), linewidth = 0.4) +
  geom_path(data = random_curve_df(roc_rnd_test_random), aes(x = Sp_inv, y = Sens), color = rgb(128, 128, 128, 120, maxColorValue = 255), linewidth = 0.4) +
  annotate("text", x = 100, y = 4, label = "Random Classifiers", color = rgb(128, 128, 128, 180, maxColorValue = 255), size = 6 / .pt, hjust = 1)
gC <- gC + labs(title = "Feature selection and Model retraining with\nrandomized labels in discovery cohort") +
		   theme(plot.title = element_text(hjust = 0.5, vjust = 1, size = 6), plot.margin = margin(5, 5, 15, 5, "pt"))

# Figure S4D & E: Other Cancers
test_otherca = read.table('./Human_model/results/4_Classification/gc_prediction_otherca.csv', sep = ',', head = TRUE)
colnames(test_otherca)[1] = 'Sample'
colnames(test_otherca)[grep('X0', colnames(test_otherca))] = 'final_prob'
test_otherca$Group = ''
bc_lab <- str_c('BC (n=', sum(grepl('^BC', test_otherca$Sample)), ')')
crc_lab <- str_c('CRC (n=', sum(grepl('^CRC', test_otherca$Sample)), ')')
lc_lab <- str_c('LC (n=', sum(grepl('^LC', test_otherca$Sample)), ')')
tc_lab <- str_c('TC (n=', sum(grepl('^TC', test_otherca$Sample)), ')')
other_ca_labs <- c(bc_lab, crc_lab, lc_lab, tc_lab)
test_otherca[which(grepl('^BC', test_otherca$Sample)), 'Group'] = bc_lab
test_otherca[which(grepl('^CRC', test_otherca$Sample)), 'Group'] = crc_lab
test_otherca[which(grepl('^LC', test_otherca$Sample)), 'Group'] = lc_lab
test_otherca[which(grepl('^TC', test_otherca$Sample)), 'Group'] = tc_lab

GC_test1_pred2 = merge(GC_test1_pred, sampleinfo[, c('Sample', 'Group')], by = 'Sample')
test_otherca = as.data.frame(rbind(GC_test1_pred2[, c('Sample', 'final_prob', 'lr', 'cb', 'source_key', 'Group')], test_otherca))
test_otherca$Group = factor(test_otherca$Group, levels = c('Non-GC', 'GC', other_ca_labs))

pD <- ggplot(data = test_otherca, aes(x = Group, y = final_prob, fill = Group)) +
	geom_boxplot(outlier.shape = NA, outlier.color = NA, linewidth = 0.3) +
	geom_jitter(width = 0.2, size = 0.5) +
	geom_hline(yintercept = cutoff_spe90, color = 'red4', linetype = 'dashed', linewidth = 0.5) +
	geom_vline(xintercept = 2.5, color = 'grey40', linetype = 'dotted', linewidth = 0.5) +
	scale_fill_manual(values = c(ggsci::pal_material("blue-grey")(10)[5], "#9F1A1AFF",
	                             pal_npg(alpha = 0.6)(10)[2], pal_npg(alpha = 0.6)(10)[3], pal_npg(alpha = 0.6)(10)[4], pal_npg(alpha = 0.6)(10)[6])) +
	theme_sig2 + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1), plot.margin = margin(t = 5, r = 60, b = 5, l = 5, unit = "pt")) +
	stat_compare_means(comparisons = lapply(other_ca_labs, function(x) c('GC', x)),
									 label = 'p.signif', method = 'wilcox.test', label.x.npc = 'center', size = 6 / .pt, hjust = 0.5, lineheight = 0.65) +
	coord_cartesian(ylim = c(0, 1.41), clip = "off") +
	annotate("text", x = Inf, y = cutoff_spe90, label = "Cutoff:\n90% specificity\nin discovery cohort", hjust = 0, vjust = 0.5, size = 5 / .pt, lineheight = 0.9) +
	labs(x = 'Test cohort\n(n = 109)', y = 'rbcDNA predictive scores')

pred = test_otherca
pred$binary_c = 0; pred[pred$final_prob >= cutoff_spe90, 'binary_c'] = 1
pred$Target = 1; pred[grep('Non-GC', pred$Group), 'Target'] = 0
calc_roc <- function(g) {
	p <- pred[pred$Group %in% c('Non-GC', g),]
	as.numeric(reportROC(gold = p$Target, predictor.binary = p$binary_c, plot = F, important = "se")[c("SEN", "SEN.low", "SEN.up")])
}
all_other <- as.data.frame(rbind(
	c(calc_roc(bc_lab), bc_lab), c(calc_roc(crc_lab), crc_lab),
	c(calc_roc(lc_lab), lc_lab), c(calc_roc(tc_lab), tc_lab)
))
colnames(all_other) = c('SEN', 'SEN.low', 'SEN.up', 'Var1')
all_other$SEN = as.numeric(all_other$SEN); all_other$SEN.low = as.numeric(all_other$SEN.low); all_other$SEN.up = as.numeric(all_other$SEN.up)
all_other$perc = str_c(round(all_other$SEN * 100), "%")

pE <- ggplot(all_other, aes(x = Var1, y = SEN * 100, fill = Var1)) +
	geom_bar(stat = "identity", color = "black", position = position_dodge(), alpha = 0.8) +
	geom_errorbar(aes(ymin = SEN.low * 100, ymax = SEN.up * 100), width = .2, position = position_dodge(.9)) +
	labs(x=NULL, y='Proportion classified as positive (%)') +
	theme_bar + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1), plot.margin = margin(t = 5, r = 5, b = 5, l = 5, unit = "pt")) +
	geom_text(aes(label = perc), vjust = 3, color = "white", position = position_dodge(.9), size = 6 / .pt) +
	scale_fill_manual(values = c(pal_npg()(10)[2], pal_npg()(10)[3], pal_npg()(10)[4], pal_npg()(10)[6])) + ylim(0, 105)


pred_GC_otherCA = pred[which(pred$Group!='Non-GC'), ]
pred_GC_otherCA[which(pred_GC_otherCA$Group=='GC'), 'Target'] = 0

roc2_GC = pROC::roc(pred_GC_otherCA$Target,pred_GC_otherCA$final_prob, percent = TRUE)

test_curves <- list(
  get_roc_curve_info(roc2_GC, "other CA vs GC", pal_material("red")(10)[10], pal_material("red", alpha=0.2)(10)[10])
)
pF <- plot_auc_panel(test_curves, title = NULL, label_y = c(22, 8, 12), test_text_y = 4, test_text = NULL)



row1 <- plot_grid(pA, pB, gC, ncol = 3, labels = c("A", "B", "C"), rel_widths = c(1.1, 1.1, 0.9), 
					label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 0.98, hjust = 0, vjust = 0.5)
row1 <- row1 + theme(plot.margin = margin(t = 12, unit = "pt"))

row2 <- plot_grid(pD, pE, pF, ncol = 3, labels = c("D", "E", "F"), align = "h", axis = "tb", rel_widths = c(1.3, 0.9, 0.9), 
					label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 1, hjust = 0, vjust = 0.5)

FigS4 <- plot_grid(row1, row2, ncol = 1, rel_heights = c(1, 1))
ggsave(file.path(out_dir, 'FigureS4.pdf'), FigS4, width = 8, height = 5.32)#, device = cairo_pdf)
