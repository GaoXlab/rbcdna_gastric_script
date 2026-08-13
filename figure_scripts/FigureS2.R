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
	library(stringr)
	library(cowplot)
	library(ggplot2)
	library(dplyr)
})

mytheme <- theme_classic(base_size = 6) +
	theme(
		axis.text.x = element_blank(),
		axis.text.y = element_text(size = 5),
		axis.ticks.x = element_blank(),
		strip.text.y = element_blank(),
		strip.text.x = element_text(size = 5),
		strip.background.x = element_blank(),
		strip.placement = "outside",
		axis.title.x = element_blank(),
		legend.position = "none",
		panel.grid.major = element_blank(),
		panel.grid.minor = element_blank()
	)

theme_sig <- theme_classic() +
	theme(
		legend.position = 'none',
		axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
		axis.text = element_text(color = "black", size = 6),
		axis.title = element_text(color = "black", size = 8),
		axis.title.x = element_blank(),
		strip.text = element_text(color = "black", size = 6),
		axis.line = element_line(linewidth = 0.4),
		axis.ticks.length = unit(0.1, "cm"),
		strip.background = element_blank()
	)

## 加载统一数据源
load('./Figures/r_g_5controls_smooth.RData')
load('./Figures/trn_HD_smooth_anno.RData')

# ==============================================================================
# Figure S2 C/D: mt proportion (线粒体比例分布)
# ==============================================================================
read_mt_log <- function(filepath) {
	df <- read.table(filepath, sep = '\t', header = TRUE)
	colnames(df)[1] <- 'Sample'
	df$Sample <- gsub('.nodup.q30.bam', '', sapply(strsplit(as.character(df$Sample), "/"), tail, n = 1))
	df$perc_mt = as.numeric(df$mt) / as.numeric(df$total_reads)
	return(na.omit(df))
}

all_mt_data <- read_mt_log('./Figures/TotalSample_MT.noalt.log')
all_mt_data$SampleType <- gsub('[0-9]+$', '', all_mt_data$Sample)
all_mt_data$value_mt <- all_mt_data$perc_mt * 100

df_controls <- all_mt_data[all_mt_data$SampleType %in% c('CRLR', 'CRLG'),]
if (nrow(df_controls) > 0) {
	df_controls$Group <- 'Controls'
	df_controls$SampleType[df_controls$SampleType == 'CRLR'] <- 'rbcDNA'
	df_controls$SampleType[df_controls$SampleType == 'CRLG'] <- 'Leukocyte DNA'
	df_controls$SampleType <- factor(df_controls$SampleType, levels = c('rbcDNA', 'Leukocyte DNA'))
}

# 2. 尝试提取白细胞梯度掺入实验组 (增加自适应判断，防止数据缺失时报错)
info_spike <- read.table('./Figures/contamination_exp.sampleinfo.csv', sep = '\t', head = TRUE)
df_spike <- all_mt_data[all_mt_data$SampleType %in% c('GLGHD', 'GLMHD'),]
df_spike <- merge(df_spike[, setdiff(colnames(df_spike), 'SampleType')], info_spike[, c('Sample', 'SampleType')], by = 'Sample')
has_spike <- nrow(df_spike) > 0
if (has_spike) {
	df_spike$Group <- 'Added leukocyte DNA (pg)'
	df_spike$SampleType <- dplyr::case_when(
		str_detect(df_spike$SampleType, "rbcDNA \\+ 0pg gDNA") ~ "+ 0pg",
		str_detect(df_spike$SampleType, "rbcDNA \\+ 6pg gDNA") ~ "+ 6pg",
		str_detect(df_spike$SampleType, "rbcDNA \\+ 12pg gDNA") ~ "+ 12pg",
		str_detect(df_spike$SampleType, "rbcDNA \\+ 18pg gDNA") ~ "+ 18pg",
		str_detect(df_spike$SampleType, "200pg gDNA") ~ "+ 200pg",
		str_detect(df_spike$SampleType, "^gDNA") ~ "gDNA",
		TRUE ~ df_spike$SampleType
	)
	spike_levels <- c('+ 0pg', '+ 6pg', '+ 12pg', '+ 18pg', '+ 200pg')
	df_spike$SampleType <- factor(df_spike$SampleType, levels = intersect(spike_levels, unique(df_spike$SampleType)))
}
# 准备绘图面板
p_controls <- ggplot(df_controls, aes(x = SampleType, y = value_mt, fill = SampleType)) +
	geom_boxplot(outlier.color = NA, linewidth = 0.1) +
	geom_jitter(width = 0.2, size = 0.003) +
	geom_hline(yintercept = 0.04, linetype = "dashed", color = 'darkgrey') +
	scale_fill_manual(values = c("#E64B35FF", "#3E60AA")) +
	theme_sig +
	theme(
        plot.title = element_text(size = 8, hjust = 0.5, color = "black"),
        axis.title = element_text(size = 8),
        axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0, unit = "pt"))
    ) +
	scale_y_continuous(breaks = c(0, 0.04, 0.05, 0.10, 0.15), limits = c(0, 0.16)) +
	labs(x = NULL, y = "Proportion of rbcDNA\nor leukocyte DNA\nmapped to MT regions (%)", title = 'Controls') 

p_spike <- ggplot(df_spike, aes(x = SampleType, y = value_mt, fill = SampleType)) +
	geom_boxplot(outlier.color = NA, linewidth = 0.1) +
	geom_jitter(width = 0.2, size = 0.003) +
	geom_hline(yintercept = 0.04, linetype = "dashed", color = 'darkgrey') +
	scale_fill_manual(values = c("#99000D", "#FCBBA1", "#FEE5D9", "#F7FBFF", "#08306B")) +
	theme_sig +
	theme(
        axis.title.y = element_blank(),
		plot.title = element_text(size = 8, hjust = 0.5, color = "black"),
		axis.text = element_text(size = 6)
	) +
	scale_y_continuous(breaks = c(0, 0.04, 0.05, 0.10, 0.15), limits = c(0, 0.16)) +
	labs(title = "Added leukocyte DNA (pg)")

# Figure S2 A/B: genome-wide distribution (全基因组及局部染色体片段分布)
HD5_gDNA_1000kb = merge(HD5_gDNA_1000kb, df_HD_1000kanno[, c('feature', 'arm')], by = 'feature')
HD5_rbcDNA_1000kb = merge(HD5_rbcDNA_1000kb, df_HD_1000kanno[, c('feature', 'arm')], by = 'feature')
HD5_gDNA_1000kb$SampleType = 'Control_gDNA'
HD5_rbcDNA_1000kb$SampleType = 'Control_rbcDNA'
df_HD_1000kanno$SampleType = 'trn_HD'

HD5_gDNA_100kb = merge(HD5_gDNA_100kb, df_HD_100kanno[, c('feature', 'arm')], by = 'feature')
HD5_rbcDNA_100kb = merge(HD5_rbcDNA_100kb, df_HD_100kanno[, c('feature', 'arm')], by = 'feature')
HD5_gDNA_100kb$SampleType = 'Control_gDNA'
HD5_rbcDNA_100kb$SampleType = 'Control_rbcDNA'
df_HD_100kanno$SampleType = 'trn_HD'

used_cols = setdiff(colnames(HD5_rbcDNA_1000kb), colnames(HD5_rbcDNA_1000kb)[grep('CRL', colnames(HD5_rbcDNA_1000kb))])
exp = as.data.frame(rbind(
	HD5_rbcDNA_1000kb[, used_cols], HD5_gDNA_1000kb[, used_cols], df_HD_1000kanno[, used_cols]
))
exp$SampleType = factor(exp$SampleType, levels = c('Control_rbcDNA', 'Control_gDNA', 'trn_HD'))

p0_base <- ggplot() +
	geom_hline(yintercept = 1, linewidth = 0.3, linetype = 'dashed', color = 'grey') +
	geom_ribbon(data = exp[exp$SampleType != 'trn_HD',], aes(x = start, ymin = min, ymax = max, fill = SampleType), alpha = 0.6) +
	geom_line(data = exp[exp$SampleType != 'trn_HD',], aes(x = start, y = median, group = 1, color = SampleType), linewidth = 0.3) +
	scale_fill_manual(values = c("#E64B35FF", "#3E60AA")) +
	scale_color_manual(values = c("#E64B35FF", "#3E60AA")) +
	facet_grid(SampleType ~ arm, switch = "x", space = "free_x", scales = "free_x") +
	ylim(0.5, 2) + labs(y=NULL) +
	mytheme + theme(strip.text.x = element_blank(), panel.background = element_rect(fill = '#F0F0F0'))

p1_base <- ggplot() +
	geom_hline(yintercept = 1, linewidth = 0.3, linetype = 'dashed', color = 'grey') +
	geom_ribbon(data = exp[exp$SampleType == 'trn_HD',], aes(x = start, ymin = min, ymax = max, fill = SampleType), alpha = 0.6) +
	geom_line(data = exp[exp$SampleType == 'trn_HD',], aes(x = start, y = median, group = 1, color = SampleType), linewidth = 0.3) +
	scale_fill_manual(values = c("#BC3C29CC")) +
	scale_color_manual(values = c("#BC3C29CC")) +
	facet_grid(SampleType ~ arm, switch = "x", space = "free_x", scales = "free_x") +
	ylim(0.5, 2) + labs(y=NULL) +
	mytheme + theme(strip.text.x = element_blank())
	
p1_anno <- ggplot(df_HD_1000kanno) +
	geom_tile(aes(x = start, y = broadPeak_y, fill = broadPeak), colour = NA) +
	scale_fill_gradient2(low = "white", high = "red4") +
	facet_grid(. ~ arm, switch = "x", space = "free_x", scales = "free_x") +
	mytheme + labs(x = 'Chromosomes', y = NULL) +
	theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.title.x = element_text(color = "black", size = 8))
	
aligned_panel_A <- cowplot::align_plots(p0_base, p1_base, p1_anno, align = "v", axis = "lr")
right_col_A <- plot_grid(plotlist = aligned_panel_A, ncol = 1, rel_heights = c(1.6, 0.8, 0.5))

y_label_A <- ggdraw() + draw_label("Median-normalized\nread counts", angle = 90, size = 8, y = 0.55)

panel_base_A <- plot_grid(y_label_A, right_col_A, ncol = 2, rel_widths = c(0.03, 1),
                          labels = c("A", ""), label_size = 12, label_x = 0.01, label_y = 1, hjust = 0, vjust = 0)

panel1 <- ggdraw(panel_base_A) +
	draw_label("Control samples, rbcDNA (n = 5)", x = 0.06, y = 0.96, hjust = 0, vjust = 1, size = 8) +
	draw_label("Control samples, leukocyte DNA (n = 5)", x = 0.06, y = 0.67, hjust = 0, vjust = 1, size = 8) +
	draw_label("Non-GC (n = 220)", x = 0.06, y = 0.4, hjust = 0, vjust = 1, size = 8)


used_cols = setdiff(colnames(HD5_rbcDNA_100kb), colnames(HD5_rbcDNA_100kb)[grep('CRL', colnames(HD5_rbcDNA_100kb))])
# chr1:198000001-199000000
exp1_1 = df_HD_100kanno[(df_HD_100kanno$chromosome == 1) &
	                        (df_HD_100kanno$start > 198000001) &
	                        (df_HD_100kanno$end < 200100001),]
exp1 = as.data.frame(rbind(
	HD5_rbcDNA_100kb[(HD5_rbcDNA_100kb$chromosome == 1) &
		                 (HD5_rbcDNA_100kb$start > 198000001) &
		                 (HD5_rbcDNA_100kb$end < 200100001), used_cols],
	HD5_gDNA_100kb[(HD5_gDNA_100kb$chromosome == 1) &
		               (HD5_gDNA_100kb$start > 198000001) &
		               (HD5_gDNA_100kb$end < 200100001), used_cols]))
exp1$SampleType = factor(exp1$SampleType, levels = c('Control_rbcDNA', 'Control_gDNA'))

p0_gc <- ggplot() +
	geom_hline(yintercept = 1, linewidth = 0.3, linetype = 'dashed', color = 'grey') +
	geom_ribbon(data = exp1, aes(x = start, ymin = min, ymax = max, fill = SampleType), alpha = 0.5) +
	geom_line(data = exp1, aes(x = start, y = median, group = 1, color = SampleType), linewidth = 0.3) +
	scale_fill_manual(values = c("#E64B35FF", "#3E60AA")) +
	scale_color_manual(values = c("#E64B35FF", "#3E60AA")) +
	facet_grid(SampleType ~ arm, switch = "x", space = "free_x", scales = "free_x") +
	ylim(0.5, 4.5) + labs(y = NULL) +
	mytheme + theme(strip.text.x = element_blank(), panel.background = element_rect(fill = '#F0F0F0'))

p0_gc1 <- ggplot() +
	geom_hline(yintercept = 1, linewidth = 0.3, linetype = 'dashed', color = 'grey') +
	geom_ribbon(data = exp1_1, aes(x = start, ymin = min, ymax = max), fill = "#BC3C29CC", alpha = 0.5) +
	geom_line(data = exp1_1, aes(x = start, y = median, group = 1), color = "#BC3C29CC", linewidth = 0.3) +
	facet_grid(SampleType ~ arm, switch = "x", space = "free_x", scales = "free_x") +
	ylim(0.5, 4.5) + labs(y = NULL) +
	mytheme + theme(strip.text.x = element_blank())
	
p1_anno_part1 <- ggplot(exp1_1) +
	geom_tile(aes(x = start, y = broadPeak_y, fill = broadPeak), colour = NA) +
	scale_fill_gradient2(low = "white", high = "red4") +
	facet_grid(. ~ arm, switch = "x", space = "free_x", scales = "free_x") +
	labs(x = NULL, y = NULL) +
	mytheme + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) 
	
aligned_panel <- cowplot::align_plots(p0_gc, p0_gc1, p1_anno_part1, align = "v", axis = "lr")
right_col_B <- plot_grid(plotlist = aligned_panel, ncol = 1, rel_heights = c(2.1, 1.05, 0.65))

y_axis_B <- ggdraw() + draw_label("Median-normalized\nread counts", angle = 90, size = 8, vjust = 0.5, y = 0.59)
panel_base <- plot_grid(y_axis_B, right_col_B, ncol = 2, rel_widths = c(0.08, 1),
                        labels = c("B", ""), label_size = 12, label_x = 0.01, label_y = 1.12, hjust = 0, vjust = 0) +
						theme(plot.margin = margin(t = 15, b = 11, unit = "pt"))

panel2 <- ggdraw(panel_base) +
	draw_label("Control samples,\nrbcDNA (n = 5)", x = 0.95, y = 0.84, hjust = 1, vjust = 1, size = 8) +
	draw_label("Control samples,\nleukocyte DNA (n = 5)", x = 0.95, y = 0.64, hjust = 1, vjust = 1, size = 8) +
	draw_label("Non-GC (n = 220)", x = 0.95, y = 0.4, hjust = 1, vjust = 1, size = 8)

figs2_CD <- plot_grid(p_controls, p_spike, ncol = 2, rel_widths = c(0.7, 1.2), align = "h", axis = "tblr",
                      labels = c('C', 'D'), label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 1.01, hjust = 0, vjust = 0)


figs2 <- plot_grid(panel1 + theme(plot.margin = margin(t = 12, unit = "pt")),
                   plot_grid(panel2, figs2_CD, ncol = 2, rel_widths = c(1, 1.4)) + theme(plot.margin = margin(t = 12, unit = "pt")),
                   ncol = 1, rel_heights = c(1, 0.8))
ggsave(file.path(out_dir, 'FigureS2.pdf'), figs2, width = 8, height = 6)# , device = cairo_pdf)
