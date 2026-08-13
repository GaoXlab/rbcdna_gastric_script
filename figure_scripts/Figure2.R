args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
	library(rlang)
	library(stringr)
	library(readxl)
	library(openxlsx)
	library(cowplot)
	library(RColorBrewer)
	library(scales)
	library(ggsci)
	library(pheatmap)
	library(ggplotify)
	library(GenomeInfoDb)
	library(GenomicRanges)
	library(reshape)
	library(ggplot2)
	library(ggpubr)
	library(annotatr)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'function.r'), chdir = TRUE)
nrc_color <- pal_npg("nrc", alpha = 0.7)(9)

load('./Figures/sampleinfo.RData')
load('./Figures/trn_100k.RData')

# 将公用格式存为列表
figure2_plot_grid <- function(...) {
  plot_grid(..., label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 1.01, hjust = 0, vjust = 0) + theme(plot.margin = margin(t = 14, unit = "pt"))
}

top_feas = read.table('./Human_model/results/2_FeatureSelection/all.gc.bed.out', sep = '\t', head = FALSE)
rownames(top_feas) = str_c('chr', top_feas[, 1], ':', top_feas[, 2], '-', top_feas[, 3])
top_feas_1000 = head(top_feas, 1000)
top_feas_1000$feature = str_c(top_feas_1000[, 1], ':', top_feas_1000[, 2], '-', top_feas_1000[, 3])
colnames(top_feas_1000)[1:3] = c('chr', 'start', 'end')

all_df = read.table('./Human_model/gc/all.gc.tab', sep = '\t', head = TRUE, comment.char = "")
colnames(all_df) = gsub('^X|.uniq.nodup.bam', '', colnames(all_df))
rownames(all_df) = str_c('chr', all_df[, 1], ':', all_df[, 2], '-', all_df[, 3])
all_df = all_df[, 4:ncol(all_df)]
all_df = as.data.frame(t(all_df))

trn_ids = read.table('./Human_model/modelData/gc.trn.ids.txt')
GC_filter = all_df[as.character(trn_ids$V1), rownames(top_feas_1000)]

trnval_gc_df = as.data.frame(GC_filter)
trnval_gc_df$label = 'Non-GC'
trnval_gc_df[grep('GLRGC', rownames(trnval_gc_df)), 'label'] = 'GC'
pvalue = c()
symp = c()
mean_all = c()
median_all = c()
sd_all = c()

for (i in c(colnames(trnval_gc_df)[grep('chr', colnames(trnval_gc_df))])) {
	tmp = trnval_gc_df[, c(i, 'label')]
	colnames(tmp)[1] = 'fea'
	tmp$label = factor(tmp$label)
	pvalue = c(pvalue, wilcox.test(fea ~ label, data = tmp)$p.value)
	symp <- c(symp, symnum(wilcox.test(fea ~ label, data = tmp)$p.value, corr = FALSE, cutpoints = c(0, 0.0001, .001, .01, .05, .1, 1), symbols = c("****", "***", "**", "*", "ns", ".")))
	avg = aggregate(. ~ label, data = tmp, FUN = mean)
	mean_all = rbind(mean_all, c(avg[avg$label == 'Non-GC', 'fea'], avg[avg$label == 'GC', 'fea']))
	med = aggregate(. ~ label, data = tmp, FUN = median)
	median_all = rbind(median_all, c(med[med$label == 'Non-GC', 'fea'], med[med$label == 'GC', 'fea']))
	sde = aggregate(. ~ label, data = tmp, FUN = sd)
	sd_all = rbind(sd_all, c(sde[sde$label == 'Non-GC', 'fea'], sde[sde$label == 'GC', 'fea']))
}

mean_all = as.data.frame(mean_all)
colnames(mean_all) = c('Non-GC(mean)', 'GC(mean)')
median_all = as.data.frame(median_all)
colnames(median_all) = c('Non-GC(median)', 'GC(median)')
sd_all = as.data.frame(sd_all)
colnames(sd_all) = c('Non-GC(sd)', 'GC(sd)')

fea_avg = cbind(c(colnames(trnval_gc_df)[grep('chr', colnames(trnval_gc_df))]), cbind(mean_all, median_all, sd_all))
fea_avg$p.value = pvalue
fea_avg$symp = symp
fea_avg[fea_avg$symp == '.', 'symp'] = 'ns'
colnames(fea_avg)[1] = 'region'
fea_avg$Label = 'NonGC_high'
fea_avg[which((fea_avg$"GC(median)" / fea_avg$"Non-GC(median)") > 1), 'Label'] = 'GC_high'
write.xlsx(fea_avg, './Figures/Figure2_GCrbcDNA_features.xlsx')

sampleinfo2 = sampleinfo[sampleinfo$Dataset == 'Dataset A, discovery cohort', c('Sample', 'Stage', 'Lauren classification', 'Group', 'Atrophic', 'IntestinalMetaplasia')]
colnames(sampleinfo2) = c('Sample', 'Clinical stage', 'Lauren subtype', 'Group', 'Atrophic', 'IM')
GC_filter = merge(GC_filter, sampleinfo2, by.x = 'row.names', by.y = 'Sample')
rownames(GC_filter) = GC_filter$Row.names


aka2 = GC_filter[, c('Atrophic', 'IM', 'Clinical stage', 'Lauren subtype', 'Group')]
rownames(aka2) = rownames(GC_filter)
colnames(aka2) = c('Atrophic', 'IM', 'Clinical stage', 'Lauren subtype', 'Group')
aka2[aka2$`Clinical stage` == '', 'Clinical stage'] = '/'
aka2$`Clinical stage` = factor(aka2$`Clinical stage`, levels = c('Non-GC', 'I', 'II', 'III'))
aka2[aka2$`Lauren subtype` == '', 'Lauren subtype'] = 'Missing'
aka2$`Lauren subtype` = factor(aka2$`Lauren subtype`, levels = c('Non-GC', 'Intestinal', 'Diffuse', 'Mix', 'Missing'))
aka2$Group = factor(aka2$Group, levels = c('Non-GC', 'GC'))
levels(aka2$Group) <- c("Non-GC (n = 220)", "GC (n = 215)")

ann_colors = list(
	Group = c("Non-GC (n = 220)" = nrc_color[4], "GC (n = 215)" = nrc_color[1]),
	`Clinical stage` = c("Non-GC" = "#F0F0F0",
	                     "I" = colorRampPalette(brewer.pal(9, "Purples"))(6)[3],
	                     "II" = colorRampPalette(brewer.pal(9, "Purples"))(6)[4],
	                     "III" = colorRampPalette(brewer.pal(9, "Purples"))(6)[5]),
	`Lauren subtype` = c("Non-GC" = "#F0F0F0", 'Intestinal' = nrc_color[5], 'Diffuse' = nrc_color[3], 'Mix' = nrc_color[4], 'Missing' = "#D9D9D9"),
	Atrophic = c("No" = "#F0F0F0", "Yes" = colorRampPalette(brewer.pal(9, "Greys"))(6)[3], "Unknown" = "white"),
	IM = c("No" = "#F0F0F0", "Yes" = colorRampPalette(brewer.pal(9, "Greys"))(6)[3], "Unknown" = "white")
)

hmmat2 = t(GC_filter[order(GC_filter$Group), grep('chr', colnames(GC_filter))])

# 1. pheatmap
p_panel_A_obj = pheatmap(hmmat2, scale = 'row',
                         clustering_distance_rows = 'correlation', clustering_distance_cols = 'correlation',
                         color = c(colorRampPalette(colors = c("#084594", "#08519c", "white"))(250),
                                   colorRampPalette(colors = c("white", "#cb181d", "firebrick3"))(250)),
                         annotation_col = aka2,
                         annotation_colors = ann_colors,
                         cutree_rows = 2,
                         show_rownames = FALSE,
                         show_colnames = FALSE,
                         treeheight_row = 20,
                         treeheight_col = 20,
                         fontsize = 6,
                         silent = TRUE) 

# 2. gtable
p_panel_A_gtable <- p_panel_A_obj$gtable
p_panel_A_gtable$widths[length(p_panel_A_gtable$widths) - 1] <- unit(1.5, "cm")

p_panel_A_final <- as.ggplot(p_panel_A_gtable)

top1000_df = t(GC_filter[grep('chr', colnames(GC_filter))])


chr7 = 7; start7 = 84120000; end7 = 84230000
trn_samples_0_top20_chr7 = c("GLRHD0736", "GLRHD0765", "GLRHD0391", "GLRHD0619", "GLRHD0717", "GLRHD0657", "GLRHD0613", "GLRHD0641", "GLRHD0712", "GLRHD0821", "GLRHD0643", "GLRHD0776", "GLRHD0684", "GLRHD0688", "GLRHD0697", "GLRHD0817", "GLRHD0120", "GLRHD0815", "GLRHD0756", "GLRHD0670")
trn_samples_1_top20_chr7 = c("GLRGC0240", "GLRGC0140", "GLRGC0321", "GLRGC0026", "GLRGC0281", "GLRGC0047", "GLRGC0165", "GLRGC0439", "GLRGC0263", "GLRGC0316", "GLRGC0300", "GLRGC0175", "GLRGC0228", "GLRGC0050", "GLRGC0296", "GLRGC0014", "GLRGC0100", "GLRGC0270", "GLRGC0232", "GLRGC0203")

HD_medianInGroup_chr7 <- MNdna_profiles_df1(trn_100k, 'HD_med', intersect(colnames(trn_100k), trn_samples_0_top20_chr7))
HD_medianInGroup_chr7 <- as.data.frame(cbind(trn_100k[, c('chr', 'start', 'end')], HD_medianInGroup_chr7))
HD_medianInGroup_chr7$median <- as.numeric(HD_medianInGroup_chr7$median)

GC_medianInGroup_chr7 <- MNdna_profiles_df1(trn_100k, 'GC_med', intersect(colnames(trn_100k), trn_samples_1_top20_chr7))
GC_medianInGroup_chr7 <- as.data.frame(cbind(trn_100k[, c('chr', 'start', 'end')], GC_medianInGroup_chr7))
GC_medianInGroup_chr7$median <- as.numeric(GC_medianInGroup_chr7$median)

HD_medianInGroup_tmp_chr7 = HD_medianInGroup_chr7[which((HD_medianInGroup_chr7$chr == chr7) &
	                                                        (HD_medianInGroup_chr7$start >= (start7 - 700000)) &
	                                                        (HD_medianInGroup_chr7$start <= (end7 + 500000))),]
GC_medianInGroup_tmp_chr7 = GC_medianInGroup_chr7[which((GC_medianInGroup_chr7$chr == chr7) &
	                                                        (GC_medianInGroup_chr7$start >= (start7 - 700000)) &
	                                                        (GC_medianInGroup_chr7$start <= (end7 + 500000))),]
label1_chr7 = str_c('Chr', as.character(chr7), ':', as.character(min(GC_medianInGroup_tmp_chr7$start)), '-', as.character(max(GC_medianInGroup_tmp_chr7$start)))

p_line_chr7 <- ggplot() +
	geom_vline(aes(xintercept = c(start7, end7)), colour = "#FDC173", size = 0.1, linetype = 'dashed') +
	geom_rect(aes(xmin = start7, xmax = end7, ymin = 53, ymax = 75), fill = '#FDC173', alpha = 0.1) +
	geom_line(data = HD_medianInGroup_tmp_chr7, aes(x = start, y = median), size = 0.5, color = pal_npg("nrc")(9)[4]) +
	geom_line(data = GC_medianInGroup_tmp_chr7, aes(x = start, y = median), size = 0.5, color = pal_npg("nrc")(9)[1]) +
	labs(title = label1_chr7, x = NULL, y = 'Normalized read counts\n(in 100kb)') + 
	theme_bar1 + theme(plot.margin = margin(5.5, 5.5, 5.5, 5.5))

feai_chr7 = as.data.frame(t(as.data.frame(t(top1000_df[str_c('chr', as.character(chr7), ':', as.character(start7), '-', as.character(end7)),]))))
feai_chr7 = merge(feai_chr7, sampleinfo[, c('Sample', 'Group')], by.x = 'row.names', by.y = 'Sample')
colnames(feai_chr7)[2] = 'fea'
feai_chr7$Group = factor(feai_chr7$Group, levels = c('Non-GC', 'GC'))

yr <- range(feai_chr7$fea, na.rm = TRUE)
p_box_chr7 <- ggplot(feai_chr7, aes(x = Group, y = fea, fill = Group)) +
	geom_boxplot(outlier.shape = NA) +
	scale_fill_manual(values = c(pal_npg("nrc")(9)[4], pal_npg("nrc")(9)[1])) +
	stat_compare_means(comparisons = list(c('Non-GC', 'GC')), label = 'p.signif', method = 'wilcox.test', label.y = yr[1] + 0.88 * diff(yr)) +
	labs(title = 'Chr7:84120000-84230000', x = NULL, y = 'Normalized read counts') + 
	theme_sig2 + theme(axis.text.x = element_text(color = "black", size = 6, angle = 45, vjust = 1, hjust=1), 
					   plot.margin = margin(5.5, 5.5, 5.5, 5.5), plot.title = element_text(size = 5, hjust = 0.5))

p_panel_B_chr7 <- plot_grid(p_box_chr7, p_line_chr7, ncol = 2, rel_widths = c(0.95, 2), align = "h", axis = "tb")

chr9 = 9; start9 = 136210000; end9 = 136820000
trn_samples_0_top20_chr9 = c("GLRHD0674", "GLRHD0702", "GLRHD0557", "GLRHD0731", "GLRHD0748", "GLRHD0704", "GLRHD0517", "GLRHD0827", "GLRHD0139", "GLRHD0782", "GLRHD0495", "GLRHD0651", "GLRHD0754", "GLRHD0587", "GLRHD0804", "GLRHD0491", "GLRHD0620", "GLRHD0708", "GLRHD0821", "GLRHD0645")
trn_samples_1_top20_chr9 = c("GLRGC0174", "GLRGC0228", "GLRGC0350", "GLRGC0146", "GLRGC0229", "GLRGC0269", "GLRGC0111", "GLRGC0218", "GLRGC0314", "GLRGC0227", "GLRGC0250", "GLRGC0175", "GLRGC0236", "GLRGC0060", "GLRGC0134", "GLRGC0446", "GLRGC0143", "GLRGC0267", "GLRGC0270", "GLRGC0369")

HD_medianInGroup_chr9 <- MNdna_profiles_df1(trn_100k, 'HD_med', intersect(colnames(trn_100k), trn_samples_0_top20_chr9))
HD_medianInGroup_chr9 <- as.data.frame(cbind(trn_100k[, c('chr', 'start', 'end')], HD_medianInGroup_chr9))
HD_medianInGroup_chr9$median <- as.numeric(HD_medianInGroup_chr9$median)

GC_medianInGroup_chr9 <- MNdna_profiles_df1(trn_100k, 'GC_med', intersect(colnames(trn_100k), trn_samples_1_top20_chr9))
GC_medianInGroup_chr9 <- as.data.frame(cbind(trn_100k[, c('chr', 'start', 'end')], GC_medianInGroup_chr9))
GC_medianInGroup_chr9$median <- as.numeric(GC_medianInGroup_chr9$median)

HD_medianInGroup_tmp_chr9 = HD_medianInGroup_chr9[which((HD_medianInGroup_chr9$chr == chr9) &
	                                                        (HD_medianInGroup_chr9$start >= (start9 - 700000)) &
	                                                        (HD_medianInGroup_chr9$start <= (end9 + 700000))),]
GC_medianInGroup_tmp_chr9 = GC_medianInGroup_chr9[which((GC_medianInGroup_chr9$chr == chr9) &
	                                                        (GC_medianInGroup_chr9$start >= (start9 - 700000)) &
	                                                        (GC_medianInGroup_chr9$start <= (end9 + 700000))),]
label1_chr9 = str_c('Chr', as.character(chr9), ':', as.character(min(GC_medianInGroup_tmp_chr9$start)), '-', as.character(max(GC_medianInGroup_tmp_chr9$start)))

p_line_chr9 <- ggplot() +
	geom_vline(aes(xintercept = c(start9, end9)), colour = "#FDC173", size = 0.1, linetype = 'dashed') +
	geom_rect(aes(xmin = start9, xmax = end9, ymin = 55, ymax = 90), fill = '#FDC173', alpha = 0.1) +
	geom_line(data = HD_medianInGroup_tmp_chr9, aes(x = start, y = median), size = 0.5, color = pal_npg("nrc")(9)[4]) +
	geom_line(data = GC_medianInGroup_tmp_chr9, aes(x = start, y = median), size = 0.5, color = pal_npg("nrc")(9)[1]) +
	labs(title = label1_chr9, x = NULL, y = 'Normalized read counts\n(in 100kb)') + 
	theme_bar1 + theme(plot.margin = margin(5.5, 5.5, 5.5, 5.5))

feai_chr9 = as.data.frame(t(as.data.frame(t(top1000_df[str_c('chr', as.character(chr9), ':', as.character(start9), '-', as.character(end9)),]))))
feai_chr9 = merge(feai_chr9, sampleinfo[, c('Sample', 'Group')], by.x = 'row.names', by.y = 'Sample')
colnames(feai_chr9)[2] = 'fea'
feai_chr9$Group = factor(feai_chr9$Group, levels = c('Non-GC', 'GC'))

yr <- range(feai_chr9$fea, na.rm = TRUE)
p_box_chr9 <- ggplot(feai_chr9, aes(x = Group, y = fea, fill = Group)) +
	geom_boxplot(outlier.shape = NA) +
	scale_fill_manual(values = c(pal_npg("nrc")(9)[4], pal_npg("nrc")(9)[1])) +
	stat_compare_means(comparisons = list(c('Non-GC', 'GC')), label = 'p.signif', method = 'wilcox.test', label.y = yr[1] + 0.85 * diff(yr)) +
	labs(title = 'Chr9:136210000-136820000', x = NULL, y = 'Normalized read counts') + 
	theme_sig2 + theme(axis.text.x = element_text(color = "black", size = 6, angle = 45, vjust = 1, hjust=1), 
					   plot.margin = margin(5.5, 5.5, 5.5, 5.5), plot.title = element_text(size = 5, hjust = 0.5))

p_panel_B_chr9 <- plot_grid(p_box_chr9, p_line_chr9, ncol = 2, rel_widths = c(0.95, 2), align = "h", axis = "tb")

p_panel_B <- plot_grid(p_panel_B_chr7, p_panel_B_chr9, ncol = 1)

## Fig2c rGreat annotation
top1000_features = get_region_anno(fea_avg$region, 'top1000')
write.xlsx(list(top1000_features), './Figures/top1000_features.xlsx', rowNames=TRUE)

all_sig_paths = read.xlsx('./Figures/top1000_features.xlsx')
top20_sel = all_sig_paths[all_sig_paths$label == 'C2_CP',]
top20_sel = top20_sel[-grep('KEGG_', top20_sel$id),]
top20_sel = top20_sel[which(top20_sel$fold_enrichment_hyper > 2),]
write.xlsx(list(top20_sel), 'Figure2_top1000_enrichments.msigdb.C2_CP.xlsx', rowNames = TRUE)
top20_sel$id <- factor(top20_sel$id, levels = top20_sel$id)

p_panel_E <- ggplot(data = top20_sel, aes(x = fold_enrichment_hyper, y = reorder(id, fold_enrichment_hyper), fill = -log10(p_adjust_hyper))) +
	scale_fill_material("red") +
	geom_bar(stat = "identity", width = 0.5, alpha = 0.8) +
	scale_x_continuous(expand = c(0, 0)) +
	geom_text(size = 6 / .pt, aes(x = 0.05, label = id), hjust = 0) +
	labs(x = "Fold enrichment", y = "", title = "msigdb, canonical pathways") +
	guides(fill = guide_colorbar(barwidth = unit(2, "cm"), barheight = unit(0.25, "cm"))) +
	mytheme + theme(axis.text.x = element_text(color = "black", size = 6),axis.ticks.x = element_line(linewidth = 0.4),
					axis.ticks.y = element_blank(), axis.text.y = element_blank(), 
					legend.position='bottom', plot.title = element_text(hjust = 0.5, size = 8))

## figure 2d
ann_intergenic <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_intergenic')
ann_promoter <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_promoters')
ann_exons <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_exons')
ann_firstexons <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_firstexons')
ann_introns <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_introns')
ann_UTR5 <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_5UTRs')
ann_UTR3 <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_3UTRs')
strand(ann_promoter) <- "*"
strand(ann_exons) <- "*"
strand(ann_UTR5) <- "*"
strand(ann_UTR3) <- "*"
strand(ann_introns) <- "*"
strand(ann_intergenic) <- "*"

gc.hg38 <- GRanges(top_feas_1000)
gr_regions <- GenomicRanges::reduce(gc.hg38)
seqlevelsStyle(gr_regions) <- "UCSC"
strand(gr_regions) <- "*"
total_length <- sum(width(gr_regions))

cat_promoter <- ann_promoter
assigned <- GenomicRanges::reduce(cat_promoter)

cat_UTR5 <- GenomicRanges::setdiff(ann_UTR5, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_UTR5))

cat_UTR3 <- GenomicRanges::setdiff(ann_UTR3, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_UTR3))

cat_exons <- GenomicRanges::setdiff(ann_exons, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_exons))

cat_introns <- GenomicRanges::setdiff(ann_introns, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_introns))

cat_intergenic <- GenomicRanges::setdiff(ann_intergenic, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_intergenic))

cat_other <- GenomicRanges::setdiff(gr_regions, assigned)

df_genomic <- data.frame(
	Percentage = c(
		calc_percent(gr_regions, cat_promoter),
		calc_percent(gr_regions, cat_exons),
		calc_percent(gr_regions, cat_UTR5),
		calc_percent(gr_regions, cat_UTR3),
		calc_percent(gr_regions, cat_introns),
		calc_percent(gr_regions, cat_intergenic),
		sum(width(cat_other)) / total_length * 100
	),
	Element = c('promoter', 'exon', "5' UTR", "3' UTR", 'intron', 'intergenic\nregion', 'other'),
	Category = 'genomic'
)

df_genomic$Element <- factor(df_genomic$Element, levels = c('promoter', 'exon', "5' UTR", "3' UTR", 'intron', 'intergenic\nregion', 'other'))
col_genomic <- c('promoter' = '#E64B35FF', 'exon' = '#4DBBD5FF', "5' UTR" = '#00A087FF',
                 "3' UTR" = '#3C5488FF', 'intron' = '#F39B7FFF', 'intergenic\nregion' = '#8491B4FF', 'other' = '#CCCCCC')

p_panel_C <- ggplot(df_genomic, aes(x = Category, y = Percentage, fill = Element)) +
	geom_bar(stat = "identity", width = 0.85) +
	geom_text(aes(label = sprintf("%.1f", Percentage)), position = position_stack(vjust = 0.5), size = 7 / .pt , family = "ArialMT") +
	scale_fill_manual(values = col_genomic) +
	scale_y_continuous(expand = c(0.01, 0.01), limits = c(0, 100)) +
	labs(y = "Proportion of GC-associated\nrbcDNA features across genomic annotations (%)") +
	theme_bar + theme(axis.text.x = element_blank(),
		axis.ticks.x = element_blank(),
		legend.position = "right",
		legend.title = element_blank(),
		legend.text = element_text(size = 6),
		legend.key.size = unit(0.3, "cm")
	)

feature_chromanno = read.table('./figure_scripts/FeatureAnno/gc.grouped_output.txt', sep = '\t', head = TRUE, row.names = 1)
chr_order = c("quescient", "polycomb.repressed", "HET", "transcription", "enhancers", "weak.enhancers", "acetylations", "weak.transcription", "transcribed.and.enhancer", "exon", "promoters", "others", "znf", "bivalent.promoters", "TSS", "DNase")
GC_chromHMM = feature_chromanno[chr_order]

rownames(fea_avg) = fea_avg$region

anno2 = fea_avg[, 'Label', drop = FALSE]
ann_colors2 = list(
	Label = c("NonGC_high" = nrc_color[4], "GC_high" = nrc_color[1])
)

colnames(GC_chromHMM) = gsub('\\.', ' ', colnames(GC_chromHMM))

p_panel_D = pheatmap(t(GC_chromHMM[names(apply(GC_chromHMM, 1, sum))[apply(GC_chromHMM, 1, sum) != 0],]),
                     cluster_cols = TRUE, cluster_rows = FALSE, scale = 'none', show_colnames = FALSE, treeheight_col = 25,
                     annotation_col = anno2, annotation_colors = ann_colors2, fontsize=6,
					 main = "Top 1000 GC-associated rbcDNA features",
					 border_color = NA,
                     color = c(colorRampPalette(colors = c("white", "firebrick3"))(100)))

p_comb <- plot_grid(
	p_panel_A_gtable,
	figure2_plot_grid(p_panel_B, p_panel_E + theme(plot.margin = margin(t=10, b=1, unit = "pt")), ncol = 2, rel_widths = c(0.9, 1), labels = c("B", "E")) ,
	figure2_plot_grid(p_panel_C, as.ggplot(p_panel_D)  + theme(plot.margin = margin(b = 15, unit = "pt")) , ncol = 2, rel_widths = c(0.5, 1.3), labels = c("C", "D")),
	ncol = 1, labels = c("A", "", ""), rel_heights = c(1.1, 1, 0.65)
)

ggsave(file.path(out_dir, 'Figure2.pdf'), p_comb, width = 8, height = 11.2, device = cairo_pdf)