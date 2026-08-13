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
  library(ggplot2)
  library(ggsci)
  library(cowplot)
  library(dplyr)
  library(ggpubr)
  library(openxlsx)
  library(GenomicRanges)
})

source(file.path(script_dir, 'function.r'), chdir = TRUE)
source(file.path(script_dir, 'addGenomicInfo.R'), chdir = TRUE)
nrc_color <- pal_npg("nrc", alpha = 0.7)(9)

theme_bar <- theme_bw() +
      theme(legend.position='none',
        axis.text = element_text(color = "black", size = 6),
        axis.title = element_text(color = "black", size = 8),
        axis.title.x = element_blank(),
        strip.text = element_text(color = "black", size = 6),
        strip.background = element_blank(),
        plot.title = element_text(color = "black", size = 6, hjust = 0.5),
        axis.line = element_blank(),
        axis.ticks = element_line(linewidth = 0.4),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
        panel.grid = element_blank(),
        plot.margin = margin(2, 2, 2, 2))

# ---------------------------------------------------------

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')
load('./Figures/trn_HD_smooth_anno.RData')
load("./figure_scripts/FeatureAnno/refData/hg38_annotation.RData")
# STAD.hg38, atac.hg38

gc_feas = read.table('Human_model/results/2_FeatureSelection/all.gc.bed.out', sep='\t', head=FALSE)
rownames(gc_feas) = str_c('chr', gc_feas[,1], ':', gc_feas[,2], '-', gc_feas[,3])
gc_feas_1000 = head(gc_feas, 1000)
colnames(gc_feas_1000)[1:3] = c('chromosome', 'start', 'end')
gc_feas_1000$chromosome = str_c('chr', gc_feas_1000$chromosome)
gc_feas_1000$feature = str_c(gc_feas_1000[,1], ':', gc_feas_1000[,2], '-', gc_feas_1000[,3])
rownames(gc_feas_1000) = gc_feas_1000$feature
gr_regions = GRanges(gc_feas_1000)

fea_avg = read.xlsx('Figures/Figure2_GCrbcDNA_features.xlsx')
gc_feas_1000[fea_avg[which(fea_avg$Label=='GC_high'), 'region'], 'type'] = 'GC_high'
gc_feas_1000[fea_avg[which(fea_avg$Label=='NonGC_high'), 'region'], 'type'] = 'NonGC_high'

gc_high_features = fea_avg[which(fea_avg$Label=='GC_high'), 'region']
nongc_high_features = fea_avg[which(fea_avg$Label=='NonGC_high'), 'region']

#### annotation
feature_bed = gc_feas_1000[, c('chromosome', 'start', 'end')]
anno_bed = add_GenomicAnnotation_userbed(feature_bed, './figure_scripts/FeatureAnno', 'GC_top1000')#

#### Figure 3A
rownames(updated_rbcDNAenriched_regions) = updated_rbcDNAenriched_regions$rbcDNAenriched_regions
rbcDNA_enriched <- updated_rbcDNAenriched_regions
rbcDNA_enriched.gr <- GRanges(rbcDNA_enriched)

hit_obj <- findOverlaps(gr_regions, rbcDNA_enriched.gr)
overlapped_regions <- length(unique(queryHits(hit_obj)))
total_regions <- nrow(gc_feas_1000)
total_rbc <- nrow(rbcDNA_enriched) # 新增：用于修复标题显示
no_overlapped_regions <- total_regions - overlapped_regions
overlapped <- updated_rbcDNAenriched_regions
df <- data.frame(name = c("overlapped", "no_overlapped"),level = c("1", "1"),
                count = c(overlapped_regions, no_overlapped_regions),
                values = c(round(100 * overlapped_regions / total_regions, 1),round(100 * no_overlapped_regions / total_regions, 1)))

df$label <- c(paste0(df$values[1], "% (", df$count[1], "/", total_regions, ")\n", "overlapped with\nrbcDNA-enriched regions"), paste0(df$values[2], "% (", df$count[2], "/", total_regions, ")"))
pA <- ggplot(df, aes(x = level, y = values, fill = name)) +
      geom_col(width = 1, color = "gray90", linewidth = 0.5, position = position_stack()) +
      coord_polar(theta = "y") +
      scale_fill_manual(values = c("overlapped" = "#1B7837", "no_overlapped" = "#D9D9D9")) +
      scale_x_discrete(breaks = NULL) +
      scale_y_continuous(breaks = NULL) +
      labs(x = NULL,y = NULL,title = paste0("GC-associated rbcDNA regions\noverlapped with\n", total_rbc, " rbcDNA-enriched regions")) +
    annotate("text",x = 1.35,y = 2,label = df$label[1],hjust = 0,vjust = 0.5,size = 6 / .pt, lineheight = 0.9) +
    annotate("text",x = 1.05,y = 55,label = df$label[2],hjust = 0.5,vjust = 0.5,size = 6 / .pt) +

    theme_minimal() +
    theme(legend.position = "none",
           plot.title = element_text(size = 8, hjust = 0.5, lineheight = 1.1),
           plot.margin = margin(10, 20, 10, 10, "pt"),panel.grid = element_blank(),
           axis.text = element_blank(),axis.title = element_blank())

### random 1000
set.seed(1234)
bed_df <- read.table("figure_scripts/FeatureAnno/total_1000w.bed", header = FALSE, col.names = c("chr", "start", "end"))
bed_df$width <- bed_df$end - bed_df$start
bed_df$chr <- str_c('chr', bed_df$chr)
rownames(bed_df) = str_c(bed_df$chr, ':', bed_df$start, '-', bed_df$end)
bed_df <- bed_df[setdiff(rownames(bed_df), rownames(gc_feas_1000)), ]
target_lengths <- width(gr_regions) - 1
length_dist <- table(target_lengths)
sampled_df_list <- list()
for (len in as.numeric(names(length_dist))) {
    n <- length_dist[as.character(len)]
    candidate_rows <- bed_df %>% filter(width == len)
    # 防止候选不足
    if (nrow(candidate_rows) >= n) {
      sampled_df <- candidate_rows %>% sample_n(n)
    } else if (nrow(candidate_rows) > 0) {
      sampled_df <- candidate_rows %>% sample_n(nrow(candidate_rows), replace = FALSE)
      warning(sprintf("Only %d candidates for length %d (needed %d)", nrow(candidate_rows), len, n))
    } else {
      next 
    }
    sampled_df_list[[as.character(len)]] <- sampled_df
}

sampled_df_all <- bind_rows(sampled_df_list)
random_gr <- GRanges(sampled_df_all)
length(intersect(rownames(as.data.frame(gr_regions)), rownames(as.data.frame(random_gr))))
GC_f_atac = atac_merged(gr_regions, 'GC_features')
random_f_atac = atac_merged(random_gr, 'random_features')
renrich_f_atac = atac_merged(rbcDNA_enriched.gr, 'rbcDNA_enriched')
length(intersect(GC_f_atac[,1], random_f_atac[,1]))

GC_f_atac[which(GC_f_atac$GC_label %in% gc_high_features), 'label'] = 'GC_features_high'
GC_f_atac[which(GC_f_atac$GC_label %in% nongc_high_features), 'label'] = 'GC_features_low'

#### normalized by length
width_nor_atac_signal=as.data.frame(rbind(renrich_f_atac[, c('GC_label','label', 'atac_sum', 'GMP','MEP','Ery','STAD_sum')],
                                          GC_f_atac[, c('GC_label','label', 'atac_sum', 'GMP','MEP','Ery','STAD_sum')],
                                          random_f_atac[, c('GC_label','label', 'atac_sum', 'GMP','MEP','Ery','STAD_sum')]))
width_nor_atac_signal$label = factor(width_nor_atac_signal$label , levels=c('rbcDNA_enriched', 'GC_features_low', 'GC_features_high', 'random_features'))
rownames(width_nor_atac_signal) = width_nor_atac_signal$GC_label
row_names <- rownames(width_nor_atac_signal)
region_strings <- sub(".*:", "", row_names)
positions <- do.call(rbind, strsplit(region_strings, "-"))
positions <- apply(positions, 2, as.numeric)

diffs <- positions[,2] - positions[,1]
width_nor_atac_signal$width = diffs / 1000
width_nor_atac_signal[is.na(width_nor_atac_signal$atac_sum), 'atac_sum'] = 0
width_nor_atac_signal$nor_atac_sum = (width_nor_atac_signal$atac_sum/width_nor_atac_signal$width)# * 100
width_nor_atac_signal[is.na(width_nor_atac_signal$STAD_sum), 'STAD_sum'] = 0
width_nor_atac_signal$nor_STAD_sum = (width_nor_atac_signal$STAD_sum/width_nor_atac_signal$width)# * 100

atac_df = width_nor_atac_signal[which(width_nor_atac_signal$label!='random_features'), ]
cutoff = median(atac_df$nor_atac_sum)

rbc_high_n <- nrow(atac_df[which((atac_df$label == "rbcDNA_enriched") & (atac_df$nor_atac_sum > cutoff)), ])
gc_high_n <- nrow(atac_df[which(((atac_df$label == "GC_features_low") | (atac_df$label == "GC_features_high")) & (atac_df$nor_atac_sum > cutoff)), ])
rbc_total_n <- nrow(rbcDNA_enriched)
gc_total_n <- 1000
p1 <- rbc_high_n / rbc_total_n
p2 <- gc_high_n / gc_total_n

df2 <- data.frame(
    group = rep(c("rbcDNA-enriched\nregions", "GC-associated\nfeatures"), each = 2),
    type = c("high", "no", "high", "no"),
    value = c(100 * p1, 100 * (1 - p1), 100 * p2, 100 * (1 - p2)),
    high_n = c(rbc_high_n, NA, gc_high_n, NA),
    total_n = c(rbc_total_n, NA, gc_total_n, NA))
df2$type <- factor(df2$type, levels = c("no", "high"))
df2$group <- factor(df2$group, levels = c("rbcDNA-enriched\nregions", "GC-associated\nfeatures"))
df2$fill_col <- ifelse(as.character(df2$type) == "no", "no", ifelse(df2$group == "rbcDNA-enriched\nregions", "rbc_high", "gc_high"))
df2$fill_col <- factor(df2$fill_col, levels = c("no", "rbc_high", "gc_high"))
label_df <- df2[df2$type == "high", ]
label_df$label <- paste0(round(label_df$value, 1),"%\n(",label_df$high_n,"/",label_df$total_n,")")
label_df$ypos <- ifelse(label_df$value < 8, label_df$value + 6, label_df$value / 2)

pB_1 <- ggplot(df2, aes(x = group, y = value, fill = fill_col)) +
    geom_bar(stat = "identity", color = "grey70", linewidth = 0.4, width = 0.9) +
    geom_text(data = label_df,aes(x = group, y = value + 4, label = label),inherit.aes = FALSE,size = 6 / .pt,color = "black",lineheight = 0.9) +
    scale_fill_manual(values = c("no" = "#F0F0F0", "rbc_high" = "#BC3C29FF", "gc_high" = "#0072B5FF")) +
    scale_y_continuous(limits = c(0, 105),breaks = c(0, 25, 50, 75, 100),expand = c(0, 0)) +
    ylab("Percentage of rbcDNA features with\nhigh chromatin accessibility across\nnormal hematopoietic cell types") +
    xlab(NULL) + theme_bar + theme(axis.text.x = element_text(size = 6, color = "black", angle = 45, hjust = 1, vjust = 1))

atac_df$label2 <- dplyr::case_when(
    atac_df$label == "rbcDNA_enriched" ~ "rbcDNA-enriched\nregions",
    atac_df$label == "GC_features_low" ~ str_c(length(nongc_high_features), " down-regulated\nrbcDNA features in GC"),
    atac_df$label == "GC_features_high" ~ str_c(length(gc_high_features), " up-regulated\nrbcDNA features in GC"),
    TRUE ~ atac_df$label)

atac_df$label2 <- factor(atac_df$label2,
    levels = c("rbcDNA-enriched\nregions","455 down-regulated\nrbcDNA features in GC","545 up-regulated\nrbcDNA features in GC"))

pB_2 <- ggplot(atac_df, aes(x = label2, y = log(nor_atac_sum), fill = label2)) +
    geom_boxplot(width = 0.6, linewidth = 0.4, outlier.size = 0.6) +
    stat_compare_means(
        comparisons = list(
            c("rbcDNA-enriched\nregions", "455 down-regulated\nrbcDNA features in GC"),
            c("545 up-regulated\nrbcDNA features in GC", "455 down-regulated\nrbcDNA features in GC"),
            c("rbcDNA-enriched\nregions", "545 up-regulated\nrbcDNA features in GC")),
        method = "wilcox.test", label = "p.signif", label.y.npc = 0.95, size = 6 / .pt, tip.length = 0.02) +
    ylab("Log-transformed chromatin accessibility\nsignal intensity across genomic regions\nin normal hematopoietic cell types") + xlab(NULL) +
    scale_fill_manual(
        values = c(
            "rbcDNA-enriched\nregions" = "red4",
            "455 down-regulated\nrbcDNA features in GC" = nrc_color[4],
            "545 up-regulated\nrbcDNA features in GC" = nrc_color[1])) +
    theme_bar + theme(axis.text.x = element_text(size = 6, color = "black", angle = 45, hjust = 1, vjust = 1))

#### figure s3c
anno <- anno_bed[, c('chr','start','end','arm')]
anno$bin = anno_bed$region
bins <- unique(GRanges(anno[, setdiff(colnames(anno), 'arm')]))

gc_feas_1000$bin = rownames(gc_feas_1000)
gr_regions2 = GRanges(gc_feas_1000)
sampled_df_all$bin = rownames(sampled_df_all)
random_gr2 <- GRanges(sampled_df_all)

gc_cnv = cnv_merged(gr_regions2)
random_cnv = cnv_merged(random_gr2)

cnv_stad <- as.data.frame(rbind(
    cbind(as.numeric(gc_cnv[[1]]), "Loss", "GC-associated\nfeatures"),
    cbind(as.numeric(gc_cnv[[2]]), "Gain", "GC-associated\nfeatures"),
    cbind(as.numeric(random_cnv[[1]]), "Loss", "random 1000 regions"),
    cbind(as.numeric(random_cnv[[2]]), "Gain", "random 1000 regions")
))
cnv_stad$V1 <- as.numeric(cnv_stad$V1)
cnv_stad$V2 <- factor(cnv_stad$V2, levels = c("Loss", "Gain"))
cnv_stad$V3 <- factor(cnv_stad$V3, levels = c("GC-associated\nfeatures", "random 1000 regions"))
pC <- ggplot(cnv_stad, aes(x = V3, y = V1, fill = V3)) +
    geom_boxplot(width = 0.6, linewidth = 0.4, outlier.size = 0.6) + facet_grid(. ~ V2) +
    stat_compare_means(aes(label = paste0("Wilcoxon,\np = ", after_stat(p.format))),
               method = "wilcox.test", label.x.npc = 'center', label.y.npc = 0.95, size = 6 / .pt, hjust = 0.5, lineheight = 0.9) +
    scale_fill_manual(values = c("GC-associated\nfeatures" = pal_nejm()(10)[2], "random 1000 regions" = "grey")) +
    ylab("Proportion of CNV gain/loss events\nacross genomic regions\nin 442 TCGA STAD Samples") + xlab(NULL) +
    theme_bar + theme(axis.text.x = element_text(size = 6, color = "black", angle = 45, hjust = 1, vjust = 1))

right_panel = plot_grid(pB_1, pB_2, pC, nrow = 1, align = "h", axis = "tb", rel_widths = c(1, 1.1, 1.2, 0.7), 
                  labels = c('B', '', 'C'), label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 1.0, hjust = 0, vjust = 1)

FigS3 = plot_grid(pA, right_panel,ncol = 2,rel_widths = c(1.2, 4.3), 
                  labels = c('A', ''),label_size = 12,label_fontface = "bold",label_x = 0.01, label_y = 1.0, hjust = 0, vjust = 1)

ggsave(file.path(out_dir, 'FigureS3.pdf'), FigS3, width=8, height=3)#, device = cairo_pdf)
