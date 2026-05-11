suppressPackageStartupMessages({
	library(stringr)
	library(tidyverse)
	library(rtracklayer)
	library(GenomicRanges)
	library(openxlsx)
})
args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

load(file.path(script_dir, 'FeatureAnno/refData', 'hg38_annotation.RData'))
load(file.path(script_dir, 'FeatureAnno/refData/CellResearch', 'ref.RData'))
load(file.path(script_dir, 'FeatureAnno/refData/CellResearch', 'HD_deep_10sample.RData'))
load(file.path(script_dir, 'FeatureAnno/refData/CellResearch', 'HD_deep_10sample_anno.RData'))

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'addGenomicInfo.R'), chdir = TRUE)


load('./Figures/trn_HD_smooth.RData')

feature_path <- file.path(script_dir, 'FeatureAnno/refData/CellResearch', 'result_d_0_merge_t_rbcDNA_c_gDNA.10samples.broadPeak.bed')
peak_feature <- read.table(feature_path, sep='\t', header=TRUE)
peak_feature$region <- str_c('chr', peak_feature$chromosome, ':', peak_feature$start, '-', peak_feature$end)

HD10_60m_1000k_MN$fc <- HD10_60m_1000k_MN$median / HD10_60m_1000k_gDNA$median
rbcDNA_fea <- HD10_60m_1000k_MN[HD10_60m_1000k_MN$fc > 1.2, ]
print(nrow(rbcDNA_fea))

df_HD_1000k <- df_HD_1000k[order(df_HD_1000k$median, decreasing = TRUE), ]
df_HD_1000k_selected <- head(df_HD_1000k, round(nrow(df_HD_1000k) * 0.05))

rbcDNA_fea <- rbcDNA_fea[rbcDNA_fea$feature %in% df_HD_1000k_selected$feature, ]
print(nrow(rbcDNA_fea))

rbcDNA_fea_region.hg38 <- GRanges(
  seqnames = rbcDNA_fea$chromosome,
  ranges = IRanges(start = rbcDNA_fea$start, end = rbcDNA_fea$end),
  feature = rbcDNA_fea$feature
)

top_feature_region.hg38 <- GRanges(
  seqnames = peak_feature$chromosome,
  ranges = IRanges(start = peak_feature$start, end = peak_feature$end),
  feature = peak_feature$region
)

overlaps_1000k <- findOverlaps(rbcDNA_fea_region.hg38, top_feature_region.hg38)

top_feature_region.hg38 <- top_feature_region.hg38[unique(subjectHits(overlaps_1000k))]
print(length(top_feature_region.hg38))
write.xlsx(list(as.data.frame(unique(top_feature_region.hg38)), df_HD_1000k_selected), './Figures/rbcDNAenriched_updated_regions.xlsx', rowNames=FALSE)

overlapping_1m_features <- unique(rbcDNA_fea_region.hg38$feature[queryHits(overlaps_1000k)])
CN_smooth_r1_1000k$broadPeak <- ifelse(CN_smooth_r1_1000k$region %in% overlapping_1m_features, 1, 0)
print(table(CN_smooth_r1_1000k$broadPeak))

HD10_60m_100k_MN.hg38 <- GRanges(HD10_60m_100k_MN)
overlaps_100k <- findOverlaps(HD10_60m_100k_MN.hg38, top_feature_region.hg38)

overlapping_100k_features <- unique(HD10_60m_100k_MN.hg38$feature[queryHits(overlaps_100k)])
CN_smooth_r1_100k$broadPeak <- ifelse(CN_smooth_r1_100k$region %in% overlapping_100k_features, 1, 0)
print(table(CN_smooth_r1_100k$broadPeak))

df_HD_1000kanno <- merge(df_HD_1000k, CN_smooth_r1_1000k[, c('region', 'arm', 'broadPeak', 'broadPeak_y')], by.x='feature', by.y='region')
df_HD_100kanno <- merge(df_HD_100k, CN_smooth_r1_100k[, c('region', 'arm', 'broadPeak', 'broadPeak_y')], by.x='feature', by.y='region')

updated_rbcDNAenriched_regions <- as.data.frame(unique(top_feature_region.hg38$feature))
colnames(updated_rbcDNAenriched_regions) <- 'rbcDNAenriched_regions'
print(nrow(updated_rbcDNAenriched_regions))

updated_rbcDNAenriched_regions <- updated_rbcDNAenriched_regions %>%
  separate(rbcDNAenriched_regions, into = c("chr", "pos"), sep = ":", remove = FALSE) %>%
  separate(pos, into = c("start", "end"), sep = "-") %>%
  mutate(start = as.numeric(start), end = as.numeric(end))

save(updated_rbcDNAenriched_regions, df_HD_1000kanno, df_HD_100kanno, file = './Figures/trn_HD_smooth_anno.RData')

update_rbcDNAenriched <- updated_rbcDNAenriched_regions[, c('chr', 'start', 'end')]
colnames(update_rbcDNAenriched) <- c('chromosome', 'start', 'end')
anno_bed <- add_GenomicAnnotation_userbed(update_rbcDNAenriched, file.path(script_dir, 'FeatureAnno'), 'update_rbcDNAenriched')
write.xlsx(list(anno_bed), './Figures/rbcDNAenriched_updated_regions_anno.xlsx', rowNames=FALSE)