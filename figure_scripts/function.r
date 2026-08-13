suppressPackageStartupMessages({
    library(stringr)
    library(ggsci)
    library(ggplot2)
    library(dbplyr)
    library(rGREAT)
    library(ggpubr)
    library(tidyr)
    library(clusterProfiler)
    library(dplyr)
})

get_region_anno <- function(featurelist, filename){
    all_sig <- c()
    region_df <- tibble(region = featurelist) %>%
        tidyr::separate(region, into = c("chr", "pos"), sep = ":") %>%
        tidyr::separate(pos, into = c("start", "end"), sep = "-") %>%
        dplyr::mutate(across(start:end, as.numeric))

    print(nrow(region_df))

    up.distal.bed <- region_df[, c('chr','start','end')]
    colnames(up.distal.bed) <- c('chr','start','end')
    gr <- GRanges(up.distal.bed)
    res <- great(gr, "msigdb:C2:CP", "txdb:hg38", basal_upstream = 0, basal_downstream = 0, extension = 500000)

    tb <- getEnrichmentTable(res)
    sig <- tb[which(tb$p_adjust_hyper < 0.05), ]
    tb <- tb[order(tb$fold_enrichment_hyper, decreasing = TRUE), ]

    if(nrow(sig) > 0){
        sig$label <- 'C2_CP'
        all_sig <- as.data.frame(rbind(all_sig, sig))
    }
    write.xlsx(list(all_sig), 'feature_region_annotation.xlsx', sheetName = filename, append = TRUE)
    return(all_sig)
}

calc_percent <- function(gr_regions, ann) {
  hits <- findOverlaps(gr_regions, ann)
  if (length(hits) == 0) return(0)
  ov <- pintersect(gr_regions[queryHits(hits)], ann[subjectHits(hits)])
  return(sum(width(GenomicRanges::reduce(ov))) / sum(width(gr_regions)) * 100)
}

atac_merged = function(bed_gr, label){
    atac_merge = as.data.frame(atac.hg38)
    STAD_merge = as.data.frame(STAD.hg38)
    feature_bed <- as.data.frame(bed_gr)
    atac_overlaps <- findOverlaps(bed_gr, atac.hg38)
    stad_overlaps <- findOverlaps(bed_gr, STAD.hg38)
    expanded_result <- data.frame(
      GC_label = rownames(feature_bed)[queryHits (atac_overlaps)],
      atac_label = rownames(atac_merge)[subjectHits(atac_overlaps)],
      atac_sum = apply(atac_merge[subjectHits(atac_overlaps), 6:ncol(atac_merge)],1,sum),
      HSC = atac_merge[subjectHits(atac_overlaps), 'HSC'],
      MPP = atac_merge[subjectHits(atac_overlaps), 'MPP'],
      LMPP = atac_merge[subjectHits(atac_overlaps), 'LMPP'],
      CMP = atac_merge[subjectHits(atac_overlaps), 'CMP'],
      GMP = atac_merge[subjectHits(atac_overlaps), 'GMP'],
      MEP = atac_merge[subjectHits(atac_overlaps), 'MEP'],
      Mono = atac_merge[subjectHits(atac_overlaps), 'Mono'],
      Bcell = atac_merge[subjectHits(atac_overlaps), 'Bcell'],
      CD4Tcell = atac_merge[subjectHits(atac_overlaps), 'CD4Tcell'],
      CD8Tcell = atac_merge[subjectHits(atac_overlaps), 'CD8Tcell'],
      NKcell = atac_merge[subjectHits(atac_overlaps), 'NKcell'],
      CLP = atac_merge[subjectHits(atac_overlaps), 'CLP'],
      Ery = atac_merge[subjectHits(atac_overlaps), 'Ery'],
      stringsAsFactors = FALSE
    )
    expanded_result2 <- data.frame(
      GC_label2 = rownames(feature_bed)[queryHits (stad_overlaps)],
      STAD = STAD_merge[subjectHits(stad_overlaps), 'Normalized.Peak.Score'],
      stringsAsFactors = FALSE
    )

    summed_result_GC <- expanded_result %>%
      group_by(GC_label) %>%
      dplyr::summarise(across(c(atac_sum:Ery), sum, na.rm = TRUE))
    summed_result_GC <- as.data.frame(summed_result_GC)
    summed_result_GC$label = label

    summed_result_GC2 <- expanded_result2 %>%
      group_by(GC_label2) %>%
      dplyr::summarise(STAD_sum = sum(STAD, na.rm = TRUE))
    summed_result_GC2 <- as.data.frame(summed_result_GC2)

    summed_result_GC = merge(summed_result_GC, summed_result_GC2, by.x='GC_label', by.y='GC_label2', all.x=TRUE)
    summed_result_GC[is.na(summed_result_GC)] = 0
    return(summed_result_GC)

}

fast_vectorize <- function(bins_gr, data_gr, min.coverage = 0.9){
    library(data.table)
    library(GenomicRanges)

    o <- findOverlaps(bins_gr, data_gr)

    if (length(o) == 0) {
        out <- rep(0, length(bins_gr))
        names(out) <- bins_gr$bin
        return(out)
    }

    intersect_ranges <- pintersect(ranges(bins_gr)[queryHits(o)], ranges(data_gr)[subjectHits(o)])
    ow <- width(intersect_ranges)

    dt <- data.table(
        bin_idx = queryHits(o),
        fraction = ow / width(bins_gr)[queryHits(o)]
    )
    coverage_dt <- dt[, .(total_fraction = sum(fraction)), by = bin_idx]
    valid_idx <- coverage_dt[total_fraction >= min.coverage, bin_idx]

    out <- rep(0, length(bins_gr))
    out[valid_idx] <- 1
    names(out) <- bins_gr$bin

    return(out)
}

cnv_merged = function(bins_gr){
    library(parallel)
    library(data.table)
    library(GenomicRanges)

    if(!exists("STAD_rm")) {
        load('figure_scripts/FeatureAnno/STAD.CopyNumberSegment.grch38.RData')
    }
    STAD <- STAD_rm

    STAD <- STAD[STAD$Chromosome %in% 1:22 | STAD$Chromosome %in% as.character(1:22), ]
    STAD$Chromosome <- paste0('chr', STAD$Chromosome)

    gc.thresh <- 0.25
    STAD$gain <- ifelse(STAD$Segment_Mean > gc.thresh, 1, 0)
    STAD$loss <- ifelse(STAD$Segment_Mean < (-1) * gc.thresh, 1, 0)

    all_samples <- as.character(unique(STAD$Sample))
    aliquot <- sapply(all_samples, function(x) strsplit(x, split = '-')[[1]][4])
    target_samples <- all_samples[aliquot %in% c('01A', '01B', '02A')]

    STAD_target <- STAD[STAD$Sample %in% target_samples, ]
    STAD_list <- split(STAD_target, STAD_target$Sample)

    evaluate_sample <- function(samp_id, type) {
        df <- STAD_list[[samp_id]]
        if (is.null(df) || nrow(df) == 0) {
            out <- rep(0, length(bins_gr))
            names(out) <- bins_gr$bin
            return(out)
        }
        gr <- makeGRangesFromDataFrame(df, keep.extra.columns = TRUE)

        if (type == "loss") {
            return(fast_vectorize(bins_gr, gr[gr$loss == 1], 0.9))
        } else {
            return(fast_vectorize(bins_gr, gr[gr$gain == 1], 0.9))
        }
    }

    n_cores <- 16
    loss_list <- mclapply(target_samples, evaluate_sample, type="loss", mc.cores = n_cores)
    gain_list <- mclapply(target_samples, evaluate_sample, type="gain", mc.cores = n_cores)

    total_n <- length(target_samples)
    STAD.loss <- Reduce("+", loss_list) / total_n
    STAD.gain <- Reduce("+", gain_list) / total_n

    return(list(STAD.loss, STAD.gain))
}

