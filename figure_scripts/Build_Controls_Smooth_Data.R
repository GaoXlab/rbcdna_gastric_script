.libPaths('/home/gaoxiaofeiLab/yaoxingyun/RLib/3.6/')

library(stringr)

ids_list <- read.table('Human_model/modelData/gc.neg.ids.txt', stringsAsFactors = FALSE)[, 1]

build_merged_smooth_data <- function(ids, binsize) {
  base_info <- NULL
  val_list <- list()

  for (id in ids) {
    file_path <- paste0("./Figures/QDNA_bin_results/", id, ".nodup.q30.", binsize, "kb_copyNumbersSmooth.txt")

    tmp_data <- read.table(file_path, header=TRUE, sep="\t", stringsAsFactors=FALSE)

    if (is.null(base_info)) {
      base_info <- tmp_data[, c("chromosome", "start", "end")]
      feature_col <- tmp_data$feature
      if (!all(grepl("^chr", feature_col))) {
        feature_col <- str_c("chr", feature_col)
      }
      base_info$feature <- feature_col
    }

    val_list[[id]] <- tmp_data[, 5]
  }

  val_df <- as.data.frame(val_list, check.names = FALSE)

  base_info$median <- apply(val_df, 1, median, na.rm=TRUE)
  base_info$min <- apply(val_df, 1, min, na.rm=TRUE)
  base_info$max <- apply(val_df, 1, max, na.rm=TRUE)

  return(base_info)
}

df_HD_1000k <- build_merged_smooth_data(ids_list, "1000")
df_HD_100k <- build_merged_smooth_data(ids_list, "100")

save(df_HD_1000k, df_HD_100k, file = './Figures/trn_HD_smooth.RData')

build_trn_100k_data <- function(ids) {
  # 读取外部索引文件
  index_path <- 'Human_model/modelData/trim_gcc_r100k_0start/sorted.tab.index'
  index_df <- read.table(index_path, header=FALSE, stringsAsFactors=FALSE, comment.char = '#')
  colnames(index_df) <- c('chr', 'start', 'end')

  row_names <- str_c(index_df$chr, ':', index_df$start, '-', index_df$end)
  rownames(index_df) <- row_names

  val_list <- list()

  for (id in ids) {
    file_path <- paste0("Human_model/modelData/trim_gcc_r100k_0start/cleaned/", id, ".raw")

    if (file.exists(file_path)) {
      tmp_val <- read.table(file_path, header=TRUE, stringsAsFactors=FALSE)

      if (nrow(tmp_val) == nrow(index_df)) {
        val_list[[id]] <- tmp_val[, 1]
      } else {
        warning(paste("行数不匹配，跳过样本:", id))
      }
    } else {
      warning(paste("找不到文件:", file_path))
    }
  }

  val_df <- as.data.frame(val_list, check.names = FALSE)
  trn_100k <- cbind(index_df, val_df)

  return(trn_100k)
}
ids_list <- read.table('Human_model/modelData/gc.trn.ids.txt', stringsAsFactors = FALSE)[, 1]

trn_100k <- build_trn_100k_data(ids_list)
save(trn_100k, file = './Figures/trn_100k.RData')

gDNA5_ids <- c('CRLG0006', 'CRLG0007', 'CRLG0008', 'CRLG0009', 'CRLG0010')

rbcDNA5_ids <- c('CRLR0006', 'CRLR0007', 'CRLR0008', 'CRLR0009', 'CRLR0010')

build_merged_smooth_data <- function(ids, binsize) {
  base_info <- NULL
  val_list <- list()

  for (id in ids) {
    file_path <- paste0("./Figures/QDNA_bin_results/", id, ".nodup.q30.", binsize, "kb_copyNumbersSmooth.txt")

    tmp_data <- read.table(file_path, header=TRUE, sep="\t", stringsAsFactors=FALSE)

    if (is.null(base_info)) {
      base_info <- tmp_data[, c("feature", "chromosome", "start", "end")]
      if (!all(grepl("^chr", base_info$feature))) {
        base_info$feature <- str_c("chr", base_info$feature)
      }
    }

    val_list[[id]] <- tmp_data[, 5]
  }

  val_df <- as.data.frame(val_list, check.names = FALSE)
  merged_data <- cbind(base_info, val_df)

  merged_data$median <- apply(merged_data[, ids], 1, median, na.rm=TRUE)
  merged_data$min <- apply(merged_data[, ids], 1, min, na.rm=TRUE)
  merged_data$max <- apply(merged_data[, ids], 1, max, na.rm=TRUE)

  return(merged_data)
}

HD5_gDNA_1000kb <- build_merged_smooth_data(gDNA5_ids, "1000")
HD5_rbcDNA_1000kb <- build_merged_smooth_data(rbcDNA5_ids, "1000")

HD5_gDNA_100kb <- build_merged_smooth_data(gDNA5_ids, "100")
HD5_rbcDNA_100kb <- build_merged_smooth_data(rbcDNA5_ids, "100")

save(HD5_gDNA_1000kb, HD5_rbcDNA_1000kb, HD5_gDNA_100kb, HD5_rbcDNA_100kb,
     file = './Figures/r_g_5controls_smooth.RData')