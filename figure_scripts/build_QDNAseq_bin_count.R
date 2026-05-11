.libPaths('/home/gaoxiaofeiLab/yaoxingyun/RLib/3.6/')
library("QDNAseq")

args <- commandArgs(trailingOnly=TRUE)
bam_in <- args[1]
binsize <- as.numeric(args[2])
species <- args[3]
out_dir <- args[4]

library(paste0("QDNAseq.", species), character.only=TRUE)

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive=TRUE)
}

bins <- getBinAnnotations(binSize=binsize, genome=species)
readCounts <- binReadCounts(bins, bamfile=bam_in)
readCountsFiltered <- applyFilters(readCounts, residual=TRUE, blacklist=TRUE)
readCountsFiltered <- estimateCorrection(readCountsFiltered)
copyNumbers <- correctBins(readCountsFiltered)
copyNumbersNormalized <- normalizeBins(copyNumbers)
copyNumbersSmooth <- smoothOutlierBins(copyNumbersNormalized)

bam_basename <- sub("\\.[^.]*$", "", basename(bam_in))
out_file <- file.path(out_dir, paste0(bam_basename, ".", binsize, "kb_copyNumbersSmooth.txt"))

exportBins(copyNumbersSmooth, file=out_file, logTransform=FALSE)