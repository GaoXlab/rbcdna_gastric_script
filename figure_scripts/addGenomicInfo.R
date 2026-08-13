library(BSgenome.Hsapiens.UCSC.hg38)
library(rtracklayer)
library(tidyverse)
library(httr)

add_GenomicAnnotation_userbed <- function(userbed, path, name){
    ### add to userbed:
    if(length(grep('chr',userbed$chromosome))==0){
        userbed$chromosome = str_c('chr',userbed$chromosome)
    }
    userbed = GRanges(userbed[, setdiff(colnames(userbed),'feature')])

    ### find overlaps to the cytoband
    userbed$arm <- arms$type[(findOverlaps(userbed, arms, select='first'))] 
    userbed$cyto <- cyto.hg38$type[(findOverlaps(userbed, cyto.hg38, select='first'))]
    userbed$CFS <- cfs.hg38$type[(findOverlaps(userbed, cfs.hg38, select='first'))]
    userbed$ERFS <- erfs.hg38$type[(findOverlaps(userbed, erfs.hg38, select='first'))]

    ### find overlaps to the known genes
    overlap_with_genes <- findOverlaps(userbed, knownGene.hg38)
    userbed <- as.data.frame(userbed); userbed$id=1:dim(userbed)[1]
    userbed$region <- str_c(userbed$seqnames,':',userbed$start,'-',userbed$end)

    # gene region
    overlap_with_genes_df <- as.data.frame(overlap_with_genes)
    knownGene.hg38_df <- as.data.frame(knownGene.hg38); knownGene.hg38_df$id=1:dim(knownGene.hg38_df)[1]
    knownGene.hg38_df <- knownGene.hg38_df[-grep('ENSG00', knownGene.hg38_df$genename), ]
    overlap_with_genes_df_merge1 <- merge(overlap_with_genes_df, knownGene.hg38_df, by.x='subjectHits', by.y='id')
    userbed_merge_genes <- merge(userbed,overlap_with_genes_df_merge1, by.x='id', by.y='queryHits', all.x=T)
    userbed_merge <- do.call(rbind,lapply(unique(userbed_merge_genes$region),FUN = function(region,df){
        data.frame(region=region,
        chr=unique(df[df$region==region,'seqnames.x']),start=unique(df[df$region==region,'start.x']),end=unique(df[df$region==region,'end.x']),
        arm=unique(df[df$region==region,'arm']), cyto=unique(df[df$region==region,'cyto']), CFS=unique(df[df$region==region,'CFS']), ERFS=unique(df[df$region==region,'ERFS']),
        overlappedGeneNumDensity=length(unique(df[df$region==region,'genename'])),
        genename=paste0(unique(df[df$region==region,'genename']),',',collapse=''), 
        transcriptClass=paste0(unique(unlist(strsplit(df[df$region==region,'transcriptClass'],','))),',',collapse=''), 
        geneType=paste0(unique(unlist(strsplit(df[df$region==region,'geneType'],','))),',',collapse=''), 
        largeGene=paste0(unique(df[df$region==region,'largeGene']),',',collapse='')
    )},df=userbed_merge_genes))

    userbed_merge$genename = gsub('^NA,|^NA,$|\\,$','',userbed_merge$genename)
    userbed_merge$transcriptClass = gsub('^NA,|^NA,$|\\,$','',userbed_merge$transcriptClass)
    userbed_merge$geneType = gsub('^NA,|^NA,$|\\,$','',userbed_merge$geneType)
    userbed_merge$largeGene = gsub('^NA,|^NA,$|\\,$','',userbed_merge$largeGene)

    userbed_merge[is.na(userbed_merge$CFS), 'CFS'] = ''
    userbed_merge[is.na(userbed_merge$ERFS), 'ERFS'] = ''

    write.table(userbed_merge, str_c(path,'/GenomicAnnotationIn_',name,'_addtoRegion.log'), sep='\t', row.names=F, quote=F)
    return(userbed_merge)
}