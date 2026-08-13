args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_path <- gsub("~\\+~", " ", script_path, fixed = FALSE)
script_dir <- dirname(normalizePath(script_path))
setwd(working_dir)

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(cowplot)
  library(ggsci)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'or_function.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

GC_ind1_pred$source_key  = 'ind-Zhejiang'
GC_ind2_pred$source_key  = 'ind-Shandong'
GC_ind3_pred$source_key  = 'ind-Henan'
GC_ind_pred <- as.data.frame(rbind(GC_ind1_pred, GC_ind2_pred, GC_ind3_pred))

cutoff_spe90 = Cutoff( 0.90, GC_trncv_pred)

GC_test2_pred = GC_ind_pred[GC_ind_pred$source_key=='ind-Zhejiang',]
GC_test3_pred_s1 = GC_ind_pred[GC_ind_pred$source_key=='ind-Henan',]
GC_test3_pred_s2 = GC_ind_pred[GC_ind_pred$source_key=='ind-Shandong',]

roc_ind1 <- pROC::roc(GC_test2_pred$Target, GC_test2_pred$final_prob, levels = c(0, 1), percent = TRUE)
roc_ind2 <- pROC::roc(GC_test3_pred_s1$Target, GC_test3_pred_s1$final_prob, levels = c(0, 1), percent = TRUE)
roc_ind3 <- pROC::roc(GC_test3_pred_s2$Target, GC_test3_pred_s2$final_prob, levels = c(0, 1), percent = TRUE)
set.seed(1234)
roc_ind1_random <- pROC::roc(sample(GC_test2_pred$Target), GC_test2_pred$final_prob, levels = c(0, 1), percent = TRUE)
roc_ind2_random <- pROC::roc(sample(GC_test3_pred_s1$Target), GC_test3_pred_s1$final_prob, levels = c(0, 1), percent = TRUE)
roc_ind3_random <- pROC::roc(sample(GC_test3_pred_s2$Target), GC_test3_pred_s2$final_prob, levels = c(0, 1), percent = TRUE)
random_curve_df <- function(roc_obj) {
  data.frame(Sp_inv = 100 - roc_obj$specificities, Sens = roc_obj$sensitivities)
}

p2 <- plot_auc_panel(
  list(
    get_roc_curve_info(roc_ind1, "ZHEJIANG", "#293E90", "#293E9014"),
    get_roc_curve_info(roc_ind2, "ANYANG", "#478AC9", "#478AC914"),
    get_roc_curve_info(roc_ind3, "SHANDONG", "#0097A6FF", "#0097A633")
  ),
  title = NULL,
  label_y = c(24, 18, 12, 6)
) +
  annotate("text", x = 100, y = 30, label = "Independent cohorts:", color = "black", size = 8 / .pt, hjust = 1) +
  geom_path(data = random_curve_df(roc_ind1_random), aes(x = Sp_inv, y = Sens), color = rgb(128, 128, 128, 120, maxColorValue = 255), linewidth = 0.4) +
  geom_path(data = random_curve_df(roc_ind2_random), aes(x = Sp_inv, y = Sens), color = rgb(128, 128, 128, 120, maxColorValue = 255), linewidth = 0.4) +
  geom_path(data = random_curve_df(roc_ind3_random), aes(x = Sp_inv, y = Sens), color = rgb(128, 128, 128, 120, maxColorValue = 255), linewidth = 0.4) +
  annotate("text", x = 100, y = 0, label = "Random Classifiers", color = rgb(128, 128, 128, 180, maxColorValue = 255), size = 6 / .pt, hjust = 1)

GC_ind1_pred2 = getinfo(GC_test2_pred)
ind1 = get_sensitivity_inxspe(GC_ind1_pred2, cutoff_spe90)
ind1_spe = get_HDspecificity_inxspe(GC_ind1_pred2, cutoff_spe90)
ind1$classify = 'IND 1'; ind1_spe$classify = 'IND 1'

GC_ind2_pred2 = getinfo(GC_test3_pred_s1)
ind2 = get_sensitivity_inxspe(GC_ind2_pred2, cutoff_spe90)
ind2_spe = get_HDspecificity_inxspe(GC_ind2_pred2, cutoff_spe90)
ind2$classify = 'IND 2'; ind2_spe$classify = 'IND 2'

GC_ind3_pred2 = getinfo(GC_test3_pred_s2)
ind3 = get_sensitivity_inxspe(GC_ind3_pred2, cutoff_spe90)
ind3_spe = get_HDspecificity_inxspe(GC_ind3_pred2, cutoff_spe90)
ind3$classify = 'IND 3'; ind3_spe$classify = 'IND 3'

GC_ind123_pred2 = getinfo(as.data.frame(rbind(GC_test2_pred, GC_test3_pred_s1, GC_test3_pred_s2)))
ind123 = get_sensitivity_inxspe(GC_ind123_pred2, cutoff_spe90)
ind123_spe = get_HDspecificity_inxspe(GC_ind123_pred2, cutoff_spe90)
ind123$classify = 'IND 123'; ind123_spe$classify = 'IND 123'

cols_ind = c(
  adjustcolor(ggsci::pal_material("blue-grey")(10)[3], alpha.f = 0.6),
  adjustcolor("#F39B7FFF", alpha.f = 0.6)
)
GC_ind = as.data.frame(rbind(GC_ind1_pred2, GC_ind2_pred2, GC_ind3_pred2))
GC_ind$Group = factor(GC_ind$Group, levels=c('Non-GC','GC'))
GC_ind$dataset_label = factor(GC_ind$dataset_label, levels=c('ZHEJIANG','ANYANG','SHANDONG'))
GC_indtest1 = GC_ind; GC_indtest1$Group_plot = GC_indtest1$Group
GC_indtest2_1 = GC_ind[GC_ind$dataset_label=='ZHEJIANG', ]; GC_indtest2_1$Group_plot = paste(GC_indtest2_1$Target, GC_indtest2_1$dataset_label, sep = '_')
GC_indtest2_2 = GC_ind[GC_ind$dataset_label=='ANYANG', ]; GC_indtest2_2$Group_plot = paste(GC_indtest2_2$Target, GC_indtest2_2$dataset_label, sep = '_')
GC_indtest2_3 = GC_ind[GC_ind$dataset_label=='SHANDONG', ]; GC_indtest2_3$Group_plot = paste(GC_indtest2_3$Target, GC_indtest2_3$dataset_label, sep = '_')
GC_indtest_all = as.data.frame(rbind(GC_indtest2_1, GC_indtest2_2, GC_indtest2_3))
GC_indtest_all$dataset_label = factor(GC_indtest_all$dataset_label, levels=c('ZHEJIANG','ANYANG','SHANDONG'))
GC_indtest_all$Group = factor(GC_indtest_all$Group, levels=c('Non-GC','GC'))
GC_indtest_all$final_prob = as.numeric(GC_indtest_all$final_prob)

p1_source_scores <- ggplot(data = GC_indtest_all, aes(x = dataset_label, y = final_prob, fill = Group)) +
  geom_violin(lwd = 0.3, position = position_dodge(width = 0.75), alpha = 0.9) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.3, dodge.width = 0.75), size = 0.05, color = "black") +
  geom_hline(yintercept = cutoff_spe90, color = 'red4', linetype = 'dashed', size = 0.5) +
  scale_fill_manual(values = cols_ind) +
  theme_sig +
  theme(
    legend.position = "right",             
    legend.justification = "bottom",       
    legend.title = element_blank(),
    legend.key.size = unit(0.3, "cm"),
    legend.background = element_blank(),
    legend.key = element_blank(),          
    legend.text = element_text(color = "black", size = 6),
    axis.title.x = element_blank()
  ) +
  ylim(0, 1) +ylab('rbcDNA predictive scores')

ind = as.data.frame(rbind(ind1, ind2, ind3))
ind_1 = ind[which(ind$Var1=='ANYANG'|ind$Var1=='SHANDONG'|ind$Var1=='ZHEJIANG'),c('Var1','SEN','SEN.low','SEN.up','perc')]
ind_1$label = 'Sensitivity\nat locked cutoff'
colnames(ind_1) = gsub('SEN','value',colnames(ind_1))

ind_spe = as.data.frame(rbind(ind1_spe, ind2_spe, ind3_spe))
ind_spe_1 = ind_spe[which(ind_spe$Var1=='ANYANG'|ind_spe$Var1=='SHANDONG'|ind_spe$Var1=='ZHEJIANG'),c('Var1','SPE','SPE.low','SPE.up','perc')]
ind_spe_1$label = 'Specificity\nat locked cutoff'
colnames(ind_spe_1) = gsub('SPE','value',colnames(ind_spe_1))

ind_sen_spe = as.data.frame(rbind(ind_1, ind_spe_1))
ind_sen_spe$label = factor(ind_sen_spe$label, levels = c('Sensitivity\nat locked cutoff', 'Specificity\nat locked cutoff'))

g1_source_performance <- ggplot(ind_sen_spe, aes(x=Var1, y=value, fill=Var1)) +
      geom_bar(stat="identity", color="black", position=position_dodge(), alpha=0.8, linewidth = 0.4) +
      geom_errorbar(aes(ymin=value.low, ymax=value.up), width=.2, position=position_dodge(.9), linewidth = 0.4) +
      geom_text(aes(label=perc), vjust=3, color="white", position = position_dodge(.9), size = 6 / .pt)+
      scale_fill_manual(values = c("#293E90","#478AC9","#0097A6")) + facet_grid(.~label, switch = "x")+
      theme_bar + theme(strip.background=element_blank(),
            axis.text.x = element_blank(), axis.title.x = element_blank(), axis.ticks.x = element_blank()) +
      ylab('Performance (%)')

g_abc = plot_grid(p2, p1_source_scores, g1_source_performance, ncol=3, 
            rel_widths=c(1,1.1,1.1), labels=c('A','B','C'), label_size = 12,label_fontface = "bold",label_x = 0.01,label_y = 1.01,hjust = 0,vjust = 0)

ind[ind$SEN.low<0, 'SEN.low'] = 0
ind[ind$SEN.up>100, 'SEN.up'] = 100
ind$Var1 <- as.character(ind$Var1)
ind$Var1[ind$Var1 == 'earlyGC'] <- 'Stage I\nGC'
ind$Var1[ind$Var1 == 'advGC'] <- 'Stage II-III\nGC'
ind$Var1 <- factor(ind$Var1, levels=c('Total','Stage I\nGC','I','Stage II-III\nGC','II','III','Intestinal','Diffuse','Mix','Missing','ZHEJIANG','ANYANG','SHANDONG'))
ind$classify = factor(ind$classify, levels=c('IND 1','IND 2','IND 3'))

ind123[ind123$SEN.low<0, 'SEN.low'] = 0
ind123[ind123$SEN.up>100, 'SEN.up'] = 100
ind123$Var1 <- as.character(ind123$Var1)
ind123$Var1[ind123$Var1 == 'earlyGC'] <- 'Stage I\nGC'
ind123$Var1[ind123$Var1 == 'advGC'] <- 'Stage II-III\nGC'
ind123$Var1 <- factor(ind123$Var1, levels=c('Total','Stage I\nGC','I','Stage II-III\nGC','II','III','Intestinal','Diffuse','Mix','Missing','ZHEJIANG','ANYANG','SHANDONG'))

ind_spe[ind_spe$SPE.low<0, 'SPE.low'] = 0
ind_spe[ind_spe$SPE.up>100, 'SPE.up'] = 100
ind_spe$Var1 <- as.character(ind_spe$Var1)
ind_spe$Var1[ind_spe$Var1 == 'Atr_No'] <- 'Non-atrophic\n(CSG)'
ind_spe$Var1[ind_spe$Var1 == 'Atr_Yes'] <- 'Atrophic\n(CAG)'
ind_spe$Var1[ind_spe$Var1 == 'IM_No'] <- 'without IM'
ind_spe$Var1[ind_spe$Var1 == 'IM_Yes'] <- 'with IM'
ind_spe$Var1 <- factor(ind_spe$Var1, levels=c('Total','Non-atrophic\n(CSG)','Atrophic\n(CAG)','Atr_Unknown','without IM','with IM','IM_Unknown','HP_No','HP_Yes','HP_Unknown','ZHEJIANG','ANYANG','SHANDONG'))
ind_spe$classify = factor(ind_spe$classify, levels=c('IND 1','IND 2','IND 3'))

ind123_spe[ind123_spe$SPE.low<0, 'SPE.low'] = 0
ind123_spe[ind123_spe$SPE.up>100, 'SPE.up'] = 100
ind123_spe$Var1 <- as.character(ind123_spe$Var1)
ind123_spe$Var1[ind123_spe$Var1 == 'Atr_No'] <- 'Non-atrophic\n(CSG)'
ind123_spe$Var1[ind123_spe$Var1 == 'Atr_Yes'] <- 'Atrophic\n(CAG)'
ind123_spe$Var1[ind123_spe$Var1 == 'IM_No'] <- 'without IM'
ind123_spe$Var1[ind123_spe$Var1 == 'IM_Yes'] <- 'with IM'
ind123_spe$Var1 <- factor(ind123_spe$Var1, levels=c('Total','Non-atrophic\n(CSG)','Atrophic\n(CAG)','Atr_Unknown','without IM','with IM','IM_Unknown','HP_No','HP_Yes','HP_Unknown','ZHEJIANG','ANYANG','SHANDONG'))

plot_locked_cutoff_bar <- function(dat, vars, y_col, low_col, up_col,
                                   fill_values, plot_theme, ylab_text,
                                   text_color) {
  ggplot(dat[dat$Var1 %in% vars, ], aes(x = Var1, y = .data[[y_col]], fill = classify)) +
      geom_bar(stat = "identity", color = "black", position = position_dodge(), alpha = 0.8, linewidth = 0.4) +
      geom_errorbar(aes(ymin = .data[[low_col]], ymax = .data[[up_col]]), width = .2, position = position_dodge(.9), linewidth = 0.4) +
      geom_text(aes(label = perc), vjust = 3, color = text_color, position = position_dodge(.9), size = 6 / .pt) +
      ylim(0, 100) +
      scale_fill_manual(values = fill_values) +
      plot_theme +
      ylab(ylab_text) +
      xlab('')
}

ind_cols <- c("#293E90", "#478AC9", "#0097A6")
total_sen_cols <- rep("#F39B7F", 3)
total_spe_cols <- rep("#A0B1BA", 3)
stage_vars <- c('Stage I\nGC', 'Stage II-III\nGC')
lauren_vars <- c('Intestinal', 'Diffuse', 'Mix')
atrophic_vars <- c('Non-atrophic\n(CSG)', 'Atrophic\n(CAG)')
im_vars <- c('without IM', 'with IM')

g1_totalind_stage <- plot_locked_cutoff_bar(ind123, stage_vars, 'SEN', 'SEN.low', 'SEN.up',
                                            total_sen_cols, theme_bar, 'Sensitivity at locked cutoff (%)', 'black')
g1_spe90_stage <- plot_locked_cutoff_bar(ind, stage_vars, 'SEN', 'SEN.low', 'SEN.up',
                                         ind_cols, theme_bar1, '', 'white')
g1_total_lauren <- plot_locked_cutoff_bar(ind123, lauren_vars, 'SEN', 'SEN.low', 'SEN.up',
                                          total_sen_cols, theme_bar, 'Sensitivity at locked cutoff (%)', 'black')
g1_spe90_lauren <- plot_locked_cutoff_bar(ind, lauren_vars, 'SEN', 'SEN.low', 'SEN.up',
                                          ind_cols, theme_bar1, '', 'white')
g1_totalind_atrspe <- plot_locked_cutoff_bar(ind123_spe, atrophic_vars, 'SPE', 'SPE.low', 'SPE.up',
                                             total_spe_cols, theme_bar, 'Specificity at locked cutoff (%)', 'black')
g1_spe90_atr <- plot_locked_cutoff_bar(ind_spe, atrophic_vars, 'SPE', 'SPE.low', 'SPE.up',
                                       ind_cols, theme_bar1, '', 'white')
g1_totalind_imspe <- plot_locked_cutoff_bar(ind123_spe, im_vars, 'SPE', 'SPE.low', 'SPE.up',
                                            total_spe_cols, theme_bar, 'Specificity at locked cutoff (%)', 'black')
g1_spe90_im <- plot_locked_cutoff_bar(ind_spe, im_vars, 'SPE', 'SPE.low', 'SPE.up',
                                      ind_cols, theme_bar1, '', 'white')


aligned_p1 <- cowplot::align_plots(g1_totalind_stage,g1_spe90_stage,g1_total_lauren,g1_spe90_lauren,align = "h",axis = "tblr")
p1 <- plot_grid(aligned_p1[[1]], aligned_p1[[2]], aligned_p1[[3]], aligned_p1[[4]],
                nrow = 1, ncol = 4, rel_widths = c(1, 2, 1.33, 3),
                labels = c("D", "", "E", ""), label_size = 12, label_fontface = "bold",
                label_x = 0.005, label_y = 1.02, hjust = 0, vjust = 1,
                align = "h", axis = "tblr")

aligned_p2 <- cowplot::align_plots(g1_totalind_atrspe,g1_spe90_atr,g1_totalind_imspe,g1_spe90_im,align = "h",axis = "tblr")
p2 <- plot_grid(aligned_p2[[1]], aligned_p2[[2]], aligned_p2[[3]], aligned_p2[[4]],
                nrow = 1, ncol = 4, rel_widths = c(1.3, 2, 1.3, 2),
                labels = c("F", "", "G", ""), label_size = 12, label_fontface = "bold",
                label_x = 0.005, label_y = 1.02, hjust = 0, vjust = 1,
                align = "h", axis = "tblr")

g_abc <- g_abc + theme(plot.margin = margin(t = 14, unit = "pt"))
p1   <- p1   + theme(plot.margin = margin(t = 14, unit = "pt"))
p2   <- p2   + theme(plot.margin = margin(t = 14, unit = "pt"))

ggsave(file.path(out_dir, 'Figure4.pdf'), plot_grid(g_abc,p1,p2,ncol=1), width=8, height=8.2)
