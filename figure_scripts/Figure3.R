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
  library(dplyr)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'or_function.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

## cutoff determined
cutoff_spe90 = Cutoff( 0.90, GC_trncv_pred)

## GC model performance
roc_trn <- pROC::roc(GC_trncv_pred$Target, GC_trncv_pred$final_prob, levels = c(0, 1), percent = TRUE)
roc_test <- pROC::roc(GC_test1_pred$Target, GC_test1_pred$final_prob, levels = c(0, 1), percent = TRUE)
set.seed(1234)
roc_trn_random <- pROC::roc(sample(GC_trncv_pred$Target), GC_trncv_pred$final_prob, levels = c(0, 1), percent = TRUE)
set.seed(1234)
roc_test_random <- pROC::roc(sample(GC_test1_pred$Target), GC_test1_pred$final_prob, levels = c(0, 1), percent = TRUE)
random_curve_df <- function(roc_obj) {
  data.frame(Sp_inv = 100 - roc_obj$specificities, Sens = roc_obj$sensitivities)
}
p1_trncv <- plot_auc_panel(
  list(get_roc_curve_info(roc_trn, "rbcDNA", "#BE202E", rgb(190, 32, 46, 20, maxColorValue = 255))),
  title = NULL,
  label_y = c(14, 8)
) +
  annotate("text", x = 100, y = 20, label = "Discovery cohort:", color = "black", size = 7 / .pt, hjust = 1) +
  geom_path(data = random_curve_df(roc_trn_random), aes(x = Sp_inv, y = Sens), color = rgb(128, 128, 128, 120, maxColorValue = 255), linewidth = 0.4) +
  annotate("text", x = 100, y = 2, label = "Random Classifiers", color = rgb(128, 128, 128, 180, maxColorValue = 255), size = 6 / .pt, hjust = 1)

p2_test <- plot_auc_panel(
  list(get_roc_curve_info(roc_test, "rbcDNA", "#512CA7FF", rgb(81, 44, 167, 20, maxColorValue = 255))),
  title = NULL,
  label_y = c(14, 8)
) +
  annotate("text", x = 100, y = 20, label = "Test cohort:", color = "black", size = 7 / .pt, hjust = 1) +
  geom_path(data = random_curve_df(roc_test_random), aes(x = Sp_inv, y = Sens), color = rgb(128, 128, 128, 120, maxColorValue = 255), linewidth = 0.4) +
  annotate("text", x = 100, y = 2, label = "Random Classifiers", color = rgb(128, 128, 128, 180, maxColorValue = 255), size = 6 / .pt, hjust = 1)

## rbcDNA predictive scores (trncv)
GC_trncv_pred_Final_model = merge(GC_trncv_pred, sampleinfo[,c('Sample','Group')], by='Sample')
GC_indtest1 = GC_trncv_pred_Final_model[, c('Sample','Target','final_prob','Group')]; GC_indtest1$plot_group = 'All'; colnames(GC_indtest1)[4] = 'Group_plot'
GC_indtest1$Group_plot = factor(GC_indtest1$Group_plot, levels = c("Non-GC", "GC"))
## total sample + Group_plot sample
p_trncv <- ggplot(data = GC_indtest1, aes(x = Group_plot, y = final_prob , fill=Group_plot)) + ##
        geom_rect(xmin = 0.5, xmax = 1.5, ymin = 0, ymax = 1, fill = "#F0F0F0", alpha = 0.2) +
        geom_boxplot(outlier.shape = NA, outlier.color = NA, lwd=0.1)+
        geom_jitter(width = 0.3, size = 0.05)+#
        geom_hline(yintercept=cutoff_spe90, color='red4', linetype='dashed', size=0.5)+
        scale_fill_manual(values=c(ggsci::pal_material("blue-grey")(10)[5], "#9F1A1AFF",ggsci::pal_material("red")(10)[1:3],
                                   ggsci::pal_material("deep-purple")(10)[2:5]))+
        annotate("text", x = 2.5, y = cutoff_spe90, label = "Cutoff:\n90% spec.", hjust = 0, vjust = 0.5, size = 5 / .pt, lineheight = 0.9) +
        coord_cartesian(ylim = c(0, 1), clip = "off") +
        theme_sig + theme(plot.margin = margin(t = 5, r = 30, b = 5, l = 5)) + labs(y = 'rbcDNA predictive scores')

## rbcDNA predictive scores (test)
GC_test1_pred_Final_model <- GC_test1_pred %>%
      left_join(sampleinfo %>% select(Sample, Group, Stage, Atrophic, IntestinalMetaplasia), by = "Sample")
GC_indtest_all <- bind_rows(
      GC_test1_pred_Final_model %>% transmute(Sample, Target, final_prob, Group_plot = Group, plot_group = "All"),
      GC_test1_pred_Final_model %>% filter(Target == 0) %>% transmute(Sample, Target, final_prob, Group_plot = recode(Atrophic, "No" = "Non-atrophic", "Yes" = "Atrophic"), plot_group = "sub"),
      GC_test1_pred_Final_model %>% filter(Target == 0) %>% transmute(Sample, Target, final_prob, Group_plot = recode(IntestinalMetaplasia, "No" = "withoutIM", "Yes" = "Gastritis\nwith IM"), plot_group = "sub"),
      GC_test1_pred_Final_model %>% filter(Target == 1) %>% transmute(Sample, Target, final_prob, Group_plot = case_when(Stage == "I" ~ "Stage I", Stage %in% c("II", "III") ~ "Stage II-III", TRUE ~ NA_character_), plot_group = "sub")
) %>% mutate(
          final_prob = as.numeric(final_prob),
          Group_plot = factor(Group_plot, levels = c("Non-GC", "GC", "Non-atrophic", "Atrophic", "withoutIM", "Gastritis\nwith IM", "Stage I", "Stage II-III"))
      ) %>%
      filter(!is.na(Group_plot)) %>% droplevels()

GC_indtest_all_select = GC_indtest_all[which(GC_indtest_all$Group_plot %in% c('Non-atrophic','Atrophic','Gastritis\nwith IM','Stage I','Stage II-III')), ]
GC_indtest_all_select$Group_plot = factor(GC_indtest_all_select$Group_plot, levels = c('Non-atrophic','Atrophic','Gastritis\nwith IM','Stage I','Stage II-III'))
group <- factor(GC_indtest_all_select$Group_plot, levels = c('Non-atrophic','Atrophic','Gastritis\nwith IM','Stage I','Stage II-III'), ordered = TRUE)
pval = clinfun::jonckheere.test(GC_indtest_all_select$final_prob, group, alternative = "increasing")$p.value
if(pval < 0.01){
      print("Jonckheere-Terpstra test\nP < 0.01")
      anno_p <- data.frame(plot_group = "sub", x_pos = 3, y_pos = 1.07, size = 4, label = "Jonckheere-Terpstra test\nP < 0.01")
}

p_test <- ggplot(data = GC_indtest_all[which(GC_indtest_all$Group_plot!='withoutIM'), ], aes(x = Group_plot, y = final_prob , fill=Group_plot)) + ##
        geom_rect(xmin = 0, xmax = 2.5, ymin = 0, ymax = 1, fill = "#FFFFFF", alpha = 0.2) +
        geom_boxplot(outlier.shape = NA, outlier.color = NA, lwd=0.1)+
        geom_jitter(width = 0.3, size = 0.05)+#
        geom_hline(yintercept=cutoff_spe90, color='red4', linetype='dashed', size=0.5)+
        geom_vline(xintercept = 3.5, color='grey40', linetype='dotted', size=0.5) +
        scale_fill_manual(values=c(ggsci::pal_material("blue-grey")(10)[5], "#9F1A1AFF", ggsci::pal_material("brown")(10)[1],
                                    ggsci::pal_material("deep-orange")(10)[c(1,2)], ggsci::pal_material("red")(10)[c(3,5)]))+
        labs(y = 'rbcDNA predictive scores')+
        facet_grid(.~plot_group, scales = "free_x", space = "free_x", switch = "x")+
        scale_x_discrete(drop = TRUE)+theme_sig + theme(strip.placement = "outside") +
        scale_y_continuous(limits = c(0, 1.08), breaks = seq(0, 1, 0.25)) +
        geom_text(data = anno_p, aes(x = x_pos, y = y_pos, label = label),
                  inherit.aes = FALSE, size = 2, fontface = "italic", lineheight = 0.8)



squash_plot <- function(p) {
    plot_grid(p, NULL, ncol=1, rel_heights=c(1, 0.20))
}

p1_trncv_sq <- squash_plot(p1_trncv)
p_trncv_sq  <- squash_plot(p_trncv)
p2_test_sq  <- squash_plot(p2_test)
row1 = plot_grid(p1_trncv_sq, p_trncv_sq, p2_test_sq, p_test, ncol=4, rel_widths=c(1,0.55,1,0.9),labels=c("B", "", "C", "D"), 
                 label_size = 12,label_fontface = "bold",label_x = 0.01,label_y = 1.02,hjust = 0,vjust = 1)

## sensitivity at 90% specificity
GC_trncv_pred2 = getinfo(GC_trncv_pred)
trn = get_sensitivity_inxspe(GC_trncv_pred2, cutoff_spe90)
trn_spe = get_HDspecificity_inxspe(GC_trncv_pred2, cutoff_spe90)
trn$classify = 'Discovery'; trn_spe$classify = 'Discovery'

GC_test1_pred2 = getinfo(GC_test1_pred)
test1 = get_sensitivity_inxspe(GC_test1_pred2, cutoff_spe90)
test1_spe = get_HDspecificity_inxspe(GC_test1_pred2, cutoff_spe90)
test1$classify = 'Test'; test1_spe$classify = 'Test'

all = as.data.frame(rbind(trn, test1))
all[all$SEN.low<0, 'SEN.low'] = 0
all[all$SEN.up>100, 'SEN.up'] = 100
all$Var1 <- as.character(all$Var1)
all$Var1[all$Var1 == 'earlyGC'] <- 'Stage I'
all$Var1[all$Var1 == 'advGC'] <- 'Stage II-III'
all$Var1 = factor(all$Var1, levels=c('Total','Stage I','I','Stage II-III','II','III','Intestinal','Diffuse','Mix','Missing'))
all$classify = factor(all$classify, levels=c('Discovery','Test'))

all_spe = as.data.frame(rbind(trn_spe, test1_spe))
all_spe[all_spe$SPE.low<0, 'SPE.low'] = 0
all_spe[all_spe$SPE.up>100, 'SPE.up'] = 100
all_spe$Var1 <- as.character(all_spe$Var1)
all_spe$Var1[all_spe$Var1 == 'Atr_No'] <- 'Non-atrophic\n(CSG)'
all_spe$Var1[all_spe$Var1 == 'Atr_Yes'] <- 'Atrophic\n(CAG)'
all_spe$Var1[all_spe$Var1 == 'IM_No'] <- 'without IM'
all_spe$Var1[all_spe$Var1 == 'IM_Yes'] <- 'with IM'
all_spe$Var1 = factor(all_spe$Var1, levels=c('Total','Non-atrophic\n(CSG)','Atrophic\n(CAG)','Atr_Unknown','without IM','with IM','IM_Unknown','HP_No','HP_Yes','HP_Unknown','trn','test'))
all_spe$classify = factor(all_spe$classify, levels=c('Discovery','Test'))

plot_perf_bar <- function(dat, vars, y_col, low_col, up_col, plot_theme, ylab_text, title_text) {
      ggplot(dat[dat$Var1 %in% vars, ], aes(x = Var1, y = .data[[y_col]], fill = classify)) +
            geom_bar(stat = "identity", color = "black", position = position_dodge(), alpha = 0.8) +
            geom_errorbar(aes(ymin = .data[[low_col]], ymax = .data[[up_col]]), width = .2, position = position_dodge(.9)) +
            geom_text(aes(label = perc), vjust = 3, color = "white", position = position_dodge(.9), size = 1.6) +
            scale_fill_manual(name = "", values = c("#A50F15", "#8491B4B2")) +
            plot_theme +
            ylim(0, 100) +
            labs(x = NULL, y = ylab_text, title = title_text)
}

g1_spe90_total <- plot_perf_bar(all, 'Total', 'SEN', 'SEN.low', 'SEN.up',
                                theme_bar, 'Sensitivity at 90% specificity (%)', 'GC group')
g1_spe90_stage <- plot_perf_bar(all, c('Stage I', 'Stage II-III'), 'SEN', 'SEN.low', 'SEN.up',
                                theme_bar1, NULL, 'Stage')
g1_spe90_lauren <- plot_perf_bar(all, c('Intestinal', 'Diffuse', 'Mix'), 'SEN', 'SEN.low', 'SEN.up',
                                 theme_bar1, NULL, 'Lauren subtype')
g1_spe_total <- plot_perf_bar(all_spe, 'Total', 'SPE', 'SPE.low', 'SPE.up',
                              theme_bar, 'Specificity (%)', 'Non-GC group')
g1_spe90_atrophic <- plot_perf_bar(all_spe, c('Atrophic\n(CAG)', 'Non-atrophic\n(CSG)'), 'SPE', 'SPE.low', 'SPE.up',
                                   theme_bar1, NULL, 'Atrophic')
g1_spe90_IM <- plot_perf_bar(all_spe, c('with IM', 'without IM'), 'SPE', 'SPE.low', 'SPE.up',
                             theme_bar1, NULL, 'Intestinal metaplasia')

shared_legend <- ggpubr::get_legend(
      g1_spe90_IM + theme(legend.position='bottom', legend.title=element_blank(), legend.direction="horizontal",
                          legend.background = element_rect(fill = NA, color = NA),
                          legend.box.background = element_rect(fill = NA, color = NA)))

aligned_p2 <- cowplot::align_plots(g1_spe90_total,g1_spe90_stage,g1_spe90_lauren,
    g1_spe_total,g1_spe90_atrophic,g1_spe90_IM,align = "h",axis = "tblr")

p2 <- plot_grid(aligned_p2[[1]],aligned_p2[[2]],aligned_p2[[3]],aligned_p2[[4]],aligned_p2[[5]],aligned_p2[[6]],
                  nrow = 1,ncol = 6,rel_widths = c(1.667, 2.05, 2.92, 1.667, 2.05, 2.05),labels = c("E", "F", "", "G", "H", ""),
                  label_size = 12,label_fontface = "bold",label_x = 0.005,label_y = 1.02,hjust = 0,vjust = 1,align = "h",axis = "tblr")

row1 <- row1 + theme(plot.margin = margin(t = 10, unit = "pt"))
p2   <- p2   + theme(plot.margin = margin(t = 10, unit = "pt"))

Fig3 = plot_grid(row1, p2, shared_legend, ncol=1, rel_heights=c(1.1,1,0.2))
ggsave(file.path(out_dir, 'Figure3.pdf'), Fig3, width=8, height=6, device = cairo_pdf)
