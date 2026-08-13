args <- commandArgs(trailingOnly = TRUE)
working_dir <- if (length(args) >= 1) gsub("~\\+~", " ", args[1]) else getwd()
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_path <- gsub("~\\+~", " ", script_path)
script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))
if (!file.exists(file.path(script_dir, "or_function.r"))) {
  script_dir <- file.path(working_dir, "figure_scripts")
}

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(pROC)
  library(ggplot2)
  library(cowplot)
  library(dplyr)
  library(ggpubr)
  library(ggsci)
  library(ggridges)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'or_function.r'), chdir = TRUE)

plot_auc_95CI_ind_gg <- function(Test1_df, Test2_df, Test3_df, label="", title_name=""){
	Test1_df$Target <- as.factor(Test1_df$Target)
	Test2_df$Target <- as.factor(Test2_df$Target)
	Test3_df$Target <- as.factor(Test3_df$Target)

	roc1 <- pROC::roc(Test1_df$Target, Test1_df$final_prob, levels = c(0, 1), percent = TRUE)
	roc2 <- pROC::roc(Test2_df$Target, Test2_df$final_prob, levels = c(0, 1), percent = TRUE)
	roc3 <- pROC::roc(Test3_df$Target, Test3_df$final_prob, levels = c(0, 1), percent = TRUE)

	curves <- list(
		get_roc_curve_info(roc1, "Discovery cohort", "#BE232F", "#BE232F20"),
		get_roc_curve_info(roc2, "Test cohort", "#543D98", "#543D9820"),
		get_roc_curve_info(roc3, "Independent cohort", "#488AC9", "#488AC920")
	)

	plot_auc_panel(curves, title = title_name, label_y = c(24, 18, 12, 6))
}

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

cutoff_spe90 <- Cutoff(0.90, GC_trncv_pred)

sampleinfo[sampleinfo$Dataset == 'Dataset A, discovery cohort', 'Dataset'] = 'Discovery\ncohort'
sampleinfo[sampleinfo$Dataset == 'Dataset A, test cohort', 'Dataset'] = 'Test\ncohort'
sampleinfo[(sampleinfo$Dataset == 'Dataset B') & (sampleinfo$Source == 'ZHEJIANG'), 'Dataset'] = 'Dataset B\n(ZHEJIANG)'
sampleinfo[(sampleinfo$Dataset == 'Dataset B') & (sampleinfo$Source == 'ANYANG'), 'Dataset'] = 'Dataset B\n(ANYANG)'
sampleinfo[(sampleinfo$Dataset == 'Dataset B') & (sampleinfo$Source == 'SHANDONG'), 'Dataset'] = 'Dataset B\n(SHANDONG)'

sampleinfo$Target = 0
sampleinfo[which(sampleinfo$Group=='GC'), 'Target'] = 1
sampleinfo$Dataset = factor(sampleinfo$Dataset, levels=c("Discovery\ncohort","Test\ncohort", "Dataset B\n(ZHEJIANG)","Dataset B\n(ANYANG)","Dataset B\n(SHANDONG)"))
sampleinfo$Group = factor(sampleinfo$Group, levels=c('Non-GC', 'GC'))

group_colors <- c("Non-GC" = ggsci::pal_material("blue-grey", alpha = 0.7)(10)[5], "GC" = "#9F1A1AFF")

make_auc_input <- function(var_name, dataset_filter) {
  out <- sampleinfo[dataset_filter, c('Sample', 'Target', var_name)]
  colnames(out)[3] <- 'final_prob'
  out
}
plot_density_auc_panel <- function(var_name, x_label, ridge_from = NULL, ridge_to = NULL, ref_line = NULL) {
  med <- sampleinfo %>%
    filter(Dataset == 'Discovery\ncohort', Group == "Non-GC") %>%
    summarise(med = median(.data[[var_name]], na.rm = TRUE))

  ridge_layer <- if (!is.null(ridge_from) || !is.null(ridge_to)) {
    stat_density_ridges(geom = "density_ridges", scale = 1.5, rel_min_height = 0.01,
                        linewidth = 0.4, alpha = 0.35, from = ridge_from, to = ridge_to)
  } else {
    stat_density_ridges(geom = "density_ridges", scale = 1.5, rel_min_height = 0.01,
                        linewidth = 0.4, alpha = 0.35)
  }

  p_density <- ggplot(sampleinfo[!is.na(sampleinfo$Dataset), ],
                      aes(x = .data[[var_name]], y = Group, fill = Group, color = Group)) +
    ridge_layer +
    scale_color_manual(values = group_colors) +
    scale_fill_manual(values = group_colors) +
    coord_flip(clip = "off") +
    facet_grid(. ~ Dataset) +
    labs(x = x_label, y = NULL) +
    geom_vline(xintercept = med$med, linetype = "dashed", linewidth = 0.3) +
    theme_sig
  if (!is.null(ref_line)) {
    p_density <- p_density + geom_vline(xintercept = ref_line, linetype = "dashed", color = 'grey')
  }

  p_auc <- plot_auc_95CI_ind_gg(
    make_auc_input(var_name, sampleinfo$Dataset == 'Discovery\ncohort'),
    make_auc_input(var_name, sampleinfo$Dataset == 'Test\ncohort'),
    make_auc_input(var_name, grepl('Dataset B', sampleinfo$Dataset)),
    title_name = x_label
  )
  plot_grid(p_density, p_auc, ncol = 2)
}

g1_mt <- plot_density_auc_panel(
  'chrM_percentage',
  "Proportion of rbcDNA\nmapped to MT regions (%)",
  ridge_from = 0,
  ridge_to = 0.04,
  ref_line = 0.04
)
g1_dep <- plot_density_auc_panel('depth', "Average sequencing depth")

row1 = plot_grid(g1_dep, g1_mt, 
        labels = c("A","","B",""), label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 0.98, hjust = 0, vjust = 0.5)

### 测序批次
prediction_all = rbind(GC_trncv_pred, GC_test1_pred, GC_ind1_pred, GC_ind2_pred, GC_ind3_pred)
prediction_all = prediction_all[, c('Sample', 'final_prob', 'source_key')]
prediction_sampleinfo = merge(sampleinfo, prediction_all, by = 'Sample')

p_lib_score_dataset_levels <- c('Discovery cohort', 'Test cohort', 'Independent validation sets')

p_lib_score_dat <- prediction_sampleinfo %>%
    filter(Dataset %in% c('Discovery\ncohort', 'Test\ncohort') | grepl('Dataset B', Dataset)) %>%
    mutate(
        Dataset2 = case_when(
            Dataset == 'Discovery\ncohort' ~ 'Discovery cohort',
            Dataset == 'Test\ncohort' ~ 'Test cohort',
            grepl('Dataset B', Dataset) ~ 'Independent validation sets',
            TRUE ~ as.character(Dataset)
        ),
        Dataset2 = factor(Dataset2, levels = p_lib_score_dataset_levels) 
    )

p_lib_score_label_mode <- 'signif' # 'pvalue' or 'signif'
p_to_sig_label <- function(p) {
    case_when(
        is.na(p) ~ 'NA',
        p < 0.001 ~ '***',
        p < 0.01 ~ '**',
        p < 0.05 ~ '*',
        TRUE ~ 'ns'
    )
}

p_lib_score_p <- p_lib_score_dat %>%
    group_by(Dataset2, Group) %>%
    summarise(
        p = if (n_distinct(library_batch) >= 2) {
            kruskal.test(final_prob ~ library_batch)$p.value
        } else {
            NA_real_
        },
        n_groups = n_distinct(library_batch),
        y.position = 1.04,
        .groups = 'drop'
    ) %>%
    mutate(
        label = if (p_lib_score_label_mode == 'signif') {
            paste0('Kruskal-Wallis, ', p_to_sig_label(p))
        } else {
            paste0('Kruskal-Wallis, p-value = ', scales::pvalue(p, accuracy = 0.001))
        },
        x.position = (n_groups + 1) / 2
    )

plot_lib_score_dataset <- function(dataset_label, show_y_axis = FALSE) {
    ggplot(
        filter(p_lib_score_dat, Dataset2 == dataset_label),
        aes(x = library_batch, y = final_prob)
    ) +
        geom_boxplot(fill = "grey85", outlier.shape = NA, width = 0.6, linewidth = 0.3, alpha = 0.7) +
        geom_jitter(width = 0.2, size = 0.3, alpha = 0.5, color = 'grey40', shape = 16) +
        geom_hline(yintercept = cutoff_spe90, color = 'red4', linetype = 'dashed', linewidth = 0.4) +
        labs(x = NULL, y = if (show_y_axis) 'rbcDNA predictive scores' else NULL, title = dataset_label) +
        theme_sig +
        theme(
            legend.position = 'none',
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.title.y = if (show_y_axis) element_text(color = "black", size = 8) else element_blank(),
            axis.text.y = if (show_y_axis) element_text(color = "black", size = 6) else element_blank(),
            axis.ticks.y = if (show_y_axis) element_line(linewidth = 0.3) else element_blank(),
            axis.line.y = if (show_y_axis) element_line(linewidth = 0.3) else element_blank(),
            plot.title = element_text(size = 8, hjust = 0.5), plot.margin = margin(1, 0.5, 1, 1)
        ) +
        facet_grid(. ~ Group, scales = 'free_x', space = 'free_x') + ylim(0, 1.05) +
        geom_text(
            data = filter(p_lib_score_p, Dataset2 == dataset_label),
            aes(x = x.position, y = y.position, label = label),
            inherit.aes = FALSE, size = 6 / .pt
        )
}

row2 <- plot_grid(
    plot_lib_score_dataset(p_lib_score_dataset_levels[1], show_y_axis = TRUE),
    plot_lib_score_dataset(p_lib_score_dataset_levels[2]),
    plot_lib_score_dataset(p_lib_score_dataset_levels[3]),
    ncol = 3,rel_widths = c(1, 0.8, 0.9),align = 'h',axis = 'tb',
    labels = c("C"),label_size = 12,label_fontface = "bold",label_x = 0.01,label_y = 0.98,hjust = 0,vjust = 0.5)

batch_groups = aggregate(prediction_sampleinfo$final_prob, list(prediction_sampleinfo$Group, prediction_sampleinfo$Dataset, prediction_sampleinfo$library_batch), median)
paired_lines <- batch_groups %>% group_by(Group.2, Group.3) %>% filter(n_distinct(Group.1)==2) %>% ungroup()

group_colors = c("Non-GC"=ggsci::pal_material("blue-grey", alpha=0.7)(10)[5], "GC"="#9F1A1AFF")
p_lib = ggplot(batch_groups, aes(x = Group.1, y = x, fill = Group.1)) +
    geom_line(data=paired_lines, aes(x=Group.1, y=x, group=Group.3), inherit.aes=FALSE, color="grey65", linewidth=0.35, alpha=0.7) +
	geom_boxplot(width = 0.7, outlier.shape = NA, color = '#333333', linewidth = 0.2, position = position_dodge(width = 0.78)) +
	geom_point(position = position_jitterdodge(jitter.width = 0.5, dodge.width = 0.78), size = 1, alpha = 1, stroke = 0) +
	scale_fill_manual(values = group_colors, drop = FALSE) +
	labs(x = NULL, y = 'Median rbcDNA predictive scores\nin each WGS library batch') +
	theme_sig + facet_grid(. ~ Group.2, scales = 'free_y') + ylim(0, 1.05) +
    stat_compare_means(aes(group = Group.1, label = ..p.signif..), method = 'wilcox.test', size = 6/.pt, label.y.npc = 1, label.x.npc = 0.5)

group_colors = c('NovaSeq 6000' = '#7E6148B2', 'NovaSeq X Plus' = '#B09C85B2')
prediction_sampleinfo$sequencing_platform = gsub('Illumina ','', prediction_sampleinfo$sequencing_platform)
p_platform = ggplot(prediction_sampleinfo, aes(x = Group, y = final_prob, fill = sequencing_platform)) +
	geom_boxplot(width = 0.7, outlier.shape = NA, color = '#333333', linewidth = 0.2, position = position_dodge(width = 0.7)) +
	geom_point(aes(color = sequencing_platform), position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.78), size = 1, stroke = 0) +
	scale_fill_manual(values = group_colors, drop = FALSE) +
	scale_color_manual(values = group_colors, drop = FALSE) +
	labs(x = NULL, y = 'rbcDNA predictive scores') +
	theme_sig + theme(legend.position = 'right', legend.title = element_text(size = 7), legend.text = element_text(size = 6)) + facet_grid(. ~ Dataset, scales = 'free_y') + ylim(0, 1.05) +
    stat_compare_means(aes(group = sequencing_platform, label = ..p.signif..), method = 'wilcox.test', size = 6/.pt, label.y.npc = 1, label.x.npc = 0.5)

row3 = plot_grid(p_lib, p_platform, ncol=2, rel_widths = c(0.8, 1), 
        labels = c("D", "E"), label_size = 12, label_fontface = "bold", label_x = 0.01, label_y = 0.98, hjust = 0, vjust = 0.5)
FigS6 = plot_grid(row1, row2, row3, ncol=1, rel_heights = c(0.9, 0.9, 1))

ggsave(file.path(out_dir, "FigureS6.pdf"), FigS6, width = 8, height = 7)
