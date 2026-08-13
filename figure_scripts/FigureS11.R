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
	library(openxlsx)
	library(readxl)
	library(dplyr)
	library(ggplot2)
	library(tidyr)
	library(ggpubr)
	library(cowplot)
	library(patchwork)
})
source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)

group_cols <- c(
	"HD" = "#1F77B4",
	"GC (Stage IA)" = ggsci::pal_npg("nrc")(10)[1]
)

load('./Figures/prediction.RData')
score_cutoff_spe90 <- Cutoff(0.90, GC_trncv_pred)

plot_vars <- c(
	"RBC (×10^12/L)",
	"HGB (g/L)",
	"HCT (L/L)",
	"MCV (fL)",
	"MCHC (g/L)",
	"rbcDNA concentration (ng/mL)",
	"mt%",
	"rbcDNA predictive scores"
)

plot_df <- read_excel("./Figures/Supplementary_Tables.xlsx", sheet = "Supplementary Table 10")
plot_vars <- intersect(plot_vars, colnames(plot_df))

plot_df <- plot_df %>%
	mutate(
		Group = factor(Group, levels = names(group_cols)),
		Sex = factor(Gender, levels = c("Male", "Female")),
		Days = as.numeric(Days)
	) %>%
	pivot_longer(
		cols = all_of(plot_vars),
		names_to = "variable",
		values_to = "value",
		values_transform = list(value = as.character)
	) %>%
	mutate(
		value = suppressWarnings(as.numeric(value)),
		value = if_else(variable == "mt%", value * 100, value),
		variable = factor(variable, levels = plot_vars)
	) %>%
	filter(!is.na(Days), !is.na(Group), !is.na(Individual))

plot_summary <- plot_df %>%
	group_by(Individual, Group, Sex, Days, variable) %>%
	summarise(
		value_mean = mean(value, na.rm = TRUE),
		value_sd = sd(value, na.rm = TRUE),
		.groups = "drop"
	)

normal_ranges <- tibble(
	variable = rep(c("RBC (×10^12/L)", "HGB (g/L)", "MCV (fL)"), each = 2),
	Sex = rep(c("Male", "Female"), times = 3),
	ymin = c(3.69, 3.69, 108, 108, 86.7, 86.7),
	ymax = c(5.46, 5.46, 164, 164, 102.3, 102.3)
) %>%
	mutate(
		variable = factor(variable, levels = plot_vars),
		Sex = factor(Sex, levels = c("Male", "Female"))
	)

normal_ranges_by_sample <- plot_df %>%
	distinct(Individual, Group, Sex) %>%
	left_join(normal_ranges, by = "Sex", relationship = "many-to-many")
print(normal_ranges_by_sample)

plot_storage_variable <- function(panel_variable) {
	panel_data <- filter(plot_df, variable == panel_variable, is.finite(value))
	panel_summary <- filter(plot_summary, variable == panel_variable, is.finite(value_mean))
	if (panel_variable == "MCV (fL)") {
		mcv_d1 <- panel_summary %>%
			filter(Days == 1) %>%
			transmute(Individual, mcv_d1 = value_mean)
		panel_data <- panel_data %>%
			left_join(mcv_d1, by = "Individual") %>%
			mutate(value = value / mcv_d1) %>%
			filter(is.finite(value))
		panel_summary <- panel_summary %>%
			left_join(mcv_d1, by = "Individual") %>%
			mutate(
				value_mean = value_mean / mcv_d1,
				value_sd = value_sd / mcv_d1
			) %>%
			filter(is.finite(value_mean))
	}
	if (panel_variable %in% c("RBC (×10^12/L)", "MCV (fL)")) {
		panel_data <- filter(panel_data, Days != 0)
		panel_summary <- filter(panel_summary, Days != 0)
	}
	if (panel_variable == "rbcDNA concentration (ng/mL)") {
		rbcdna_d1 <- panel_summary %>%
			filter(Days == 1) %>%
			transmute(Individual, rbcdna_d1 = value_mean)
		panel_data <- panel_data %>%
			left_join(rbcdna_d1, by = "Individual") %>%
			mutate(value = log2(value / rbcdna_d1)) %>%
			filter(is.finite(value))
		panel_summary <- panel_summary %>%
			left_join(rbcdna_d1, by = "Individual") %>%
			mutate(
				value_mean = log2(value_mean / rbcdna_d1),
				value_sd = NA_real_
			) %>%
			filter(is.finite(value_mean))
	}

	ggplot(panel_data, aes(x = Days, y = value, color = Group)) +
		annotate("rect", xmin = 8, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "grey90", alpha = 0.5) +
		geom_hline(
			data = normal_ranges_by_sample %>%
				filter(variable == panel_variable, variable %in% c("RBC (×10^12/L)", "HGB (g/L)")) %>%
				distinct(variable, ymin, ymax) %>%
				pivot_longer(cols = c(ymin, ymax), values_to = "yintercept"),
			aes(yintercept = yintercept),
			linetype = "dashed",
			color = "grey50",
			linewidth = 0.3
		) +
		geom_point(size = 1.5, alpha = 0.55, position = position_jitter(width = 0.25, height = 0)) +
		geom_line(
			data = panel_summary %>%
				group_by(Individual) %>%
				filter(n() > 1) %>%
				ungroup(),
			aes(y = value_mean, group = Individual),
			linewidth = 0.7
		) +
		geom_point(data = panel_summary, aes(y = value_mean), size = 1) +
		scale_color_manual(values = group_cols, drop = FALSE) +
		scale_x_continuous(breaks = sort(unique(plot_df$Days))) +
		{
			if (panel_variable == "MCV (fL)") {
				geom_hline(yintercept = 1, linetype = "dashed", color = "grey40", linewidth = 0.3)
			}
		} +
		{
			if (panel_variable == "rbcDNA concentration (ng/mL)") {
				geom_hline(yintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.3)
			}
		} +
		{
			if (panel_variable == "mt%") {
				geom_hline(yintercept = 0.04, linetype = "dashed", color = "grey40", linewidth = 0.3)
			}
		} +
		{
			if (panel_variable == "rbcDNA predictive scores") {
				geom_hline(yintercept = score_cutoff_spe90, linetype = "dashed", color = "grey40", linewidth = 0.3)
			}
		} +
		{
			if (panel_variable == "rbcDNA concentration (ng/mL)") {
				coord_cartesian(ylim = c(-2.5, 0.5))
			}
		} +
		{
			if (panel_variable == "MCV (fL)") {
				coord_cartesian(ylim = c(0.95, 1.15))
			}
		} +
		labs(
			x = "Days",
			y = case_when(
				panel_variable == "mt%" ~ "Proportion of rbcDNA\nmapped to MT regions (%)",
				panel_variable == "rbcDNA concentration (ng/mL)" ~ "rbcDNA concentration\nlog2 fold change vs D1",
				panel_variable == "MCV (fL)" ~ "MCV normalized to D1",
				TRUE ~ panel_variable
			),
			color = "Group"
		) +
		theme_bw(base_size = 8) +
		theme(
			legend.position = "bottom",
			panel.grid.minor = element_blank(),
			panel.grid.major.y = element_blank(),
			axis.text.x = element_text(size = 6),
			axis.text.y = element_text(size = 6),
			axis.title.x = element_text(size = 8),
			axis.title.y = element_text(size = 8),
			plot.margin = margin(5, 5, 5, 5)
		)
}

plot_storage_panel <- function(panel_variables, ncol, show_legend = TRUE) {
	wrap_plots(lapply(panel_variables, plot_storage_variable), ncol = ncol, guides = "collect") &
		theme(legend.position = if (show_legend) "bottom" else "none")
}

cn_1000k_raw <- read.table(file.path(out_dir, "gc_used.blood_storage.1000kb_copyNumbersSmooth.txt"), header = TRUE, sep = "\t", check.names = FALSE)

cn_1000k_pair_map <- tibble(
	Group = c("HD", "GC (Stage IA)"),
	x_col = c("ELR1_4.nodup.q30", "ELR2_5.nodup.q30"),
	y_col = c("ELR1_5.nodup.q30", "ELR2_6.nodup.q30")
) %>%
	mutate(Group = factor(Group, levels = names(group_cols)))

cn_1000k_plot_df <- bind_rows(lapply(seq_len(nrow(cn_1000k_pair_map)), function(i) {
	pair_info <- cn_1000k_pair_map[i, ]
	cn_1000k_raw %>%
		transmute(
			feature,
			chromosome,
			start,
			end,
			Group = pair_info$Group,
			copy_number_x = suppressWarnings(as.numeric(.data[[pair_info$x_col]])),
			copy_number_y = suppressWarnings(as.numeric(.data[[pair_info$y_col]]))
		)
})) %>%
	filter(is.finite(copy_number_x), is.finite(copy_number_y))

cor_method <- "spearman"
cn_1000k_cor <- cn_1000k_plot_df %>%
	group_by(Group) %>%
	summarise(
		n = n(),
		correlation = suppressWarnings(cor(copy_number_x, copy_number_y, method = cor_method)),
		p_value = suppressWarnings(cor.test(copy_number_x, copy_number_y, method = cor_method, exact = FALSE)$p.value),
		.groups = "drop"
	)

p_cn_1000k <- ggplot(cn_1000k_plot_df, aes(x = copy_number_x, y = copy_number_y, color = Group)) +
	geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.3) +
	geom_point(size = 0.45, alpha = 0.35) +
	stat_cor(aes(color = Group), method = cor_method, size = 6 / .pt, show.legend = FALSE) +
	scale_color_manual(values = group_cols, drop = FALSE) +
	labs(x = "replicate 1", y = "replicate 2", color = "Group") +
	theme_bw(base_size = 8) +
	theme(
		legend.position = "bottom",
		panel.grid.minor = element_blank(),
		panel.grid.major = element_blank(),
		axis.text.x = element_text(size = 6),
		axis.text.y = element_text(size = 6),
		axis.title.x = element_text(size = 8),
		axis.title.y = element_text(size = 8),
		plot.margin = margin(5, 5, 5, 5)
	)

write.csv(cn_1000k_cor, file.path(out_dir, "FigureS11_cn_1000k_three_sample_pairs_correlations.csv"), row.names = FALSE)

p_blood1 <- plot_storage_panel(c("RBC (×10^12/L)", "MCV (fL)", "rbcDNA concentration (ng/mL)"), ncol = 3, show_legend = FALSE)
p_blood2 <- plot_storage_panel(c("mt%", "rbcDNA predictive scores"), ncol = 2, show_legend = TRUE)
aligned_bottom <- cowplot::align_plots(p_cn_1000k, p_blood2, align = "h", axis = "tb")

FigS11 <- plot_grid(
	p_blood1,
	plot_grid(aligned_bottom[[1]],aligned_bottom[[2]],ncol = 2,rel_widths = c(1, 2),
	align = "h",axis = "b",labels = c("C", "D", "E"),label_size = 12,label_fontface = "bold",label_x = 0.01,label_y = 1,hjust = 0,vjust = 1),
	ncol = 1,rel_heights = c(1, 1.2),labels = c("A", "", "B"),label_size = 12,label_fontface = "bold",label_x = 0.01,label_y = 1,hjust = 0,vjust = 1)

ggsave(file.path(out_dir, "FigureS11.pdf"), FigS11, width = 8, height = 5)
