
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
	library(ggplot2)
	library(pROC)
	library(cowplot)
})

source(file.path(script_dir, "or_function.r"), chdir = TRUE)

model_cols <- c("X0", "lr", "cb")
model_labels <- c("Combined", "Logistic regression", "CatBoost")
model_colors <- c("#374E55", "#00A1D5", "#DF8F44")
model_fills <- c(
	rgb(55, 78, 85, 18, maxColorValue = 255),
	rgb(0, 161, 213, 18, maxColorValue = 255),
	rgb(223, 143, 68, 18, maxColorValue = 255)
)

random_curve_df <- function(roc_obj) {
	data.frame(Sp_inv = 100 - roc_obj$specificities, Sens = roc_obj$sensitivities)
}

read_prediction_file <- function(pred_file) {
	a <- read.table(pred_file, sep = ",", header = TRUE, check.names = TRUE)
	if ("target" %in% colnames(a)) {
		a$Target <- as.integer(a$target)
	} else {
		a$Target <- 0
		a[grep("GLRGC", a$seqID), "Target"] <- 1
	}
	a
}

make_nestedcv_auc <- function(pred_file, dataset_label, panel_title) {
	a <- read_prediction_file(pred_file)
	roc_list <- lapply(model_cols, function(model_col) {
		pROC::roc(a$Target, a[[model_col]], levels = c(0, 1), direction = "<", percent = TRUE, quiet = TRUE)
	})
	names(roc_list) <- model_cols

	auc_results <- data.frame(
		dataset = dataset_label,
		model = model_cols,
		label = model_labels,
		AUC = vapply(roc_list, function(x) as.numeric(pROC::auc(x)) / 100, numeric(1)),
		AUC_percent = vapply(roc_list, function(x) as.numeric(pROC::auc(x)), numeric(1))
	)

	roc_test_results <- data.frame(
		dataset = dataset_label,
		comparison = c("X0 vs lr", "X0 vs cb", "lr vs cb"),
		p_value = c(pROC::roc.test(roc_list$X0, roc_list$lr, method = "delong")$p.value,
		            pROC::roc.test(roc_list$X0, roc_list$cb, method = "delong")$p.value,
		            pROC::roc.test(roc_list$lr, roc_list$cb, method = "delong")$p.value)
	)

	set.seed(1234)
	roc_random <- pROC::roc(sample(a$Target), a$X0, levels = c(0, 1), direction = "<", percent = TRUE, quiet = TRUE)
	curves <- lapply(seq_along(roc_list), function(i) {
		get_roc_curve_info(
			roc_obj = roc_list[[i]],
			label = model_labels[i],
			color = model_colors[i],
			fill_color = model_fills[i]
		)
	})

	p_auc <- plot_auc_panel(
		curves,
		title = panel_title,
		label_y = c(34, 28, 22, 16)
	) +
		geom_path(
			data = random_curve_df(roc_random),
			aes(x = Sp_inv, y = Sens),
			color = rgb(128, 128, 128, 120, maxColorValue = 255),
			linewidth = 0.4
		) +
		annotate(
			"text", x = 100, y = 8, label = "Random Classifiers",
			color = rgb(128, 128, 128, 180, maxColorValue = 255),
			size = 6 / .pt, hjust = 1
		)

	list(plot = p_auc, auc_results = auc_results, roc_test_results = roc_test_results)
}

pred_file <- "Human_model/results/4_Classification/gc_full_cv_prediction.csv"
npcs_50_pred_file <- "Human_model/results/4_Classification/gc_full_cv_rnd_prediction.csv"

nestedcv_result <- make_nestedcv_auc(pred_file, "nestedcv_training_ids", "Nested 5-fold CV")
npcs_50_result <- make_nestedcv_auc(npcs_50_pred_file, "npcs_50_predictions", "Nested 5-fold CV (NPCs 50)")

auc_results <- nestedcv_result$auc_results
roc_test_results <- nestedcv_result$roc_test_results
auc_results_all <- rbind(nestedcv_result$auc_results, npcs_50_result$auc_results)
roc_test_results_all <- rbind(nestedcv_result$roc_test_results, npcs_50_result$roc_test_results)

print(auc_results_all)
print(roc_test_results_all)

p_auc <- nestedcv_result$plot
p_auc_npcs_50 <- npcs_50_result$plot
p_auc_grid <- cowplot::plot_grid(p_auc,p_auc_npcs_50,labels = c("A", "B"),nrow = 1,align = "h",axis = "tb",label_size = 12)

ggsave(file.path(out_dir, "response_nestedcv_auc_plot.pdf"), p_auc_grid, width = 6.4, height = 3.0)
