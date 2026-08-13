
font_normal <- 8; font_small <- 6; font_label = 12
plot_score_subgroup <- function(data, x_var, x_label, facet_by_group = FALSE, label_sep = ", ", rotate_x = FALSE) {
  p <- ggplot(data = data, aes(x = .data[[x_var]], y = final_prob)) +
    geom_hline(yintercept=cutoff_spe90, color='red4', linetype='dashed', linewidth=0.2)+
    geom_boxplot(outlier.colour = NA, width=0.5, linewidth = 0.4)+
    geom_jitter(shape=16, position=position_jitter(0.2), size=0.5, color="darkgrey")+
    stat_compare_means(aes(label = paste0(after_stat(method), !!label_sep, after_stat(p.signif))),
                       method = "kruskal.test", label.x.npc = 'center', label.y = 1, size = 5 / .pt, hjust = 0.5, lineheight = 0.65) +
    theme_sig3 + ylab('rbcDNA predictive score') + xlab(x_label) + ylim(0, y_max)
  if (facet_by_group) {
    p <- p + facet_grid(.~Group)
  }
  if (rotate_x) {
    p <- p + theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }
  p
}

keep_available_vars <- function(vars, dat) {
  vars <- vars[vars %in% colnames(dat)]
  vars[sapply(vars, function(v) sum(!is.na(dat[[v]])) > 0)]
}

make_model_df <- function(dat) {
  get_col <- function(name) {
    if (name %in% colnames(dat)) dat[[name]] else NA
  }
  get_num <- function(name) {
    as.numeric(get_col(name))
  }
  dat %>%
  transmute(
    sample_id = get_col("Sample"),
    disease_status = get_col("Group"),
    rbcDNA_score = get_num("final_prob"),
    rbcDNA_score_z = as.numeric(scale(get_num("final_prob"))),
    age = get_col("Age"), sex = factor(get_col("Gender")),
    Smoking = factor(get_col("Smoking.state")), Alcohol = factor(get_col("Alcohol.state")), Hp = factor(get_col("HP")),
    Hb = get_num("Hb"), RBC = get_num("RBC"), WBC = get_num("WBC"), PLT = get_num("PLT"),
    NEU = get_num("NEU"), LYM = get_num("LYM"),
    NLR = get_num("NLR"), PLR = get_num("PLR"), SII = get_num("SII"), LMR = get_num("LMR"),
    RDW = get_num("RDW"), MCV = get_num("MCV"), MCHC = get_num("MCHC"), MCH = get_num("MCH")
  )
}

make_univariate_input <- function(dat, vars, positive_groups = "GC", negative_group = "Non-GC") {
  vars <- keep_available_vars(vars, dat)
  dat %>%
    filter(disease_status %in% c(positive_groups, negative_group)) %>%
    mutate(
      disease_status = factor(disease_status, levels = c(negative_group, positive_groups)),
      disease_numeric = case_when(
        disease_status == negative_group ~ 0,
        disease_status %in% positive_groups ~ 1,
        TRUE ~ NA_real_
      ),
      sex = factor(sex)
    ) %>%
    select(sample_id, disease_status, disease_numeric, all_of(unique(vars)))
}

run_univariate_set <- function(dat, vars) {
  dat_model <- make_univariate_input(dat, vars)
  vars <- setdiff(colnames(dat_model), c("sample_id", "disease_status", "disease_numeric"))
  bind_rows(lapply(vars, function(v) run_univariate_glm(dat_model, v)))
}

fit_gc_model <- function(dat, vars) {
  vars <- keep_available_vars(vars, dat)
  run_cluster_glm(
    data = dat,
    positive_groups = "GC",
    negative_group = "Non-GC",
    comparison_name = "GC_vs_non-GC",
    glm_vars = vars
  )
}

display_term <- function(x, line_break = FALSE) {
  replacement <- if (line_break) "rbcDNA score\n(per SD increase)" else "rbcDNA score (per SD increase)"
  if_else(x == "rbcDNA_score_z", replacement, x)
}

prepare_or_plot_data <- function(or_table) {
  plot_data <- or_table %>%
    group_by(comparison) %>%
    arrange(OR, .by_group = TRUE) %>%
    ungroup() %>%
    mutate(term = display_term(term, line_break = TRUE))

  term_order <- c(
    "rbcDNA score\n(per SD increase)", "age",
    grep("^sex", plot_data$term, value = TRUE),
    grep("^Smoking", plot_data$term, value = TRUE),
    grep("^Alcohol", plot_data$term, value = TRUE),
    grep("^HpYes", plot_data$term, value = TRUE),
    "Hb", "RDW", "NLR", "PLR", "SII"
  )
  plot_data %>%
    mutate(term = factor(term, levels = rev(term_order[term_order %in% term])))
}

make_or_forest <- function(or_table, title, rbc_only = FALSE) {
  plot_data <- prepare_or_plot_data(or_table)
  if (rbc_only) {
    plot_data <- filter(plot_data, term == "rbcDNA score\n(per SD increase)")
  }
  x_pad <- diff(range(c(plot_data$log_CI_lower, plot_data$log_CI_upper), na.rm = TRUE)) * 0.06

  ggplot(plot_data, aes(x = log_OR, y = term, group = comparison)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = log_CI_lower, xmax = log_CI_upper), height = 0.18, linewidth = 0.4) +
    geom_point(size = 2, shape = 21, fill = "#E64B35", color = "#333333", stroke = 0.3) +
    geom_text(aes(x = log_CI_upper + x_pad, label = significance), color = "black", size = 8 / .pt, hjust = 0, vjust = 0.5) +
    scale_y_discrete(drop = TRUE) +
    xlim(-0.5, 6) +
    labs(x = "Adjusted log(odds ratio)", y = NULL, title = title) +
    fig5_theme_common(base = "bw", base_size = 8, base_family = "", plot_margin = margin(t = 5, r = 16, b = 5, l = 5)) +
    theme(plot.title = element_text(size = 8, hjust = 0.5, margin = margin(b = 1)))
}

make_adjusted_pred <- function(base_dat, model_with_rbcDNA, model_without_rbcDNA) {
  base_dat %>%
    select(sample_id, disease_status, rbcDNA_score) %>%
    inner_join(
      model_with_rbcDNA$data %>% select(sample_id, pred_with_rbcDNA = pred),
      by = "sample_id"
    ) %>%
    inner_join(
      model_without_rbcDNA$data %>% select(sample_id, pred_without_rbcDNA = pred),
      by = "sample_id"
    ) %>%
    mutate(Target = if_else(disease_status == "GC", 1, 0)) %>%
    select(sample_id, disease_status, Target, rbcDNA_score, pred_with_rbcDNA, pred_without_rbcDNA)
}

format_model_section <- function(model_key, res) {
  formula_text <- gsub("rbcDNA_score_z", "rbcDNA score (per SD increase)", paste(res$formula, collapse = " "))
  bind_rows(
    tibble(
      Model_formula = paste0(model_labels[[model_key]], ": ", formula_text),
      term = NA_character_, statistic = NA_real_, OR_95CI = NA_character_, P_value = NA_real_, significance = NA_character_),
    res$or_table %>%
      select(all_of(model_export_cols)) %>%
      mutate(Model_formula = NA_character_, term = display_term(term)) %>%
      select(Model_formula, all_of(model_export_cols))
  )
}


theme_base_custom <- function(base_family = "") {
  font_normal <- 8; font_small <- 6; font_label = 12
  theme_classic(base_family = base_family) +
    theme(
      axis.title = element_text(size = font_normal, color = "black"),
      axis.text = element_text(size = font_small, color = "black"),
      axis.line = element_blank(),
      axis.ticks = element_line(linewidth = 0.4, color = "black"),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.4),
      legend.title = element_text(size = font_normal, color = "black"),
      legend.text = element_text(size = font_small, color = "black"),
      plot.title = element_text(size = font_normal, color = "black", hjust = 0.5),
      strip.background = element_blank(),
      strip.text = element_text(size = font_normal, color = "black")
    )
}

p_value_to_text <- function(p_value) {
  if (p_value < 0.001) {
    "DeLong test: P < 0.001"
  } else {
    str_c("DeLong test: P = ", round(p_value, 3))
  }
}

get_roc_curve_info <- function(roc_obj, label, color, fill_color = color, boot_n = 2000) {
  set.seed(1234)
  sp_obj <- pROC::ci.sp(roc_obj, sensitivities = seq(0, 100, 1), boot.n = boot_n, conf.level = 0.95)
  ci_df <- data.frame(
    Sens = as.numeric(gsub("%", "", rownames(sp_obj))),
    Sp_inv_low = 100 - as.numeric(sp_obj[, 3]),
    Sp_inv_up = 100 - as.numeric(sp_obj[, 1])
  )
  ci_df <- na.omit(ci_df)
  ci_df <- ci_df[order(ci_df$Sens), ]
  ci_auc <- pROC::ci.auc(roc_obj)
  list(
    roc_df = data.frame(Sp_inv = 100 - roc_obj$specificities, Sens = roc_obj$sensitivities),
    poly_df = data.frame(x = c(ci_df$Sp_inv_low, rev(ci_df$Sp_inv_up)), y = c(ci_df$Sens, rev(ci_df$Sens))),
    auc_text = paste0(label, ": ", round(ci_auc[2], 0), " (", round(ci_auc[1], 0), "-", round(ci_auc[3], 0), ")"),
    color = color,
    fill_color = fill_color
  )
}

plot_auc_panel <- function(curves, title, label_y, test_text_y = NULL, test_text = NULL) {
  p <- ggplot() +
    annotate("segment", x = 0, y = 0, xend = 100, yend = 100, color = "grey", linewidth = 0.4)
  for (curve in curves) {
    p <- p +
      geom_polygon(data = curve$poly_df, aes(x = x, y = y), fill = curve$fill_color) +
      geom_path(data = curve$roc_df, aes(x = Sp_inv, y = Sens), color = curve$color, linewidth = 1)
  }
  p <- p + annotate("text", x = 100, y = label_y[1], label = "AUC (%, 95CI%)", color = "black", size = 7 / .pt, hjust = 1)
  for (i in seq_along(curves)) {
    p <- p + annotate("text", x = 100, y = label_y[i + 1], label = curves[[i]]$auc_text, color = curves[[i]]$color, size = 6 / .pt, hjust = 1)
  }
  if (!is.null(test_text) && !is.null(test_text_y)) {
    for (i in seq_along(test_text)) {
      p <- p + annotate("text", x = 100, y = test_text_y[i], label = test_text[i], color = "black", size = 7 / .pt, hjust = 1)
    }
  }
  p + scale_x_continuous(name = "100-Specificity (%)", breaks = seq(0, 100, 20), expand = expansion(mult = c(0.04, 0.04))) +
      scale_y_continuous(name = "Sensitivity (%)", breaks = seq(0, 100, 20), expand = expansion(mult = c(0.04, 0.04))) +
      coord_cartesian(xlim = c(0, 100), ylim = c(0, 100)) +
      labs(title = title) +
      theme_base_custom()
}

plot_tm_box <- function(var_name, log_val, ylab_text) {
  df <- GC_pred_m[!is.na(GC_pred_m[[var_name]]), ]
  print(table(df$source_key))
  df$LogVal <- log(as.numeric(df[[var_name]]))
  ggplot(data = df, aes(x = Group, y = LogVal)) +
    geom_hline(yintercept=log(log_val), color='red4', linetype='dashed', size=0.2) +
    geom_boxplot(aes(fill=Group), outlier.colour = NA, position = position_dodge(width = 0.75)) +
    geom_jitter(shape = 16, position = position_jitter(0.2), size = 0.3, color = "black") +
    scale_color_manual(values=color_mapping) + scale_fill_manual(values=color_mapping) + facet_grid(.~source_key) +
    theme_classic() + theme_sig2 + ylab(ylab_text) + xlab('') + 
    theme(legend.position='none', axis.text.x = element_text(angle=45, hjust=1), panel.spacing = unit(0.1, "lines"))+
    stat_compare_means(aes(label = paste0(after_stat(method), ",\n", after_stat(p.signif))),
                       method = "wilcox.test", label.x.npc = 'center', label.y.npc = 0.95, size= 6 / .pt, hjust = 0.5)
}

format_cor_df <- function(cor_by_group, vars, labels = cor_group_labels) {
  cor_by_group %>%
    mutate(
      group = factor(group, levels = rev(names(labels))),
      variable = factor(variable, levels = vars),
      sig = case_when(
        is.na(pvalue) ~ "",
        pvalue < 0.001 ~ "***",
        pvalue < 0.01 ~ "**",
        pvalue < 0.05 ~ "*",
        TRUE ~ ""
      ),
      label = ifelse(is.na(cor), "", paste0(sprintf("%.2f", cor), sig))
    )
}

make_cor_by_group <- function(data, vars, score_var = "final_prob") {
  data %>%
    filter(Group %in% group_levels, source_key %in% source_levels) %>%
    group_by(Group, source_key) %>%
    group_split() %>%
    purrr::map_dfr(function(dat) {
      group_name <- paste0(as.character(unique(dat$Group)), "_", as.character(unique(dat$source_key)))
      purrr::map_dfr(vars, ~ cor_test_one(dat, .x, score_var)) %>%
        mutate(group = group_name)
    })
}

plot_cor_heatmap <- function(cor_df, vars, title, labels = cor_group_labels, legend_position = "right", plot_margin = margin(5, 5, 5, 5)) {
  cor_df$variable = factor(cor_df$variable, levels = vars)
  ggplot(cor_df, aes(x = variable, y = group, fill = cor)) +
    geom_tile(color = "white", linewidth = 0.4) +
    annotate("rect", xmin = 0.5, xmax = length(vars) + 0.5, ymin = 0.5, ymax = length(labels) + 0.5, fill = NA, color = "grey", linewidth = 0.3) +
    geom_text(aes(label = label), size = 6 / .pt, color = "black") +
    scale_fill_gradient2(
      low = "#2166AC", mid = "white", high = "#B2182B",
      midpoint = 0, limits = c(-1, 1), breaks = c(-1, -0.5, 0, 0.5, 1),
      name = "Correlation\ncoefficient"
    ) +
    scale_y_discrete(labels = labels) +
    labs(title = title) +
    fig5_theme_common(base_size = 8, base_family = "", axis_line = FALSE, legend_position = legend_position, plot_margin = plot_margin) +
    theme(
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      legend.title = element_text(size = 6),
      legend.text = element_text(size = 6),
      panel.border = element_blank(),
      plot.title = element_text(size = 8, color = "black", hjust = 0.5),
      plot.caption = element_text(size = 6, color = "black", hjust = 0),
      legend.key.width = grid::unit(2, "mm"),
      legend.spacing = grid::unit(0, "mm"),
      legend.box.spacing = grid::unit(0, "mm")
    )
}
