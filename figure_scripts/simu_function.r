sim_theme <- theme_classic(base_size = 7) +
  theme(
    axis.text = element_text(color = "black", size = 6),
    axis.text.x = element_text(color = "black", angle = 45, hjust = 1, vjust = 1, size = 6),
    axis.title = element_text(color = "black", size = 8),
    axis.title.x = element_text(color = "black", size = 8),
    axis.title.y = element_text(color = "black", size = 8),
    strip.background = element_blank(),
    strip.text = element_text(color = "black", size = 6),
    plot.title = element_text(color = "black", size = 6, hjust = 0.5),
    legend.title = element_blank(),
    legend.text = element_text(color = "black", size = 6),
    legend.position = "bottom", panel.spacing = unit(0.08, "lines"),
    plot.margin = margin(t = 5, r = 2, b = 5, l = 0, unit = "pt")
  )

find_params <- function(i, dat) {
  dat_i <- dat[i, ]
  params_i <- epi.betabuster(mode = dat_i$mode,conf = 0.975,imsure = "greater than",x = dat_i$lower,max.shape1 = 500,conf.level = 0.95)
  dat_i$shape1 <- params_i$shape1
  dat_i$shape2 <- params_i$shape2
  dat_i
}

simulate_performance <- function(i, params, N = 10000){
    p <- params[i, ]
    rbeta(N, shape1=p$shape1, shape2=p$shape2)
}

run_simulation <- function(perf_with_adherence, prevalence_tbl, scenario_name) {
  perf_with_adherence %>%
    left_join(prevalence_tbl %>% filter(scenario == scenario_name), by = "iter") %>%
    mutate(
      scenario = scenario_name,
      population_size = population_size,

      tested_n = population_size * adherence,
      disease_n = tested_n * prevalence,
      non_disease_n = tested_n * (1 - prevalence),

      TP = disease_n * sensitivity,
      FN = disease_n * (1 - sensitivity),
      FP = non_disease_n * (1 - specificity),
      TN = non_disease_n * specificity,

      # For direct endoscopy, endoscopy burden is all completed endoscopies.
      # For blood tests, confirmatory endoscopy burden is positive tests = TP + FP.
      confirmatory_endoscopy = if_else(tool == "Endoscopy", tested_n, TP + FP),

      PPV = TP / (TP + FP),
      NPV = TN / (TN + FN),
      FPR = FP / non_disease_n,
      FNR = FN / disease_n
    )
}

make_boxplot_panel <- function(dat, value_col, y_label, percent = FALSE, ylim = NULL, color_by = c("scenario", "uptake_scenario"), legend_position = "bottom") {
  color_by <- match.arg(color_by)
  color_var <- ifelse(color_by == "scenario", "scenario", "uptake_scenario")
  panel_palette <- if (color_by == "scenario") scenario_palette else color_sim_palette

  y_scale <- scale_y_continuous(expand = expansion(mult = c(0, 0.1)))
  pd_scenario <- position_dodge2(width = 0.55, preserve = "single")

  p <- dat %>%
    ggplot(aes(
      x = uptake_scenario,y = .data[[value_col]],
      fill = .data[[color_var]],group = interaction(uptake_scenario, .data[[color_var]])
    )) +
    geom_boxplot(outlier.size = 0.05, linewidth=0.1, outlier.shape=NA,
                 position = pd_scenario, width = 0.6) +
    facet_grid(. ~ method, scales = "free_x", space = "free_x") +
    labs(x = NULL, y = y_label, color = NULL, fill = NULL) +
    # scale_color_manual(values = panel_palette) +
    scale_fill_manual(values = panel_palette) +
    y_scale + sim_theme + theme(legend.position = legend_position)# + guides(fill = "none")

  if (!is.null(ylim)) {
    p <- p + coord_cartesian(ylim = ylim)
  }
  p
}

calc_ppv <- function(sens, spec, prev) {
  sens * prev / (sens * prev + (1 - spec) * (1 - prev))
}
