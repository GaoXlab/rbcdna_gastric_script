args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_path <- gsub("~\\+~", " ", script_path)
script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))
if (!file.exists(file.path(script_dir, "plot_function.r"))) {
  script_dir <- getwd()
}

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(pROC)
  library(stringr)
  library(ggplot2)
  library(cowplot)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggpubr)
  library(ggplotify)
  library(patchwork)
  library(openxlsx)
  library(clinfun)
  library(epiR)
  library(tidyverse)
  library(magrittr)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'simu_function.r'), chdir = TRUE)
source(file.path(script_dir, 'or_function.r'), chdir = TRUE)

set.seed(1234)

n_sim <- 10000
population_size <- 100000

# Endoscopy adherence:
# 作为固定情景梯度
# 17.4%, 43.8%, 48.0%, 100%

# Blood-test adherence:
# 作为不确定性参数
# mode = 75%, lower = 60%，用 epi.betabuster() 模拟

tools <- read.xlsx("./Figures/simulation.xlsx") %>% filter(population == "All" & (tool != 'CA242'))

color_sim_palette <- c("grey", ggsci::pal_material("grey")(10)[3:5], ggsci::pal_material("brown")(10)[c(8,4)], ggsci::pal_material("blue")(10)[c(9,10)], ggsci::pal_material("deep-purple")(10)[c(9)])
names(color_sim_palette) = c("Endoscopy", "17.4%","43.8%","100","CEA","CA199","rbcDNA-1","rbcDNA-2")                    

# 1. Simulate sensitivity and specificity
tools_perf <- tools %>% filter(metric %in% c("sensitivity", "specificity"))

params_perf <- dplyr::bind_rows(lapply(seq_len(nrow(tools_perf)), find_params, dat = tools_perf))

params_perf$simulation <- lapply(seq_len(nrow(params_perf)),simulate_performance,params = params_perf,N = n_sim)

perf_sim <- params_perf %>%
  select(tool, population, metric, simulation) %>%
  unnest(simulation) %>%
  group_by(tool, population, metric) %>%
  mutate(iter = row_number()) %>%
  ungroup() %>%
  pivot_wider(id_cols = c(tool, population, iter),names_from = metric,values_from = simulation)

# 2. Simulate blood-test adherence only Endoscopy adherence will be fixed scenarios
tools_blood_adherence <- tools %>% filter(metric == "adherence",tool != "Endoscopy")
params_blood_adherence <- dplyr::bind_rows(lapply(seq_len(nrow(tools_blood_adherence)), find_params, dat = tools_blood_adherence))

params_blood_adherence$simulation <- lapply(
  seq_len(nrow(params_blood_adherence)),
  simulate_performance,
  params = params_blood_adherence,
  N = n_sim
)

blood_adherence_sim <- params_blood_adherence %>%
  select(tool, population, simulation) %>%
  unnest(simulation) %>%
  group_by(tool, population) %>%
  mutate(iter = row_number()) %>%
  ungroup() %>%
  rename(adherence = simulation) %>%
  mutate(uptake_scenario = "Blood uptake beta distribution")

# 3. Fixed endoscopy adherence scenarios
endo_adherence_scenarios <- tibble::tibble(
  tool = "Endoscopy",
  uptake_scenario = c("17.4%", "43.8%", "100%"),
  adherence = c(0.174, 0.438, 1.000)
)

endo_adherence_sim <- perf_sim %>%
  filter(tool == "Endoscopy") %>%
  distinct(tool, population, iter) %>%
  crossing(endo_adherence_scenarios %>% select(uptake_scenario, adherence))

adherence_sim <- bind_rows(endo_adherence_sim, blood_adherence_sim %>% mutate(uptake_scenario = "Blood uptake beta distribution"))

perf_with_adherence <- perf_sim %>%
  left_join(
    adherence_sim,
    by = c("tool", "population", "iter")
  )

# 4. Prevalence distributions
# Scenario A:
# (PMID: 33228549; multi-center cluster randomized trial) GC-prevalence: 167 / 37922
prev_A <- rbeta(n_sim, shape1=167, shape2=37922-167)
prev_A_tbl <- tibble(iter = seq_len(n_sim),scenario = "Scenario A",prevalence = prev_A)

# Scenario B:
# (PMID: 41517275; hospital-based cross-sectional population) GC-prevalence: 619 / 58218
prev_B <- rbeta(n_sim, shape1=619, shape2=58218-619)
prev_B_tbl <- tibble(iter = seq_len(n_sim),scenario = "Scenario B",prevalence = prev_B)

prevalence_tbl <- bind_rows(prev_A_tbl, prev_B_tbl)

sim_A <- run_simulation(perf_with_adherence, prevalence_tbl, "Scenario A")
sim_B <- run_simulation(perf_with_adherence, prevalence_tbl, "Scenario B")
sim_all <- bind_rows(sim_A, sim_B)

sim_A$tool = factor(sim_A$tool, levels = c("Endoscopy", "CEA","CA199","CA242","rbcDNA-1","rbcDNA-2") )

# 6. Summary table
sim_all[which(sim_all$uptake_scenario == 'Blood uptake beta distribution'), 'uptake_scenario'] = sim_all[which(sim_all$uptake_scenario == 'Blood uptake beta distribution'), 'tool']

summary_metrics <- c(
  "adherence", "prevalence", "tested_n",
  "TP", "FP", "confirmatory_endoscopy", "PPV", "NPV")

summary_all <- sim_all %>%
  group_by(scenario, tool, population, uptake_scenario) %>%
  summarise(
    across(
      all_of(summary_metrics),
      list(
        median = ~ median(.x, na.rm = TRUE),
        lower = ~ quantile(.x, 0.025, na.rm = TRUE),
        upper = ~ quantile(.x, 0.975, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

# Optional export
write.xlsx(summary_all,file = "./Figures/simulation_summary.xlsx",overwrite = TRUE)

plot_sim <- sim_all %>%
  mutate(
    method = if_else(tool == "Endoscopy", "Endoscopy", "Blood-based tests"),
    tool = factor(tool, levels = c("Endoscopy", "CEA", "CA199", "CA242", "rbcDNA-1", "rbcDNA-2")),
    uptake_scenario = factor(
      uptake_scenario,
      levels = c("17.4%","43.8%","100%","CEA","CA199","CA242","rbcDNA-1","rbcDNA-2")
    ),
    method = factor(method, levels = c("Endoscopy", "Blood-based tests")),
    scenario = factor(scenario, levels = c("Scenario A", "Scenario B"))
  )

plot_sim_ppv_npv <- plot_sim %>% filter(uptake_scenario!='17.4%' & uptake_scenario!='100%')

scenario_palette <- c("Scenario A" = "#4E79A7", "Scenario B" = "#E15759")

# 10. PPV curve by GC prevalence for rbcDNA

rbcdna_modes <- tools %>%
  filter(population == "All", tool %in% c("rbcDNA-1", "rbcDNA-2"),
         metric %in% c("sensitivity", "specificity")) %>%
  select(tool, metric, mode) %>%
  pivot_wider(names_from = metric, values_from = mode)

endoscopy_modes <- tools %>%
  filter(population == "All", tool == "Endoscopy",
         metric %in% c("sensitivity", "specificity")) %>%
  select(tool, metric, mode) %>%
  pivot_wider(names_from = metric, values_from = mode)

rbcdna_1_sens <- rbcdna_modes$sensitivity[which(rbcdna_modes$tool == "rbcDNA-1")]
rbcdna_1_spec <- rbcdna_modes$specificity[which(rbcdna_modes$tool == "rbcDNA-1")]
rbcdna_2_sens <- rbcdna_modes$sensitivity[which(rbcdna_modes$tool == "rbcDNA-2")]
rbcdna_2_spec <- rbcdna_modes$specificity[which(rbcdna_modes$tool == "rbcDNA-2")]
endoscopy_sens <- endoscopy_modes$sensitivity[which(endoscopy_modes$tool == "Endoscopy")]
endoscopy_spec <- endoscopy_modes$specificity[which(endoscopy_modes$tool == "Endoscopy")]

ppv_curve_settings <- tibble::tibble(
  setting = c(
    "Endoscopy",
    "rbcDNA-1 observed: SEN 86.1%, SPE 91.0%",
    "rbcDNA-2 observed: SEN 52.7%, SPE 98.1%"
  ),
  sensitivity = c(endoscopy_sens, rbcdna_1_sens, rbcdna_2_sens),
  specificity = c(endoscopy_spec, rbcdna_1_spec, rbcdna_2_spec)
)

ppv_prevalence_curve <- tidyr::expand_grid(
  prevalence = 10^seq(log10(0.001), log10(0.15), length.out = 500),
  ppv_curve_settings
) %>%
  mutate(PPV = calc_ppv(sensitivity, specificity, prevalence))

prevalence_marks <- tibble::tibble(
  prevalence = c(167 / 37922, 619 / 58218),
  label = c("Scenario A\n0.44%", "Scenario B\n1.06%")
)

ppv_mark_points <- prevalence_marks %>%
  tidyr::crossing(ppv_curve_settings) %>%
  mutate(PPV = calc_ppv(sensitivity, specificity, prevalence))

ppv_curve_palette <- c(
  "Endoscopy" = unname(color_sim_palette["Endoscopy"]),
  "rbcDNA-1 observed: SEN 86.1%, SPE 91.0%" = "#2C7FB8",
  "rbcDNA-2 observed: SEN 52.7%, SPE 98.1%" = "#084081"
)
ppv_prevalence_curve$setting = factor(ppv_prevalence_curve$setting, 
                                levels=c("Endoscopy",
                                        "rbcDNA-1 observed: SEN 86.1%, SPE 91.0%",
                                        "rbcDNA-2 observed: SEN 52.7%, SPE 98.1%"))
p_ppv_prevalence <- ggplot(ppv_prevalence_curve, aes(x = prevalence, y = PPV, color = setting)) +
  geom_line(linewidth = 0.8) +
  geom_point(data = ppv_mark_points, aes(x = prevalence, y = PPV, color = setting), size = 1.4) +
  geom_vline(data = prevalence_marks, aes(xintercept = prevalence),
             linetype = "dashed", linewidth = 0.2, color = c("#4E79A7", "#E15759" )) +
  geom_text(data = prevalence_marks, aes(x = prevalence, y = 0.4, label = label),
            inherit.aes = FALSE, angle = 0, hjust = 0, vjust = 0, size = 6/.pt, color = "grey25") +
  scale_x_log10()+#labels = scales::percent_format(accuracy = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0.02))) +
  coord_cartesian(ylim = c(0, 1.02)) +
  scale_color_manual(values = ppv_curve_palette) +
  labs(x = "Gastric cancer prevalence", y = "Positive predictive value", color = NULL) +
  sim_theme +
  theme(
    axis.text.x = element_text(color = "black", angle = 0, hjust = 0.5, vjust = 0.5, size = 6),
    legend.position = "inside",
    legend.position.inside = c(0.03, 0.97),
    legend.justification = c(0, 1),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key.height = unit(0.3, "cm"),
    legend.text = element_text(size = 6),
    legend.spacing.y = unit(0.01, "cm")) +
  guides(color = guide_legend(override.aes = list(size = 1, linewidth = 0.3)))

p_tp <- make_boxplot_panel(plot_sim, "TP", "Number of GC cases detected", color_by = "scenario")
p_fp <- make_boxplot_panel(plot_sim, "FP", "False positives", color_by = "scenario")
p_ppv <- make_boxplot_panel(plot_sim_ppv_npv, "PPV", "Positive predictive value", percent = TRUE)
p_npv <- make_boxplot_panel(plot_sim_ppv_npv, "NPV", "Negative predictive value", percent = TRUE, ylim = c(0.992, 1.000))

row_top <- plot_grid(
  p_tp, p_fp,
  nrow = 1, align = "h", axis = "tb", rel_widths = c(1, 1),
  labels = c("A", ""), label_size = 12, label_fontface = "bold",
  label_x = 0.005, hjust = 0, label_y = 1, vjust = 1.1
)

row_bottom <- plot_grid(
  p_ppv, p_npv, p_ppv_prevalence,
  nrow = 1, align = "h", axis = "tb", rel_widths = c(0.9, 0.9, 1.1),
  labels = c("B", "", "C"), label_size = 12, label_fontface = "bold",
  label_x = 0.005, hjust = 0, label_y = 1, vjust = 1.1
)

p_scenario_metrics <- plot_grid(row_top, row_bottom,ncol = 1, rel_heights = c(1, 1))

ggsave(file.path(out_dir, 'response_relatedto_Figure6.pdf'), p_scenario_metrics, width = 8, height = 6.5)#, device = cairo_pdf)
