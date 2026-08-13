args <- commandArgs(trailingOnly = TRUE)
working_dir <- if (length(args) >= 1) gsub("~\\+~", " ", args[1]) else getwd()
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_path <- gsub("~\\+~", " ", script_path)
script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))
if (!file.exists(file.path(script_dir, "plot_function.r"))) {
  script_dir <- file.path(working_dir, "figure_scripts")
}

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(pROC)
  library(stringr)
  library(ggplot2)
  library(cowplot)
  library(ggsci)
  library(dplyr)
  library(tidyr)
  library(openxlsx)
  library(epiR)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'simu_function.r'), chdir = TRUE)
source(file.path(script_dir, 'or_function.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

GC_ind1_pred$source_key  = 'ind-Zhejiang'
GC_ind2_pred$source_key  = 'ind-Shandong'
GC_ind3_pred$source_key  = 'ind-Henan'
GC_ind_pred <- as.data.frame(rbind(GC_ind1_pred, GC_ind2_pred, GC_ind3_pred))
GC_prediction <- as.data.frame(GC_ind_pred)

## cutoff determined
cutoff_spe90 <- Cutoff(0.90, GC_trncv_pred)
cutoff_spe95 <- Cutoff(0.95, GC_trncv_pred)

## GC model performance
needed_cols <- unique(c('Sample', 'Group', 'Source', 'Age', 'Gender', "CEA (ng/mL)", "CA19-9 (U/mL)", "CA242 (U/mL)"))
sampleinfo_tmp <- sampleinfo[, needed_cols]

colnames(sampleinfo_tmp)[colnames(sampleinfo_tmp) == 'Source'] <- 'dataset_label'
sampleinfo_tmp$dataset_label <- factor(sampleinfo_tmp$dataset_label, levels = c("ZHEJIANG", "ANYANG", "SHANDONG"))
colnames(sampleinfo_tmp)[grep('CEA', colnames(sampleinfo_tmp))] <- 'CEA'
colnames(sampleinfo_tmp)[grep('CA19-9', colnames(sampleinfo_tmp))] <- 'CA199'
colnames(sampleinfo_tmp)[grep('CA242', colnames(sampleinfo_tmp))] <- 'CA242'

GC_pred_m <- merge(GC_prediction, sampleinfo_tmp, by='Sample')

GC_pred_m$Group <- factor(GC_pred_m$Group, levels=c('Non-GC','GC'))
GC_pred_m <- GC_pred_m %>% filter(!is.na(CEA), !is.na(CA199))
print(paste0("Number of samples with CEA/CA199 available: ", nrow(GC_pred_m)))  

### rbcDNA/CEA/CA199 available
roc1_test = pROC::roc(GC_pred_m$Target,GC_pred_m$CEA, percent = TRUE) 
roc2_test = pROC::roc(GC_pred_m$Target,GC_pred_m$CA199, percent = TRUE) 
roc3_test = pROC::roc(GC_pred_m$Target,GC_pred_m$final_prob, percent = TRUE) 

GC_pred_m1 <- merge(GC_prediction, sampleinfo_tmp, by='Sample')
GC_pred_m1$Group <- factor(GC_pred_m1$Group, levels=c('Non-GC','GC'))
GC_pred_m1 <- GC_pred_m1 %>% filter(!is.na(CA242))
roc4_test = pROC::roc(GC_pred_m1$Target,GC_pred_m1$CA242, percent = TRUE) 

test_curves <- list(
  get_roc_curve_info(roc4_test, "CA 242 available", pal_material("brown")(10)[2], pal_material("brown", alpha=0.2)(10)[2]),
  get_roc_curve_info(roc2_test, "CA 19-9 available", pal_material("brown")(10)[4], pal_material("brown", alpha=0.2)(10)[4]),
  get_roc_curve_info(roc1_test, "CEA available", pal_material("brown")(10)[8], pal_material("brown", alpha=0.2)(10)[8]),
  get_roc_curve_info(roc3_test, "rbcDNA", pal_material("light-blue")(10)[9], pal_material("light-blue", alpha=0.2)(10)[9])
)
pA <- plot_auc_panel(test_curves, title = "Tumor marker-available samples\nin the independent validation sets", label_y = c(32, 4, 10, 16, 22))

GC_pred_m <- merge(GC_prediction, sampleinfo_tmp, by='Sample')
### sensitivity and specificity of CEA/CA199
  CEA_dat <- GC_pred_m %>% mutate(predicted=ifelse(CEA >= 5, 1, 0))
  perf_CEA_dat <- get_performance_stats(CEA_dat)

  CA199_dat <- GC_pred_m %>% mutate(predicted=ifelse(CA199 >= 37, 1, 0))
  perf_CA199_dat <- get_performance_stats(CA199_dat)

  CA242_dat <- GC_pred_m %>% mutate(predicted=ifelse(CA242 >= 20, 1, 0))
  perf_CA242_dat <- get_performance_stats(CA242_dat)

  rbcDNA_2_dat <- GC_pred_m %>% mutate(predicted=ifelse(final_prob >= cutoff_spe95, 1, 0))
  perf_rbcDNA_2_dat <- get_performance_stats(rbcDNA_2_dat)

perf_list <- list(
  CEA = perf_CEA_dat,
  `CA19-9` = perf_CA199_dat,
  `CA242` = perf_CA242_dat,
  `rbcDNA-2` = perf_rbcDNA_2_dat
)

all_sen <- bind_rows(lapply(names(perf_list), function(x) {
  make_perf_row(perf_list[[x]], x, "Sensitivity")
}))

all_spe <- bind_rows(lapply(names(perf_list), function(x) {
  make_perf_row(perf_list[[x]], x, "Specificity")
}))

cmb_sen_spe <- bind_rows(all_sen, all_spe) %>% mutate(
    Subgroup=factor(Subgroup, levels=c("CEA","CA19-9","CA242","rbcDNA-2")),
    Metric=factor(Metric, levels=c("Sensitivity","Specificity")),
    Result=sprintf("%.0f%%\n%s", Estimate, Tag))

pB <- ggplot(cmb_sen_spe,aes(x=Subgroup, y=Estimate))+
    geom_errorbar(aes(ymin=Lower,ymax=Upper),width=0.18,linewidth=0.4)+
    geom_point(size=2,shape=21,fill="#E64B35",color="#333333",stroke=0.3)+
    geom_text(aes(y=pmin(Upper + 2, 104),label=Result),vjust=0,size=5/.pt)+
    facet_grid(.~Metric)+ labs(x=NULL,y="Performance, % (95% CI)")+ 
    scale_y_continuous(breaks=seq(0,100,20))+ theme_sig2 +
    theme(strip.background=element_blank(),strip.text=element_text(size=8,face="plain"), 
          axis.text.x=element_text(angle=45,hjust=1,vjust=1, size=6))+
    coord_cartesian(ylim=c(0,110),clip="off")




### simulation
set.seed(1234)

n_sim <- 10000
population_size <- 100000

# Endoscopy adherence:
# 作为固定情景梯度
# 17.4%, 43.8%, 48.0%, 100%

# Blood-test adherence:
# 作为不确定性参数
# mode = 75%, lower = 60%，用 epi.betabuster() 模拟

tools <- read.xlsx("./Figures/simulation.xlsx") %>% 
          filter((population == "All") & (tool != 'rbcDNA-1 + tumor biomarkers') & (tool != 'rbcDNA-2 + tumor biomarkers') & (tool != 'rbcDNA-1'))# & (tool != 'CA242'))

color_sim_palette <- c("grey", ggsci::pal_material("grey")(10)[3:5], ggsci::pal_material("brown")(10)[c(8,4,2)], ggsci::pal_material("blue")(10)[c(9,10)], ggsci::pal_material("deep-purple")(10)[c(9)])
names(color_sim_palette) = c("Endoscopy", "17.4%","43.8%","100","CEA","CA199","CA242","rbcDNA-1","rbcDNA-2","rbcDNA-2 + tumor biomarkers")                    
color_sim_palette["rbcDNA-2 +\ntumor biomarkers"] <- color_sim_palette["rbcDNA-2 + tumor biomarkers"]

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
endo_adherence_scenarios <- tibble::tibble(tool = "Endoscopy",uptake_scenario = c("17.4%", "43.8%","100%"),adherence = c(0.174, 0.438, 1))

endo_adherence_sim <- perf_sim %>%
  filter(tool == "Endoscopy") %>%
  distinct(tool, population, iter) %>%
  crossing(endo_adherence_scenarios %>% select(uptake_scenario, adherence))

adherence_sim <- bind_rows(endo_adherence_sim, blood_adherence_sim %>% mutate(uptake_scenario = "Blood uptake beta distribution"))

perf_with_adherence <- perf_sim %>% left_join(adherence_sim,by = c("tool", "population", "iter"))

# 4. Prevalence distributions
# Scenario B:
# (PMID: 41517275; hospital-based cross-sectional population) GC-prevalence: 619 / 58218
prev_B <- rbeta(n_sim, shape1=619, shape2=58218-619)
prev_B_tbl <- tibble(iter = seq_len(n_sim),scenario = "Scenario B",prevalence = prev_B)

# 5. Simulation function
sim_B <- run_simulation(perf_with_adherence, prev_B_tbl, "Scenario B")

sim_B$tool = factor(sim_B$tool, levels = c("Endoscopy", "CEA","CA199","CA242","rbcDNA-2") )
pC_data = sim_B %>% distinct(tool, population, iter, sensitivity, specificity)

pC <- pC_data %>% 
    ggplot(aes(specificity*100, sensitivity*100)) +
    geom_point(aes(color=tool), size=0.01, alpha=0.01) +
    geom_density2d(aes(color=tool), linewidth = 0.1) +
    theme_base_custom() + 
    theme(axis.text.x = element_text(color = "black", angle = 0, hjust = 0.5, vjust = 0.5, size = 6),legend.title = element_blank(),
          legend.position="inside", legend.position.inside=c(0.3, 0.25), legend.background = element_rect(fill="transparent"),
          legend.key.height = unit(0.3, "cm"), legend.text = element_text(size = 6), legend.spacing.y =unit(0.01, "cm")) +
    guides(color=guide_legend(override.aes=list(size=1, linewidth = 0.3))) +
    labs(x="100-Specificity (%)", y="Sensitivity (%)") +  
    scale_x_continuous(breaks=seq(85,100,5), limits=c(85,100))+
    scale_y_continuous(breaks=seq(0,100,20), limits=c(0,100))+
    scale_color_manual(values=color_sim_palette)

sim_B[which(sim_B$uptake_scenario == 'Blood uptake beta distribution'), 'uptake_scenario'] = sim_B[which(sim_B$uptake_scenario == 'Blood uptake beta distribution'), 'tool']

plot_sim <- sim_B %>%
  mutate(
    method = if_else(tool == "Endoscopy", "Endoscopy", "Blood-based tests"),
    tool = if_else(as.character(tool) == "rbcDNA-2 + tumor biomarkers", "rbcDNA-2 +\ntumor biomarkers", as.character(tool)),
    uptake_scenario = if_else(as.character(uptake_scenario) == "rbcDNA-2 + tumor biomarkers", "rbcDNA-2 +\ntumor biomarkers", as.character(uptake_scenario)),
    tool = factor(tool, levels = c("Endoscopy", "CEA","CA199","CA242","rbcDNA-2")),
    uptake_scenario = factor(
      uptake_scenario,
      levels = c("17.4%","43.8%","100%","CEA","CA199","CA242","rbcDNA-2")
    ),
    method = factor(method, levels = c("Endoscopy", "Blood-based tests"))
  )

plot_sim_ppv_npv <- plot_sim %>% filter(uptake_scenario!='17.4%' & uptake_scenario!='100%')
x_axis_width <- function(dat) nlevels(droplevels(dat$uptake_scenario))

pTP <- make_boxplot_panel(plot_sim, "TP", "Number of GC cases detected", color_by = "uptake_scenario", legend_position = 'none')
pFP <- make_boxplot_panel(plot_sim, "FP", "False positives", color_by = "uptake_scenario", legend_position = 'none')
pFNR <- make_boxplot_panel(plot_sim_ppv_npv, "FNR", "False negative rate", color_by = "uptake_scenario", legend_position = 'none')
pPPV <- make_boxplot_panel(plot_sim_ppv_npv, "PPV", "Positive predictive value", percent = TRUE, color_by = "uptake_scenario", legend_position = 'none')
pNPV <- make_boxplot_panel(plot_sim_ppv_npv, "NPV", "Negative predictive value", percent = TRUE, ylim = c(0.992, 1.000), color_by = "uptake_scenario", legend_position = 'none')

bottom_right_widths <- c(
  x_axis_width(plot_sim),x_axis_width(plot_sim),x_axis_width(plot_sim_ppv_npv)+1,
  x_axis_width(plot_sim_ppv_npv)+1,x_axis_width(plot_sim_ppv_npv)+1)
bottom_widths <- c(0.5, 1.2 * bottom_right_widths / sum(bottom_right_widths))

top_row <- plot_grid(
  pA, pB, pC,
  ncol = 3, align = "h", axis = "tb", rel_widths = c(0.5, 0.5, 0.5),
  labels = c("A", "B", "C"), label_size = 12, label_fontface = "bold",
  label_x = 0.01, label_y = 0.98, hjust = 0, vjust = 0.5
)

bottom_row <- plot_grid(
  pTP, pFP, pFNR, pPPV, pNPV,
  ncol = 5, align = "h", axis = "tb", rel_widths = c(1, 1, 0.9,0.9,0.9),
  labels = c("D", "", "", "E", ""), label_size = 12, label_fontface = "bold",
  label_x = 0.01, label_y = 0.98, hjust = 0, vjust = 0.5
)

g <- plot_grid(top_row, bottom_row, ncol = 1, align = "v", axis = "lr", rel_heights = c(1.1,1))

ggsave(file.path(out_dir, "Figure6.pdf"), g, width = 8, height = 5.6)


summary_df <- Reduce(
  function(x, y) merge(x, y, by = "uptake_scenario", all = TRUE),
  list(
    setNames(aggregate(TP ~ uptake_scenario, sim_B, mean), c("uptake_scenario", "mean_TP")),
    setNames(aggregate(FP ~ uptake_scenario, sim_B, mean), c("uptake_scenario", "mean_FP")),
    setNames(aggregate(FNR ~ uptake_scenario, sim_B, mean), c("uptake_scenario", "mean_FNR")),
    setNames(aggregate(PPV ~ uptake_scenario, sim_B, mean), c("uptake_scenario", "mean_PPV")),
    setNames(aggregate(NPV ~ uptake_scenario, sim_B, mean), c("uptake_scenario", "mean_NPV"))
  )
)

print(summary_df)