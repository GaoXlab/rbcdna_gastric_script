library(ggplot2)


read_and_merge <- function(filename, sampleinfo) {
  df <- read.table(
    str_c(filename), sep = ',', header = TRUE,
    col.names = c('Sample', 'final_prob', 'lr', 'cb', 'source_key')
  )
  df <- merge(df, sampleinfo[, c('Sample', 'Group')], by = 'Sample')
  df$Target <- ifelse(df$Group == 'GC', 1, 0)
  return(df)
}

Cutoff <- function(desired_specificity, pred){
    library(dplyr)
    library(cutpointr)
    roc2 <- cutpointr::roc
    pred$Target = factor(pred$Target)
    cutoff_atspe <- pred %>%
        cutpointr::roc(x = final_prob, class = Target,
            pos_class="1",
            neg_class="0",
            direction = ">=") %>%
        mutate(sens=tp/(tp+fn),
               spec=1-fpr) %>%
        filter(spec > desired_specificity, is.finite(x.sorted)) %>%
        pull(x.sorted) %>%
        min()
    detach("package:cutpointr", unload = TRUE)
    cutoff_atspe
}

mytheme <- theme_classic(base_size=6) + theme( #
    axis.text.x = element_blank(),#
    axis.text.y = element_text(size=5),#
    axis.ticks.x = element_blank(),#
    strip.text.y = element_blank(),#
    strip.text.x = element_text(size=5),#
    strip.background.x = element_blank(), #
    strip.placement = "outside",#
    axis.title.x = element_blank(),#
    legend.position = "none",#
    panel.grid.major = element_blank(),#
    panel.grid.minor = element_blank())#

theme_bar <- theme_bw() +
      theme(legend.position='none',
        axis.text = element_text(color = "black", size = 6),
        axis.title = element_text(color = "black", size = 8),
        axis.title.x = element_blank(),
        strip.text = element_text(color = "black", size = 6),
        strip.background = element_blank(),
        plot.title = element_text(color = "black", size = 6, hjust = 0.5),
        axis.line = element_blank(),
        axis.ticks = element_line(linewidth = 0.4),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
        panel.grid = element_blank(),
        plot.margin = margin(2, 2, 2, 2))

theme_bar1 <- theme_bw() +
      theme(legend.position='none',
        axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_text(color = "black", size = 6),
        axis.title.x = element_text(color = "black", size = 8),
        strip.text = element_text(color = "black", size = 6),
        plot.title = element_text(color = "black", size = 6, hjust = 0.5),
        axis.line = element_blank(),
        axis.ticks = element_line(linewidth = 0.4),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
        panel.grid = element_blank(),
        plot.margin = margin(2, 2, 2, 2))

theme_sig <- theme_classic() + theme(
    legend.position='none',
    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
    axis.text = element_text(color = "black", size = 6),
    axis.title = element_text(color = "black", size = 8),
    axis.title.x = element_blank(),
    strip.text = element_text(color = "black", size = 6),
    axis.line = element_line(linewidth = 0.4),
    axis.ticks.length = unit(0.1, "cm"),
    strip.background=element_blank())

theme_sig2 <- theme_classic() + theme(
    legend.position='none',
    axis.text = element_text(color = "black", size = 6),
    axis.title = element_text(color = "black", size = 8),
    strip.text = element_text(color = "black", size = 6),
    axis.line = element_line(linewidth = 0.02),
    axis.ticks = element_line(linewidth = 0.02),
    axis.line.x = element_line(color = "black", linewidth = 0.3),
    axis.line.y = element_line(color = "black", linewidth = 0.3),
    axis.ticks.x = element_line(color = "black", linewidth = 0.3),
    axis.ticks.y = element_line(color = "black", linewidth = 0.3),
    strip.background=element_blank(),
    plot.margin = margin(5, 5, 5, 5))

theme_cor <- theme_bw() + theme(
    legend.position='none',
    axis.text = element_text(color = "black", size = 6),
    axis.title = element_text(color = "black", size = 8),
    strip.text = element_text(color = "black", size = 6),
    plot.title = element_text(color = "black", size = 6, hjust = 0.5),
    axis.line = element_blank(),
    axis.ticks = element_line(linewidth = 0.4),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
    panel.grid = element_blank(),
    strip.background=element_blank())

getinfo = function(pred_df){
      pred_df_pred2 = unique(merge(pred_df, sampleinfo[,c('Sample','Group','Source','Stage','Lauren classification','Tumor location','Helicobacter pylori','Avg.tumorsize','Tumor size group','Atrophic','IntestinalMetaplasia')], by='Sample'))
      colnames(pred_df_pred2)[grep('Source', colnames(pred_df_pred2))] = 'dataset_label'
      pred_df_pred2$Stage_cmb = 'Non-GC'; pred_df_pred2[which((pred_df_pred2$Stage=='0')|(pred_df_pred2$Stage=='I')),'Stage_cmb' ] = 'early GC'
      pred_df_pred2[which((pred_df_pred2$Stage=='II')|(pred_df_pred2$Stage=='III')),'Stage_cmb' ] = 'advanced GC'
      pred_df_pred2[which((pred_df_pred2$Target==1) & (pred_df_pred2$"Lauren classification"=='')), 'Lauren classification'] = 'Missing'
      pred_df_pred2[which((pred_df_pred2$Target==1) & is.na(pred_df_pred2$"Lauren classification")), 'Lauren classification'] = 'Missing'
      return(pred_df_pred2)
}

MNdna_profiles_df1 <- function(df, label, label_samples){
    # Input(df): chr, region, sample1, sample2, ..., sampleN
    # Output(df1): chr, region, median, min, max, label, feature
    df <- df[, label_samples]
    df <- cbind(rownames(df), apply(df, 1, median), apply(df, 1, min), apply(df, 1, max)); colnames(df) <- c('region', 'median', 'min', 'max')
    df <- as.data.frame(df)
    df$label <- label
    df$median <- as.numeric(df$median)
    df$min <- as.numeric(df$min)
    df$max <- as.numeric(df$max)
    return(df)
}

get_sensitivity_inxspe <- function(pred, threshold) {
    library(dplyr); library(stringr)
    pred <- pred %>%
        mutate(
            binary_c = ifelse(final_prob >= threshold, 1, 0),
            dataset_label = as.character(dataset_label)
        )
    calc_sen <- function(df) {
        n <- sum(df$Target == 1, na.rm = TRUE)
        detected <- sum(df$Target == 1 & df$binary_c == 1, na.rm = TRUE)
        if (n == 0) {
            return(data.frame(Freq = 0, Detected = 0, SEN = 0, SEN.low = 0, SEN.up = 0))
        }
        exact_prop <- detected / n
        se.low <- exact_prop - 1.96 * sqrt(exact_prop * (1 - exact_prop) / n)
        se.up  <- exact_prop + 1.96 * sqrt(exact_prop * (1 - exact_prop) / n)
        se.low <- max(se.low, 0)
        se.up  <- min(se.up, 1)
        data.frame(Freq = n, Detected = detected, SEN = exact_prop, SEN.low = se.low, SEN.up = se.up)
    }
    make_row <- function(label, df) {cbind(Var1 = label, calc_sen(df))}
    dataset_levels <- names(table(pred$dataset_label))
    summary <- bind_rows(
        make_row("Total", pred),
        make_row("earlyGC", pred %>% filter(Target == 0 | Stage_cmb == "early GC")),
        make_row("advGC", pred %>% filter(Target == 0 | Stage_cmb == "advanced GC")),
        make_row("II", pred %>% filter(Target == 0 | Stage == "II")),
        make_row("III", pred %>% filter(Target == 0 | Stage == "III")),
        make_row("Intestinal", pred %>% filter(Target == 0 | `Lauren classification` == "Intestinal")),
        make_row("Diffuse", pred %>% filter(Target == 0 | `Lauren classification` == "Diffuse")),
        make_row("Mix", pred %>% filter(Target == 0 | `Lauren classification` == "Mix")),
        make_row("Missing", pred %>% filter(Target == 0 | `Lauren classification` == "Missing")),
        bind_rows(lapply(dataset_levels, function(x) {
            make_row(x, pred %>% filter(dataset_label == x))
        }))
    ) %>%
        mutate(SEN = round(SEN * 100, 0), 
            SEN.low = round(SEN.low * 100, 0), # SEN.low * 100, #
            SEN.up = round(SEN.up * 100, 0), # SEN.up * 100, #
            SEN.up = ifelse(SEN.up > 100, 100, SEN.up),
            CI95 = str_c(SEN.low, "-", SEN.up), SEN_95Ci = str_c(SEN, "(", CI95, ")"),
            perc = str_c(SEN, "%\n(", Detected, "/", Freq, ")"),
            Var1 = factor(Var1,
                levels = c("Total", "earlyGC", "advGC", "I", "II", "III",
                          "Intestinal", "Diffuse", "Mix", "Missing", dataset_levels))
        ) %>% arrange(Var1) %>%
        select(Var1, Freq, Detected, SEN, SEN.low, SEN.up, CI95, SEN_95Ci, perc)
    return(summary)
}

get_HDspecificity_inxspe <- function(pred, threshold) {
    library(dplyr); library(stringr)
    pred <- pred %>%
        mutate(
            binary_c = ifelse(final_prob >= threshold, 1, 0),
            dataset_label = as.character(dataset_label))
    calc_spe <- function(df) {
        n <- sum(df$Target == 0, na.rm = TRUE)
        detected <- sum(df$Target == 0 & df$binary_c == 0, na.rm = TRUE)
        if (n == 0) {
            return(data.frame(Freq = 0, Detected = 0, SPE = 0, SPE.low = 0, SPE.up = 0))
        }
        exact_prop <- detected / n
        spe.low <- exact_prop - 1.96 * sqrt(exact_prop * (1 - exact_prop) / n)
        spe.up  <- exact_prop + 1.96 * sqrt(exact_prop * (1 - exact_prop) / n)
        spe.low <- max(spe.low, 0)
        spe.up  <- min(spe.up, 1)
        data.frame(Freq = n, Detected = detected, SPE = exact_prop, SPE.low = spe.low, SPE.up = spe.up)
    }

    make_row <- function(label, df) {cbind(Var1 = label, calc_spe(df))}
    dataset_levels <- names(table(pred$dataset_label))
    atrophic_levels <- names(table(pred$Atrophic))
    im_levels <- names(table(pred$IntestinalMetaplasia))
    hp_levels <- names(table(pred$`Helicobacter pylori`))
    summary <- bind_rows(
        make_row("Total", pred),
        bind_rows(lapply(dataset_levels, function(x) {
            make_row(x, pred %>% filter(dataset_label == x))
        })),
        bind_rows(lapply(atrophic_levels, function(x) {
            make_row(str_c("Atr_", x), pred %>% filter(Atrophic == x))
        })),
        bind_rows(lapply(im_levels, function(x) {
            make_row(str_c("IM_", x), pred %>% filter(IntestinalMetaplasia == x))
        })),
        bind_rows(lapply(hp_levels, function(x) {
            make_row(str_c("HP_", x), pred %>% filter(`Helicobacter pylori` == x))
        }))
    ) %>% mutate(SPE = round(SPE * 100, 0), 
            SPE.low = round(SPE.low * 100, 0), # SPE.low * 100, #
            SPE.up = round(SPE.up * 100, 0), # SPE.up * 100, #
            SPE.up = ifelse(SPE.up > 100, 100, SPE.up),
            CI95 = str_c(SPE.low, "-", SPE.up), SPE_95Ci = str_c(SPE, "(", CI95, ")"),
            perc = str_c(SPE, "%\n(", Detected, "/", Freq, ")"),
            Var1 = factor(Var1,
                levels = c("Total", dataset_levels, str_c("Atr_", atrophic_levels),
                            str_c("IM_", im_levels), str_c("HP_", hp_levels)))
        ) %>% arrange(Var1) %>%
        select(Var1, Freq, Detected, SPE, SPE.low, SPE.up, CI95, SPE_95Ci, perc)
    return(summary)
}

get_performance_stats <- function(dat) {
  dat <- dat[!is.na(dat$Target) & !is.na(dat$predicted), ]

  calc_ci <- function(x, n) {
    if (n == 0) return(c(est=NA, low=NA, up=NA))
    p <- x / n
    se <- sqrt(p * (1-p) / n)
    c(est=p, low=max(0, p-1.96*se), up=min(1, p+1.96*se))
  }

  n_pos <- sum(dat$Target == 1)
  tp <- sum(dat$Target == 1 & dat$predicted == 1)
  sen <- calc_ci(tp, n_pos)

  n_neg <- sum(dat$Target == 0)
  tn <- sum(dat$Target == 0 & dat$predicted == 0)
  spe <- calc_ci(tn, n_neg)

  list(n_pos=n_pos, tp=tp, sen=sen, n_neg=n_neg, tn=tn, spe=spe)
}

make_perf_row <- function(perf, subgroup, metric) {
  if (metric == "Sensitivity") {
    ci <- perf$sen
    lab <- paste0("(n = ", perf$n_pos, ")")
    tag <- paste0("(",perf$tp,"/",perf$n_pos,")")
  } else {
    ci <- perf$spe
    lab <- paste0("(n = ", perf$n_neg, ")")
    tag <- paste0("(",perf$tn,"/",perf$n_neg,")")
  }

  data.frame(
    Subgroup = subgroup,
    Metric = metric,
    Estimate = round(ci["est"] * 100, 2),
    Lower = round(ci["low"] * 100, 2),
    Upper = round(ci["up"] * 100, 2),
    Lab = lab,
    Tag = tag
  )
}

get_performance_by_variable <- function(pred, variable, variable_cutoff, prob_cutoff, direction=">") {
  library(dplyr)
  x <- pred[[variable]]
  keep <- if (direction == ">") {
    x > variable_cutoff
  } else if (direction == ">=") {
    x >= variable_cutoff
  } else if (direction == "<") {
    x < variable_cutoff
  } else {
    x <= variable_cutoff
  }

  dat <- pred %>% filter(keep, !is.na(final_prob), !is.na(Target)) %>% mutate(predicted=ifelse(final_prob >= prob_cutoff, 1, 0))
  perf <- get_performance_stats(dat)

  data.frame(
    Subgroup=paste(variable, direction, variable_cutoff),
    N=nrow(dat),
    Sensitivity=round(perf$sen["est"]*100, 1),
    SEN.low=round(perf$sen["low"]*100,1),
    SEN.up=round(perf$sen["up"]*100,1),
    SEN_n=paste0('(n = ',perf$n_pos,')'),
    SEN_lab=paste0(round((perf$tp/perf$n_pos)*100,0),'%\n(',paste0(perf$tp, "/", perf$n_pos),')'),
    Specificity=round(perf$spe["est"]*100, 1),
    SPE.low=round(perf$spe["low"]*100,1),
    SPE.up=round(perf$spe["up"]*100,1),
    SPE_n=paste0('(n = ',perf$n_neg,')'),
    SPE_lab=paste0(round((perf$tn/perf$n_neg)*100,0),'%\n(',paste0(perf$tn, "/", perf$n_neg),')')
  )
}

cor_test_one <- function(data, xvar, yvar = score_var) {
  tmp <- data %>%
    select(all_of(c(xvar, yvar))) %>%
    mutate(across(everything(), as.numeric)) %>%
    filter(if_all(everything(), ~ !is.na(.) & is.finite(.)))
  if (nrow(tmp) < 3) {
    return(data.frame(variable = xvar,cor = NA_real_,pvalue = NA_real_,n = nrow(tmp)))
  }
  test <- suppressWarnings(
    cor.test(tmp[[xvar]],tmp[[yvar]],method = "spearman",exact = FALSE))
  data.frame(variable = xvar,cor = unname(test$estimate),pvalue = test$p.value,n = nrow(tmp))
}

font_normal = 8
fig5_theme_common <- function(base = c("classic", "bw"), base_size = font_normal,
                              base_family = "", border_width = 0.2,
                              axis_line = TRUE, legend_position = "none",
                              plot_margin = margin(t = 10, r = 5, b = 12, l = 5)) {
	base <- match.arg(base)
	base_theme <- if (base == "bw") {
		theme_bw(base_size = base_size, base_family = base_family)
	} else {
		theme_classic(base_size = base_size, base_family = base_family)
	}
	base_theme +
		theme(
			legend.position = legend_position,
			legend.title = element_blank(),
			legend.text = element_text(size = 6, color = "black"),
			legend.background = element_blank(), 
			axis.text = element_text(color = "black", size = 6),
			axis.title = element_text(color = "black", size = font_normal),
			plot.title = element_text(hjust = 0.5, size = font_normal, color = "black"),
			panel.border = element_rect(color = "black", fill = NA, linewidth = border_width),
			axis.line = if (axis_line) element_line(color = "black", linewidth = 0.2, lineend = "square") else element_blank(),
			axis.ticks = element_line(color = "black", linewidth = 0.2),
			strip.background = element_blank(),
			strip.text = element_text(size = 6, color = "black"),
			panel.grid.major = element_blank(),
			panel.grid.minor = element_blank(),
			plot.margin = plot_margin
		)
}

run_univariate_glm <- function(data, variable) {
  dat_uni <- data %>%
    select(sample_id, disease_status, disease_numeric, all_of(variable)) %>%
    drop_na()
  if (is.character(dat_uni[[variable]]) || is.factor(dat_uni[[variable]])) {
    dat_uni[[variable]] <- factor(dat_uni[[variable]])
  }

  if (length(unique(dat_uni$disease_numeric)) < 2) {
    return(data.frame())
  }

  form <- reformulate(variable, response = "disease_numeric")
  fit <- glm(form, data = dat_uni, family = binomial())

  broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      variable = variable,
      n = nrow(dat_uni),
      event_n = sum(dat_uni$disease_numeric == 1),
      non_event_n = sum(dat_uni$disease_numeric == 0),
      OR_95CI = sprintf("%.2f (%.2f-%.2f)", estimate, conf.low, conf.high),
      P_value = p.value,
      significance = case_when(
        p.value < 0.001 ~ "***",
        p.value < 0.01 ~ "**",
        p.value < 0.05 ~ "*",
        TRUE ~ ""
      )
    ) %>%
    select(variable, term, n, event_n, non_event_n, OR = estimate, CI_lower = conf.low, CI_upper = conf.high, OR_95CI, P_value, significance)
}

## Function: GLM + OR extraction
run_cluster_glm <- function(data,positive_groups,negative_group = "Control",comparison_name,glm_vars) {
  term_mapping <- c()
  dat_model <- data %>%
    filter(disease_status %in% c(positive_groups, negative_group)) %>%
    mutate(
      disease_status = factor(disease_status, levels = c(negative_group, positive_groups)),
      disease_numeric = case_when(
        disease_status == "Non-GC" ~ 0, disease_status == "GC" ~ 1,
        TRUE ~ NA_real_
      ),
      sex = factor(sex)
    ) %>%
    select(sample_id, disease_status, disease_numeric, all_of(glm_vars)) %>%
    drop_na()

  if (length(unique(dat_model$disease_numeric)) < 2) {
    stop("Only one outcome class remains after missing-value removal.")
  }

  categorical_vars <- glm_vars[sapply(dat_model[, glm_vars, drop = FALSE], function(x) is.factor(x) || is.character(x))]
  dat_model <- dat_model %>%
    mutate(across(any_of(categorical_vars), ~ factor(.x)))
  continuous_vars <- setdiff(glm_vars, categorical_vars)
  form <- reformulate(glm_vars, response = "disease_numeric")
  print(form)
  fit <- glm(form, data = dat_model, family = binomial())
  model_name <- "Multivariable-adjusted logistic regression"

  dat_model$pred <- predict(fit,type="response")

  res_base <- broom::tidy(fit, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      comparison = comparison_name,
      model = model_name,
      term_label = ifelse(term %in% names(term_mapping), term_mapping[term], term),
      significance = case_when(p.value < 0.001 ~ "***", p.value < 0.01 ~ "**", p.value < 0.05 ~ "*", TRUE ~ "")
    ) %>%
    rename(P_value = p.value, beta = estimate, beta_CI_lower = conf.low, beta_CI_upper = conf.high)

  or_table <- res_base %>%
    mutate(
      OR = exp(beta), CI_lower = exp(beta_CI_lower), CI_upper = exp(beta_CI_upper),
      OR_95CI = sprintf("%.2f (%.2f-%.2f)", OR, CI_lower, CI_upper),
      log_OR = beta, log_CI_lower = beta_CI_lower, log_CI_upper = beta_CI_upper
    ) %>%
    select(-beta, -beta_CI_lower, -beta_CI_upper)

  return(list(data = dat_model, fit = fit, or_table = or_table, formula = deparse(form)))
}


make_hp_perf_data <- function(df, metric) {
  res <- do.call(rbind, lapply(c("No", "Yes"), function(hp_stat) {
    do.call(rbind, lapply(source_levels, function(src) {
      tmp <- df[df$HP == hp_stat & df$source_key == src, ]
      if (nrow(tmp) == 0) {
        return(data.frame(HP = hp_stat, Source = src, Rate = 0, Low = 0, Up = 0, Label = ""))
      }
      tmp$predicted <- ifelse(tmp$final_prob >= cutoff_spe90, 1, 0)
      perf <- get_performance_stats(tmp)
      perf_row <- make_perf_row(perf, hp_stat, metric)
      data.frame(
        HP = hp_stat,
        Source = src,
        Rate = perf_row$Estimate,
        Low = perf_row$Lower,
        Up = perf_row$Upper,
        Label = paste0(round(perf_row$Estimate, 0), "%\n", perf_row$Tag)
      )
    }))
  }))
  res$HP <- factor(res$HP, levels = c("No", "Yes"))
  res$Source <- factor(res$Source, levels = source_levels)
  res
}

make_age_cor <- function(df, pos, show_y_axis, title_text) {
  df_tmp <- df[!is.na(df$Age), c("Sample", "final_prob", "source_key", "Group", "Target", "Age")]
  df_tmp$Age <- as.numeric(df_tmp$Age)
  if (pos == "bottom") {
    df_tmp$source_key <- factor(df_tmp$source_key, levels = c("Test", "Discovery"))
  }
  p <- ggscatter(
    df_tmp, x = "Age", y = "final_prob", size = 0.8, color = "source_key", combine = TRUE,
    cor.coef.size = 2.5, xlab = "Age (year)", ylab = "rbcDNA predictive score"
  ) +
    ylim(0, 1.05) +
    stat_cor(aes(color = source_key), method = "spearman", label.y.npc = pos,
             label.x.npc = 0.05, size = 6 / .pt, show.legend = FALSE) +
    scale_color_manual(values = color_mapping[levels(df_tmp$source_key)]) +
    theme_cor +
    theme(legend.position = "none") +
    ggtitle(title_text)
  if (!show_y_axis) {
    p <- p + theme_cor
  }
  p
}
