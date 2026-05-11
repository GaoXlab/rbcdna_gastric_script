args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
    library(openxlsx)
    library(tableone)
})

load('./Figures/sampleinfo.RData')

# 定义人群标签
sampleinfo$group.label <- ""
sampleinfo$group.label[sampleinfo$Dataset == 'Dataset A, discovery cohort'] <- 'Discovery'
sampleinfo$group.label[sampleinfo$Dataset == 'Dataset A, test cohort'] <- 'Test'
sampleinfo$group.label[sampleinfo$Dataset == 'Dataset B'] <- 'Validation 1'
sampleinfo$group.label[sampleinfo$Source == 'ANYANG'] <- 'Validation 2'
sampleinfo$group.label[sampleinfo$Source == 'SHANDONG'] <- 'Validation 3'

sampleinfo$Group <- factor(sampleinfo$Group, levels = c('Non-GC', 'GC'))
sampleinfo$group.label <- factor(sampleinfo$group.label, levels = c('Discovery', 'Test', 'Validation 1', 'Validation 2', 'Validation 3'))

# 变量列表
vars <- c('Age', 'Gender', 'Ethnicity', 'Smoking status', 'Alcohol status', 'Stage',
          'Lauren classification', 'Pathological type', 'Tumor size group', 'Gastritis',
          'Atrophic', 'IntestinalMetaplasia', 'Helicobacter pylori',
          'RBC (×10^12/L)', 'HGB (g/L)', 'WBC (×10^9/L)', 'PLT (×10^9/L)',
          'CEA (ng/mL)', 'CA19-9 (U/mL)', 'CA242 (U/mL)')

# 指定进行 wilcox.test 的连续变量
target_vars <- c('RBC (×10^12/L)', 'HGB (g/L)', 'WBC (×10^9/L)', 'PLT (×10^9/L)',
                 'CEA (ng/mL)', 'CA19-9 (U/mL)', 'CA242 (U/mL)')

# 指定需要增加 (n=?) 的变量
n_label_vars <- c('CEA (ng/mL)', 'CA19-9 (U/mL)', 'CA242 (U/mL)')

tumor_specific_vars <- c('Tumor size group', 'Pathological type', 'Lauren classification', 'Stage')


eps <- 1e-9
cont_vars <- c('Age', 'RBC (×10^12/L)', 'HGB (g/L)', 'WBC (×10^9/L)', 'PLT (×10^9/L)',
               'CEA (ng/mL)', 'CA19-9 (U/mL)', 'CA242 (U/mL)')

for (v in cont_vars) {
    if (v %in% colnames(sampleinfo) && is.numeric(sampleinfo[[v]])) {
        sampleinfo[[v]] <- sampleinfo[[v]] + eps
    }
}

group_labels <- levels(sampleinfo$group.label)
table_list <- list()

for (lab in group_labels) {
    sub_data <- sampleinfo[sampleinfo$group.label == lab, ]
    if (nrow(sub_data) > 0) {
        tab <- CreateTableOne(vars = vars, data = sub_data, strata = "Group", test = TRUE)
        tab_mat <- print(tab, showAllLevels = TRUE, quote = FALSE, noSpaces = TRUE, printToggle = FALSE, formatOptions = list(big.mark = ","))

        # 1. 替换 P 值逻辑 (使用 wilcox.test)
        if ("p" %in% colnames(tab_mat)) {
            for (var in target_vars) {
                row_idx <- grep(var, rownames(tab_mat), fixed = TRUE)
                if (length(row_idx) > 0) {
                    actual_row_name <- rownames(tab_mat)[row_idx[1]]
                    g1 <- sub_data[[var]][sub_data$Group == 'Non-GC']
                    g2 <- sub_data[[var]][sub_data$Group == 'GC']
                    if (length(na.omit(g1)) > 0 && length(na.omit(g2)) > 0) {
                        wt <- wilcox.test(g1, g2, exact = FALSE, correct = TRUE)
                        pval <- wt$p.value
                        tab_mat[actual_row_name, "p"] <- if (pval < 0.001) "<0.001" else sprintf("%.3f", pval)
                    }
                }
            }
        }

        for (var in n_label_vars) {
            row_idx <- grep(var, rownames(tab_mat), fixed = TRUE)
            if (length(row_idx) > 0) {
                actual_row_name <- rownames(tab_mat)[row_idx[1]]

                n_non_gc <- sum(!is.na(sub_data[sub_data$Group == 'Non-GC', var]))
                n_gc <- sum(!is.na(sub_data[sub_data$Group == 'GC', var]))

                if ("Non-GC" %in% colnames(tab_mat)) {
                    tab_mat[actual_row_name, "Non-GC"] <- paste0("(n=", n_non_gc, ") ", tab_mat[actual_row_name, "Non-GC"])
                }
                if ("GC" %in% colnames(tab_mat)) {
                    tab_mat[actual_row_name, "GC"] <- paste0("(n=", n_gc, ") ", tab_mat[actual_row_name, "GC"])
                }
            }
        }

        table_list[[lab]] <- tab_mat
    }
}

levels_col <- NULL
if (length(table_list) > 0 && "level" %in% colnames(table_list[[1]])) {
    levels_col <- table_list[[1]][, "level", drop = FALSE]
}

for (i in seq_along(table_list)) {
    cols_to_remove <- c("level", "test")
    if (names(table_list)[i] %in% c('Validation 1', 'Validation 2', 'Validation 3')) {
        cols_to_remove <- c(cols_to_remove, "p")
    }
    cols_to_keep <- !colnames(table_list[[i]]) %in% cols_to_remove
    table_list[[i]] <- table_list[[i]][, cols_to_keep, drop = FALSE]
}

final_mat <- do.call(cbind, table_list)
final_df <- as.data.frame(final_mat, stringsAsFactors = FALSE)

if (!is.null(levels_col)) {
    final_df <- cbind(Variable = rownames(final_mat), Level = as.character(levels_col[, 1]), final_df, stringsAsFactors = FALSE)
} else {
    final_df <- cbind(Variable = rownames(final_mat), final_df, stringsAsFactors = FALSE)
}

colnames(final_df) <- gsub(".*\\.", "", colnames(final_df))

non_gc_cols <- which(colnames(final_df) == "Non-GC")
current_var <- ""

for (i in 1:nrow(final_df)) {
    row_var_label <- as.character(final_df$Variable[i])
    if (row_var_label != "") {
        clean_name <- trimws(gsub(" \\(%\\)| \\(mean \\(SD\\)\\)", "", row_var_label))
        if (clean_name %in% vars) {
            current_var <- clean_name
        }
    }

    if (current_var %in% tumor_specific_vars) {
        final_df[i, non_gc_cols] <- "-"
    }
}

# --- 创建 Excel ---
wb <- createWorkbook()
addWorksheet(wb, "Table 1")

header_style <- createStyle(halign = "center", valign = "center", textDecoration = "bold", border = "TopBottomLeftRight")
sub_header_style <- createStyle(halign = "center", textDecoration = "bold", border = "bottom")
body_style <- createStyle(halign = "left")

writeData(wb, "Table 1", final_df, startRow = 2)

current_col <- if (!is.null(levels_col)) 3 else 2
for (lab in names(table_list)) {
    n_cols <- ncol(table_list[[lab]])
    mergeCells(wb, "Table 1", cols = current_col:(current_col + n_cols - 1), rows = 1)
    writeData(wb, "Table 1", lab, startCol = current_col, startRow = 1)
    addStyle(wb, "Table 1", style = header_style, rows = 1, cols = current_col:(current_col + n_cols - 1), gridExpand = TRUE)
    current_col <- current_col + n_cols
}

addStyle(wb, "Table 1", style = sub_header_style, rows = 2, cols = 1:ncol(final_df))
addStyle(wb, "Table 1", style = body_style, rows = 3:(nrow(final_df) + 2), cols = 1:ncol(final_df), gridExpand = TRUE)

setColWidths(wb, "Table 1", cols = 1, widths = 35)
setColWidths(wb, "Table 1", cols = 2:ncol(final_df), widths = 18)

saveWorkbook(wb, file.path(out_dir, 'Table1.xlsx'), overwrite = TRUE)