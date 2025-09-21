library(ggpubr)
library(rstatix)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(ggridges)
library(patchwork)
library(RColorBrewer)

# ============================
# Función para cargar y procesar datos de cada banda
# ============================
get_df <- function(band = "gamma") {
  # Cargamos los CSVs
  df_10 <- read_csv(paste0("resultados_rqa_", band, "/carga_10_", band, ".csv"))
  df_5  <- read_csv(paste0("resultados_rqa_", band, "/carga_5_", band, ".csv"))
  df_0  <- read_csv(paste0("resultados_rqa_", band, "/carga_0_", band, ".csv"))
  
  # Función para filtrar y clasificar en periodos
  filtro_columnas <- function(df) {
    df %>%
      filter(determinism != 0) %>%
      mutate(
        periodo = case_when(
          Binaria > 0.5 ~ "Actividad",
          Binaria > -0.5 & Binaria < 0.5 ~ "Rest",
          TRUE ~ NA_character_
        )
      ) %>%
      filter(!is.na(periodo)) %>%
      select(sujeto, sesion, condicion, periodo, everything(), -banda, -archivo)
  }
  
  df_10 <- filtro_columnas(df_10)
  df_5  <- filtro_columnas(df_5)
  df_0  <- filtro_columnas(df_0)
  
  # Concatenamos
  df <- bind_rows(df_10, df_5, df_0)
  df$band <- band
  
  # Promediar por sujeto, sesión, condición y periodo
  df <- df %>%
    group_by(sujeto, sesion, condicion, periodo, band) %>%
    summarise(across(where(is.numeric), mean, .names = "mean_{.col}"), .groups = "drop")
  
  return(df)
}

# ============================
# Función para renombrar condiciones
# ============================
rename_conditions <- function(df) {
  df %>%
    mutate(condicion = case_when(
      condicion == "10% Torque" ~ "SE10",
      condicion == "5% Torque"  ~ "SE5",
      condicion == "0% Torque"  ~ "AM",
      TRUE ~ condicion
    ))
}

# ============================
# Cargar bandas
# ============================
big_mu    <- rename_conditions(get_df("mu"))
big_beta  <- rename_conditions(get_df("beta"))
big_gamma <- rename_conditions(get_df("gamma"))

big_all <- bind_rows(big_mu, big_beta, big_gamma)



bands <- list(
  mu    = list(df = big_mu,    palette = "Reds"),
  beta  = list(df = big_beta,  palette = "Greens"),
  gamma = list(df = big_gamma, palette = "Blues")
)

column = cols[8]
print(column)
  all_values <- c(
  bands$mu$df[[column]],
  bands$beta$df[[column]],
  bands$gamma$df[[column]]
)

x_min <- min(all_values, na.rm = TRUE)
x_max <- max(all_values, na.rm = TRUE)
x_std <- sd(all_values, na.rm = TRUE)



plot_ridgeline_band_shades <- function(df, palette_name = "Reds", 
                                       col = "mean_recurrence_rate", 
                                       y_name = "Value", 
                                       band_name = "μ", 
                                       x_limits = c(0,1)) {
  
  df <- df %>%
    mutate(
      condicion = factor(condicion, levels = c("AM", "SE5", "SE10")),  
      fill_group = ifelse(periodo == "Actividad", as.character(condicion), "Rest"),
      fill_group = factor(fill_group, levels = c("SE10", "SE5", "AM", "Rest"))
    )
  
  fill_colors <- c(
    "Rest" = "grey91",  
    "SE10" = brewer.pal(9, palette_name)[7],
    "SE5"  = brewer.pal(9, palette_name)[5],
    "AM"   = brewer.pal(9, palette_name)[3]
  )
  
  ggplot(df, aes(x = .data[[col]], y = condicion, fill = fill_group)) +
    geom_density_ridges(alpha = 0.7, scale = 1.5, color = "black", linewidth = 0.4) +
    scale_fill_manual(
      values = fill_colors,
      name   = "Condition",
      breaks = c("Rest","SE10","SE5","AM"),   # Legend order
      labels = c("Rest","SE10","SE5","AM")
    ) +
    guides(fill = guide_legend(
      override.aes = list(size = 4),  # size of lines for line geoms (not ridgelines)
      keywidth = 1,                 # width of squares
      keyheight = 1                 # height of squares
    )) +
    labs(x = y_name, title = band_name) +
    theme_classic(base_size = 10, base_rect_size = 5) + 
    theme(
      axis.title.y   = element_blank(),
      axis.title.x   = element_blank(),
      legend.position = "right"
    ) +
    xlim(x_limits[1] - (x_limits[2] * (x_std) * 0.8),
         x_limits[2] + (x_limits[2] * (x_std) * 0.8))
}



plot_mu    <- plot_ridgeline_band_shades(bands$mu$df,    palette_name = bands$mu$palette,    col = column, y_name = "Recurrence Rate (μ)", band_name = "μ", x_limits = c(x_min, x_max))
plot_beta  <- plot_ridgeline_band_shades(bands$beta$df,  palette_name = bands$beta$palette,  col = column, y_name = "Recurrence Rate (β)", band_name = "β", x_limits = c(x_min, x_max))
plot_gamma <- plot_ridgeline_band_shades(bands$gamma$df, palette_name = bands$gamma$palette, col = column, y_name = "Recurrence Rate (γ)", band_name = "γ", x_limits = c(x_min, x_max))

combined_plot <- plot_mu / plot_beta / plot_gamma +
  plot_annotation(
    title = "TT",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0) 
    )
  )

# Show the plot
combined_plot



