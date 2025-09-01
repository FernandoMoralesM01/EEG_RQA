library(patchwork)
library(ggpubr)
library(rstatix)
library(tidyverse)
library(ggstatsplot)
library(ggbeeswarm)

# Función para cargar y procesar datos de cada banda
get_df <- function(band = "gamma") {
  df_10 <- read_csv(paste0("resultados_rqa_", band, "/carga_10_", band, ".csv"))
  df_5  <- read_csv(paste0("resultados_rqa_", band, "/carga_5_", band, ".csv"))
  df_0  <- read_csv(paste0("resultados_rqa_", band, "/carga_0_", band, ".csv"))
  
  # Filtramos
  df_10 <- df_10 %>% filter(determinism != 0, Binaria > 0.5) %>% 
    select(-banda, -archivo, -sujeto, -sesion)
  df_5 <- df_5 %>% filter(determinism != 0, Binaria > 0.5) %>% 
    select(-banda, -archivo, -sujeto, -sesion)
  df_0 <- df_0 %>% filter(determinism != 0, Binaria > 0.5) %>% 
    select(-banda, -archivo, -sujeto, -sesion)
  
  # Concatenamos
  df <- bind_rows(df_10, df_5, df_0)
  df$band <- band
  return(df)
}

# Función para renombrar condiciones (elimina repetición de código)
rename_conditions <- function(df) {
  df %>%
    mutate(condicion = case_when(
      condicion == "10% Torque" ~ "SE10",
      condicion == "5% Torque"  ~ "SE5",
      condicion == "0% Torque"  ~ "AM",
      TRUE ~ condicion
    ))
}

# Cargamos todas las bandas
big_mu <- get_df("mu")
big_beta <- get_df("beta")
big_gamma <- get_df("gamma")

# Aplicamos el renombramiento a todas las bandas
big_mu <- rename_conditions(big_mu)
big_beta <- rename_conditions(big_beta)
big_gamma <- rename_conditions(big_gamma)

# Combinamos todos los datos
big_all <- bind_rows(big_mu, big_beta, big_gamma)

  # Función con pruebas estadísticas (CORREGIDA)
plot_violins_side_by_side <- function(big_mu, big_beta, big_gamma, 
                                        label = "condicion", col = "recurrence_rate", paired = FALSE) {
    
    bands <- list(
      mu    = list(df = big_mu,    palette = "Reds"),
      beta  = list(df = big_beta,  palette = "Greens"),
      gamma = list(df = big_gamma, palette = "Blues")
    )
    
    make_plot <- function(df, palette, band, show_ylabel = FALSE) {
    
    
      # Convertir a factor con orden específico (descomentado)
      
      #comps <- combn(unique(df[[label]]), 2, simplify = FALSE)
      df[[label]] <- factor(df[[label]], levels = c("SE10", "SE5", "AM"))
      my_comparisons <- list( c("SE10", "SE5"), c("SE5", "AM"), c("SE10", "AM") )
      ggplot(df, aes_string(x = label, y = col, fill = label)) +
        geom_violin(trim = FALSE, alpha = 1, show.legend = FALSE, adjust = 1) +
        geom_boxplot(width = 0.15, alpha = 0.6, show.legend = FALSE, 
                     fill = "white", outlier.shape = 1, outlier.alpha = 0.3, 
                     outlier.size = 2) +
        #geom_beeswarm(corral.width = 0.1, col = label)+
        scale_fill_brewer(palette = palette) +
        stat_compare_means(comparisons = my_comparisons, method = "t.test",
                           label = "p.signif", hide.ns = TRUE, show.legend = TRUE, paried=paired) +
        stat_compare_means(label.y = max(df[[col]] + 0.5, na.rm = TRUE)) +
        theme_classic(base_size = 10) +
        theme(
          legend.position = "none",
          axis.title.x = element_blank(),
          axis.title.y = if (show_ylabel) element_text(size = 14) else element_blank()
        ) +
        labs(title = band, y = if (show_ylabel) "Recurrence Rate" else NULL)
    }
    
    p_mu    <- make_plot(bands$mu$df,    bands$mu$palette,    "μ", show_ylabel = TRUE)
    p_beta  <- make_plot(bands$beta$df,  bands$beta$palette,  "β", show_ylabel = FALSE)
    p_gamma <- make_plot(bands$gamma$df, bands$gamma$palette, "γ", show_ylabel = FALSE)
    
    combined <- patchwork::wrap_plots(p_mu, p_beta, p_gamma) + plot_layout(guides = "collect")
    
    # AGREGADO: return para que la función devuelva el plot
    return(combined)
  }
  
  # Ejemplo de llamada
  plot_result <- plot_violins_side_by_side(big_mu, big_beta, big_gamma, 
                                           label = "condicion", col = "recurrence_rate")
  
  # Mostrar el plot
  plot_result

# ANOVA (opcional - análisis estadístico adicional)
# anova_result <- aov(recurrence_rate ~ condicion * band, data = big_all)
# summary(anova_result)



big_mu %>% 
  group_by(condicion) %>% 
  summarise(
    n = n(),
    media = mean(recurrence_rate, na.rm = TRUE),
    mediana = median(recurrence_rate, na.rm = TRUE),
    sd = sd(recurrence_rate, na.rm = TRUE),
    min = min(recurrence_rate, na.rm = TRUE),
    max = max(recurrence_rate, na.rm = TRUE)
  )
  



# Promediando cada sesion por persona

library(dplyr)
library(readr)

# Función para cargar y procesar datos de cada banda
get_df <- function(band = "gamma") {
  # Cargamos los CSVs
  df_10 <- read_csv(paste0("resultados_rqa_", band, "/carga_10_", band, ".csv"))
  df_5  <- read_csv(paste0("resultados_rqa_", band, "/carga_5_", band, ".csv"))
  df_0  <- read_csv(paste0("resultados_rqa_", band, "/carga_0_", band, ".csv"))
  
  
  
  # Filtramos y seleccionamos columnas necesarias
  filtro_columnas <- function(df) {
    df %>%
      filter(determinism != 0, Binaria > 0.5) %>%
      select(sujeto, sesion, condicion, everything(), -banda, -archivo)
  }
  
  df_10 <- filtro_columnas(df_10)
  df_5  <- filtro_columnas(df_5)
  df_0  <- filtro_columnas(df_0)
  
  # Concatenamos
  df <- bind_rows(df_10, df_5, df_0)
  df$band <- band
  
  # Promediar por sujeto, sesión y condición
  df <- df %>%
    group_by(sujeto, sesion, condicion, band) %>%
    summarise(across(where(is.numeric), mean, .names = "mean_{.col}"), .groups = "drop")
  
  return(df)
}



# Función para renombrar condiciones (elimina repetición de código)
rename_conditions <- function(df) {
  df %>%
    mutate(condicion = case_when(
      condicion == "10% Torque" ~ "SE10",
      condicion == "5% Torque"  ~ "SE5",
      condicion == "0% Torque"  ~ "AM",
      TRUE ~ condicion
    ))
}

# Cargamos todas las bandas
big_mu <- get_df("mu")
big_beta <- get_df("beta")
big_gamma <- get_df("gamma")

# Aplicamos el renombramiento a todas las bandas
big_mu <- rename_conditions(big_mu)
big_beta <- rename_conditions(big_beta)
big_gamma <- rename_conditions(big_gamma)

# Combinamos todos los datos
big_all <- bind_rows(big_mu, big_beta, big_gamma)

# Ejemplo de llamada
plot_result <- plot_violins_side_by_side(big_mu, big_beta, big_gamma, 
                                         label = "condicion", col = "mean_recurrence_rate", paired = TRUE)

# Mostrar el plot
plot_result

