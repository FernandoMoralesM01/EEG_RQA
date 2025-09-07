library(dplyr)
library(readr)
library(ggplot2)
library(patchwork)
library(ggforce)  # para geom_sina (jitter elegante)

# === Cargar CSVs ===
df_mu    <- read_csv("resultados_rqa_unitario/mu.csv")
df_beta  <- read_csv("resultados_rqa_unitario/beta.csv")
df_gamma <- read_csv("resultados_rqa_unitario/gamma.csv")

# === Calcular longitud mínima ===
minlen <- min(nrow(df_mu), nrow(df_beta), nrow(df_gamma))

# === Recortar todos al mismo tamaño ===
df_mu    <- df_mu[1:minlen, ]
df_beta  <- df_beta[1:minlen, ]
df_gamma <- df_gamma[1:minlen, ]

# === Crear columna de tiempo ===
time <- seq(0, by = 1/1.4, length.out = minlen)

df_mu    <- df_mu    %>% mutate(time = time)
df_beta  <- df_beta  %>% mutate(time = time)
df_gamma <- df_gamma %>% mutate(time = time)

# === Unir todo con etiquetas griegas ===
df_all <- bind_rows(
  list("μ" = df_mu, "β" = df_beta, "γ" = df_gamma),
  .id = "Banda"
)

#df_all <- df_all %>%
  #mutate(Bin = ifelse(Bin > 0.5, 1, 0))

# Factor para ordenar
df_all$Banda <- factor(df_all$Banda, levels = c("μ", "β", "γ"))

# === Paleta de colores por banda ===
colores <- c("μ" = "#DE2D26",   # Reds
             "β" = "#31A354",   # Greens
             "γ" = "#3182BD")   # Blues

# === Detectar intervalos de reposo (Bin = 0) ===
bin_df <- df_all %>%
  select(time, Bin) %>%
  distinct() %>%
  arrange(time) %>%
  mutate(change = Bin != lag(Bin, default = first(Bin)),
         group = cumsum(change))

activity_intervals <- bin_df %>%
  group_by(group, Bin) %>%
  summarise(xmin = min(time), xmax = max(time), .groups = "drop") %>%
  filter(Bin > 0.5)

rest_intervals <- bin_df %>%
  group_by(group, Bin) %>%
  summarise(xmin = min(time), xmax = max(time), .groups = "drop") %>%
  filter(Bin < 0.5 & Bin > -0.5)

no_intervals <- bin_df %>%
  group_by(group, Bin) %>%
  summarise(xmin = min(time), xmax = max(time), .groups = "drop") %>%
  filter(Bin < -0.5)


# === Gráfico de líneas con franjas de actividad ===
p1 <- ggplot(df_all, aes(x = time, y = recurrence_rate, color = Banda)) +
  geom_rect(data = activity_intervals, inherit.aes = FALSE,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            fill = "#DFDFDC", alpha = 0.5) +
  geom_rect(data = rest_intervals, inherit.aes = FALSE,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            fill = "#827D82", alpha = 0.5) +
  geom_line(size = 1, show.legend = FALSE) +
  scale_color_manual(values = colores) +
  labs(y = "Recurrence Rate", x = "Time (s)")
  theme_minimal(base_size = 14)


# === Boxplots diferenciando reposo vs actividad ===
p2 <- ggplot(df_all, aes(x = Banda, y = recurrence_rate, fill = Banda)) +
  # Reposo
  geom_boxplot(data = df_all %>% filter(Bin == 0),
               aes(group = interaction(Banda, Bin), fill = Banda),
               position = position_nudge(x = -0.2),
               alpha = 0.5, color = "black", show.legend = FALSE, width = 0.25) +
  # Actividad
  geom_boxplot(data = df_all %>% filter(Bin == 1),
               aes(group = interaction(Banda, Bin), fill = Banda),
               position = position_nudge(x = 0.2),
               alpha = 0.5, color = "black", show.legend = FALSE, width = 0.25) +
  scale_fill_manual(values = colores) +
  scale_color_manual(values = colores) +
  labs(y = NULL,
       x = "Band") +   # 👈 Y label para boxplots
  theme_minimal(base_size = 14)
  
# === Combinar gráficos ===
p1 + p2 + plot_layout(widths = c(3, 1))


