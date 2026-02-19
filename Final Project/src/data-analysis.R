# Data analysis
library(ggplot2)
library(dplyr)
library(ggcorrplot)
library(caret)


# Load data
load("df_tidy.RData")

# Normalize Tempo
df_tidy <- df_tidy %>%
  mutate(tempo = (tempo - min(tempo)) / (max(tempo) - min(tempo)))


# Summary of features by genre
ggplot(df_tidy %>%
         tidyr::gather(feature, val, 7:14),
       aes(x = feature, y = val, colour = feature)) +
  geom_boxplot() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  theme(legend.position = "none") +
  ylab("Value") +
  facet_wrap(~ track_genre_top, nrow = 2)

# Plot histogram of bottom 95% of track_listens
ggplot(df_tidy %>% filter(track_listens < quantile(track_listens, 0.95)), aes(x = track_listens)) +
  geom_histogram(binwidth = 200, fill = "purple", color = "black") +
  theme_minimal() +
  labs(title = "Histogram of Track Listens (Filtered)", 
       x = "Track Listens", 
       y = "Frequency")

# Since track_listens is very skewed, we will log-transform it
df_tidy$log_track_listens <- log(df_tidy$track_listens)

percentiles <- quantile(df_tidy$log_track_listens, probs = c(0.025, 0.975))
ggplot(df_tidy, aes(x = log_track_listens)) +
  geom_histogram(fill = "purple", color = "black") +
  geom_vline(aes(xintercept = percentiles[1]), linetype = "dashed") +
  geom_vline(aes(xintercept = percentiles[2]), linetype = "dashed") +
  theme_minimal() +
  labs(title = "Histogram of Track Listens", 
       x = "Log Track Listens", 
       y = "Frequency") +
  theme(axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        title = element_text(size = 20))
ggsave("img/histogram.png", width = 3000, height = 2000, units = "px")

# Correlation matrix
numeric_features <- df_tidy %>% 
  select_if(is.numeric) %>% 
  select(-log_track_listens, -track_ID, -artist_id)
cor_matrix <- cor(numeric_features)

cor_features <- c("track_listens", "track_duration", "acousticness", "danceability", 
                  "energy", "instrumentalness", "liveness", "speechiness", "tempo", "valence")
cor_matrix <- cor_matrix[cor_features, cor_features]

ggcorrplot(cor_matrix, lab = TRUE, lab_size = 4, type = "lower")+
  theme(axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16))

# Filter the data to keep only the middle 95%
df_filtered <- df_tidy %>%
  filter(log_track_listens > percentiles[1], log_track_listens < percentiles[2])

# Normalize numeric features of filtered data
df_scaled <- df_filtered %>%
  select(track_duration, acousticness, danceability, energy, instrumentalness,
         liveness, speechiness, tempo, valence) %>%
  scale() %>%
  as.data.frame()

ggplot(df_filtered, aes(y = log_track_listens)) +
  geom_boxplot() +
  theme_classic()

ggplot(df_tidy, aes(y = track_listens)) +
  geom_boxplot() +
  theme_classic() +
  theme(
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank()
  )



# Run PCA ----------------------------------------------------------------------
pca_result <- prcomp(df_scaled, center = TRUE, scale. = TRUE)

# Summary of PCA to check variance explained by each principal component
summary(pca_result)

# Plot explained variance by each component
plot(pca_result, type = "l", main = "Scree Plot of PCA Components")
pca_result$rotation

# Extract the eigenvalues and calculate variance explained
eigenvalues <- pca_result$sdev^2
variance_explained <- eigenvalues / sum(eigenvalues) * 100
cumulative_variance_5 <- sum(eigenvalues[1:5]) / sum(eigenvalues) * 100

# Create a data frame for ggplot
df <- data.frame(
  PC = 1:length(variance_explained),
  VarianceExplained = variance_explained
)

# Plot with ggplot
p <- ggplot(df, aes(x = PC, y = VarianceExplained)) +
  geom_line(group = 1) +
  geom_point() +
  geom_vline(xintercept = 5, linetype = "dashed", color = "red") +
  annotate("text", x = 5.5, y = 20, 
           label = paste0(
             "Total variance explained (5 PCs): ", 
             round(cumulative_variance_5, 2), 
             "%"
            ),
           hjust = 0, 
           size = 6
           ) +
  scale_x_continuous(breaks = 1:9) +
  labs(
    x = "Principal Component",
    y = "Percentage of Variance Explained (%)",
    title = "Variance Explained by Principal Components"
  ) +
  theme(panel.grid = element_blank(), 
        panel.background = element_rect(fill = "white", color = NA), 
        panel.border = element_rect(fill = NA, color = "black", size = 1), 
        strip.background = element_rect(fill = "white", color = "black", size = 1), 
        strip.text = element_text(color = "black")) +
  theme(axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16), 
        title = element_text(size = 20))

# Save the plot
ggsave("img/variance_explained_plot.png",p, width = 3500, height = 2000, units = "px")


# Extract the first 5 principal components
df_pca <- as.data.frame(pca_result$x[, 1:5])

# Add the target variable to the reduced dataset
df_pca$track_listens <- df_filtered$track_listens
df_pca$log_track_listens <- df_filtered$log_track_listens


# Split train and test set -----------------------------------------------------
set.seed(123)
train_indices <- sample(1:nrow(df_pca), size = 0.8 * nrow(df_pca))
train_data <- df_pca[train_indices, ]
test_data <- df_pca[-train_indices, ]

# Random Forest Regression -----------------------------------------------------
library(randomForest)

# Use tuneRF() to refine the mtry parameter
tuned_rf <- tuneRF(
  train_data[, c("PC1", "PC2", "PC3", "PC4", "PC5")], # Predictors
  train_data$log_track_listens,                       # Target variable
  stepFactor = 2,                                     # Increment to try larger `mtry`
  ntreeTry = 500,                                     # Number of trees to try
  improve = 0.001,                                    # Minimum improvement to keep tuning
  trace = TRUE,                                       # Print progress
  plot = TRUE
)

rf_model_1 <- randomForest(
  log_track_listens ~ PC1 + PC2 + PC3 + PC4 + PC5,
  data = train_data,
  ntree = 500,      # Number of trees
  mtry = 1,         # Number of variables to try at each split
  importance = TRUE # Track feature importance
)

# rf_model_2 <- randomForest(
#   log_track_listens ~ PC1 + PC2 + PC3 + PC4 + PC5,
#   data = train_data,
#   ntree = 500,      # Number of trees
#   mtry = 2,         # Number of variables to try at each split
#   importance = TRUE # Track feature importance
# )

test_data$rf_model_1_pred <- predict(rf_model_1, test_data)
rf_model_1_mse <- mean((test_data$rf_model_1_pred - test_data$log_track_listens)^2)
sqrt(rf_model_1_mse)
rf_rmse <- sqrt(rf_model_1_mse)

# test_data$rf_model_2_pred <- predict(rf_model_2, test_data)
# rf_model_2_mse <- mean((test_data$rf_model_2_pred - test_data$log_track_listens)^2)
# sqrt(rf_model_2_mse)

# rf_model_1 (with mtry = 1) has lower RMSE

# Importance of PCA components
# Higher value for %IncMSE indicates higher importance for prediction accuracy
importance(rf_model_1) # Importance scores
varImpPlot(rf_model_1)  # Visualization


rf_model_graph <- as.data.frame(importance(rf_model_1))
rf_model_graph$PC <- rownames(rf_model_graph)
print(rf_model_graph)
ggplot(rf_model_graph, aes(x = reorder(PC, `%IncMSE`), y = `%IncMSE`)) +
  geom_bar(stat = "identity", aes(fill = PC)) +
  coord_flip() +
  labs(
    title = "Variable Importance (Random Forest)",
    x = "Principal Component",
    y = "% Increase in MSE", 
    fill = "Feature"
  ) +
  theme(
    panel.grid = element_blank(), 
    panel.background = element_rect(fill = "white", color = NA), 
    panel.border = ggplot2::element_rect(fill = NA, color = "black", size = 1), 
    strip.background = ggplot2::element_rect(fill = "white", color = "black", size = 1), 
    strip.text = element_text(color = "black"),
    title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )



# XGBoost
library(xgboost)

# Create training and testing matrices (train_x and test_x)
train_matrix <- as.matrix(train_data[, c("PC1", "PC2", "PC3", "PC4", "PC5")])
test_matrix <- as.matrix(test_data[, c("PC1", "PC2", "PC3", "PC4", "PC5")])

# Target variable (train_y and test_y)
train_target <- train_data$log_track_listens
test_target <- test_data$log_track_listens

# Define final training and testing sets
xgb_train <- xgb.DMatrix(data = train_matrix, label = train_target)
xgb_test <- xgb.DMatrix(data = test_matrix, label = test_target)

# Define watchlist to track performance
watchlist <- list(train = xgb_train, test = xgb_test)

# Train the model and monitor RMSE
set.seed(123)
model <- xgb.train(
  data = xgb_train,
  max.depth = 2,                  # Maximum tree depth
  eta = 0.1,                      # Learning rate
  nrounds = 200,                  # Total boosting rounds
  objective = "reg:squarederror", # Regression objective
  watchlist = watchlist,          # Track training and testing performance
  verbose = 1                     # Print RMSE for each round
)
min(model$evaluation_log$test_rmse)
model$evaluation_log$iter[which.min(model$evaluation_log$test_rmse)]
# Lowest Test RMSE with max_depth = 5: 1.047334 (iteration 53)
# Lowest Test RMSE with max_depth = 4: 1.04653  (iteration 91)
# Lowest Test RMSE with max_depth = 3: 1.046522 (iteration 137) 
# Lowest Test RMSE with max_depth = 2: 1.048087 (iteration 186)
# Use max_depth = 3 for lower test RMSE but without overfitting

# Define final model
final_model <- xgboost(
  data = xgb_train,
  max.depth = 3,
  eta = 0.1,
  nrounds = 137,
  objective = "reg:squarederror",
  verbose = 0
)

# Predict on test data
test_data$boost_pred <- predict(final_model, test_matrix)
boost_mse <- mean((test_data$boost_pred - test_target)^2)
boost_rmse <- sqrt(boost_mse)
boost_rmse

# Extract feature importance and visualize
xgb.importance(model = final_model)
xgb.plot.importance(xgb.importance(model = final_model))

# Extract feature importance
importance_matrix <- xgb.importance(model = final_model)

# Convert to a data frame
importance_df <- as.data.frame(importance_matrix)

# Create a ggplot barplot
ggplot(importance_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", aes(fill = Feature)) +
  coord_flip() +
  labs(
    title = "Variable Importance (XGBoost)",
    x = "Principal Component",
    y = "Gain (Importance)"
  ) +
  theme(
  panel.grid = element_blank(), 
  panel.background = element_rect(fill = "white", color = NA), 
  panel.border = ggplot2::element_rect(fill = NA, color = "black", size = 1), 
  strip.background = ggplot2::element_rect(fill = "white", color = "black", size = 1), 
  strip.text = element_text(color = "black"),
  title = element_text(size = 16),
  axis.text = element_text(size = 16)
  )

# RMSE Comparison
boost_rmse
rf_rmse

# Feature Importance Comparison (Importance of PCA components)

# Higher value for %IncMSE indicates higher importance for prediction accuracy
importance(rf_model_1)  # Importance scores
varImpPlot(rf_model_1)  # Visualization

xgb.importance(model = final_model)
xgb.plot.importance(xgb.importance(model = final_model))

save(pca_result, file = "pca_result.RData")
pca_result$rotation[, 1:5]

# Heatmap of PC loadings
# Convert the rotation matrix into a dataframe
heatmap_data <- as.data.frame(pca_result$rotation[, 1:5])  # Use first 5 PCs
heatmap_data$Feature <- rownames(heatmap_data)

# Reshape data for ggplot
library(reshape2)
heatmap_long <- melt(heatmap_data, id.vars = "Feature", 
                     variable.name = "PC", value.name = "Contribution")

# Create a heatmap
ggplot(data = heatmap_long, aes(x = PC, y = Feature, fill = Contribution)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  geom_text(aes(label = round(Contribution, 2)), size = 3) +
  theme_minimal() +
  labs(
    title = "Feature Contributions to PCs", 
    x = "Principal Components", 
    y = "Features"
  ) +
  theme(
    panel.grid = element_blank(), 
    panel.background = element_rect(fill = "white", color = NA), 
    panel.border = ggplot2::element_rect(fill = NA, color = "black", size = 1), 
    strip.background = ggplot2::element_rect(fill = "white", color = "black", size = 1), 
    strip.text = element_text(color = "black"),
    title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )



View(test_data)
save(test_data, file = "test_data.RData")

test_indices <- as.numeric(rownames(test_data))  # Extract the original row indices from test_data
df_test_pieces <- df_filtered[test_indices, ]
df_test_pieces$rf_group <- ifelse(test_data$rf_model_1_pred >= median(test_data$rf_model_1_pred),
                                  "RF High", "RF Low")
df_test_pieces$b_group <- ifelse(test_data$boost_pred >= median(test_data$boost_pred),
                                 "Boost High", "Boost Low")

# Gather data for plotting
plot_data <- df_test_pieces %>%
  tidyr::gather(feature, val, acousticness, danceability, energy, 
                instrumentalness, liveness, speechiness, tempo, valence)

# Plot faceted by b_group
plot_b <- ggplot(plot_data, aes(x = feature, y = val, colour = feature)) +
  geom_boxplot() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  theme(legend.position = "none") +
  ylab("Value") +
  facet_wrap(~ b_group) +
  ggtitle("Faceted by Boost Groups")

# Plot faceted by rf_group
plot_rf <- ggplot(plot_data, aes(x = feature, y = val, colour = feature)) +
  geom_boxplot() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  theme(legend.position = "none") +
  ylab("Value") +
  facet_wrap(~ rf_group) +
  ggtitle("Faceted by RF Groups")

# Combine the two plots with patchwork
library(patchwork)
plot_b / plot_rf


# Feature significance for each group
features <- c("acousticness", "danceability", "energy", "instrumentalness", 
              "liveness", "speechiness", "tempo", "valence")

# Function to run ANOVA for each feature
anova_results <- lapply(features, function(feature) {
  rf_aov <- aov(df_test_pieces[[feature]] ~ df_test_pieces$rf_group)
  b_aov <- aov(df_test_pieces[[feature]] ~ df_test_pieces$b_group)
  
  data.frame(
    feature = feature,
    rf_p_value = round(summary(rf_aov)[[1]][["Pr(>F)"]][1], 4),
    boost_p_value = round(summary(b_aov)[[1]][["Pr(>F)"]][1], 4)
  )
})

# Combine results into a single data frame
anova_results_df <- do.call(rbind, anova_results)

# Filter significant results
significant_anova <- anova_results_df %>%
  filter(rf_p_value < 0.05 | boost_p_value < 0.05)

significant_anova

# Gather the significant features into long format
df_long <- df_test_pieces %>%
  tidyr::gather(feature, val, acousticness, danceability, energy, instrumentalness, 
                liveness, speechiness, tempo, valence)

# Filter only significant features
signif_rf <- c("valence", "acousticness", "danceability", "energy")
signif_b <- c("acousticness", "danceability", "tempo", "valence")

# RF Groups
rf_plot <- ggplot(df_long %>% filter(feature %in% signif_rf),
                  aes(x = rf_group, y = val, fill = rf_group)) +
  geom_boxplot(notch = TRUE) + 
  facet_wrap(~ feature) +
  theme_minimal() +
  labs(title = "Significant Features (RF Groups)", y = "Value") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(
    panel.grid = element_blank(), 
    panel.background = element_rect(fill = "white", color = NA), 
    panel.border = element_rect(fill = NA, color = "black", size = 1), 
    strip.background = element_rect(fill = "white", color = "black", size = 1), 
    strip.text = element_text(color = "black", size = 14),
    title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )

# Boost Groups
boost_plot <- ggplot(df_long %>% filter(feature %in% signif_b),
                     aes(x = b_group, y = val, fill = b_group)) +
  geom_boxplot(notch = TRUE) + 
  facet_wrap(~ feature) +
  theme_minimal() +
  labs(title = "Significant Features (Boosting Groups)", y = "Value") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(
    panel.grid = element_blank(), 
    panel.background = element_rect(fill = "white", color = NA), 
    panel.border = element_rect(fill = NA, color = "black", size = 1), 
    strip.background = element_rect(fill = "white", color = "black", size = 1), 
    strip.text = element_text(color = "black", size = 14),
    title = element_text(size = 16),
    axis.text = element_text(size = 16),
  )

# Combine plots with patchwork
rf_plot + boost_plot
ggsave("img/feat_box.png", width = 4000, height = 2200, units = "px")

s################################################################################
# Residual Analysis
ggplot(test_data, aes(x = rf_model_1_pred, y = rf_model_1_pred - log_track_listens)) +
  geom_point(alpha = 0.5) + 
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residuals of Random Forest", x = "Predicted", y = "Residuals") +
  theme_minimal()

ggplot(test_data, aes(x = boost_pred, y = boost_pred - log_track_listens)) +
  geom_point(alpha = 0.5) + 
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residuals of XGBoost", x = "Predicted", y = "Residuals") +
  theme_minimal()


################################################################################
# Plot for Random Forest
rf_p <- ggplot(test_data, aes(x = log_track_listens, y = rf_model_1_pred)) +
  geom_point(alpha = 0.5, color = "green") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Random Forest: Predicted vs Actual",
       x = "Actual log_track_listens",
       y = "Predicted log_track_listens") +
  ggpubr::stat_cor(
    aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),
    size = 5
  ) +
  xlim(c(4, 9)) +
  ylim(c(5, 8.5)) +
  theme(
    panel.grid = element_blank(), 
    panel.background = element_rect(fill = "white", color = NA), 
    panel.border = ggplot2::element_rect(fill = NA, color = "black", size = 1), 
    strip.background = ggplot2::element_rect(fill = "white", color = "black", size = 1), 
    strip.text = element_text(color = "black"),
    title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )

# Plot for XGBoost
b_p <- ggplot(test_data, aes(x = log_track_listens, y = boost_pred)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "XGBoost: Predicted vs Actual",
       x = "Actual log_track_listens",
       y = "Predicted log_track_listens") +
  ggpubr::stat_cor(
    aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),
    size = 5
  ) +
  xlim(c(4, 9)) +
  ylim(c(5, 8.5)) +
  theme(
    panel.grid = element_blank(), 
    panel.background = element_rect(fill = "white", color = NA), 
    panel.border = ggplot2::element_rect(fill = NA, color = "black", size = 1), 
    strip.background = ggplot2::element_rect(fill = "white", color = "black", size = 1), 
    strip.text = element_text(color = "black"),
    title = element_text(size = 16),
    axis.text = element_text(size = 16)
  )
library(patchwork)
rf_p / b_p
ggsave("img/regression.png", width = 2500, height = 3000, units = "px")






