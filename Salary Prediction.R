# ============================================================
# SALARY PREDICTION USING LINEAR REGRESSION
# ============================================================

# ============================================================
# 1. INSTALL AND LOAD REQUIRED PACKAGES
# ============================================================

packages <- c("ggplot2", "dplyr", "Metrics")

new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if(length(new_packages) > 0) {
  install.packages(new_packages)
}

library(ggplot2)
library(dplyr)
library(Metrics)


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Salary dataset: Years of Experience vs Salary
url <- "https://raw.githubusercontent.com/pravinknr/DataScience_R_Codes/master/2.%20Implemetation%20of%20the%20Algorithms%20on%20Datasets/Linear%20Regression/Simple%20Linear%20Regression/Salary%20Data/Salary_Data.csv"

salary_data <- read.csv(url)

# Display first few observations
head(salary_data)

# Dataset structure
str(salary_data)

# Summary statistics
summary(salary_data)


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Check missing values
colSums(is.na(salary_data))

# Remove missing observations if any
salary_data <- na.omit(salary_data)

# Check duplicate observations
sum(duplicated(salary_data))

# Remove duplicates
salary_data <- salary_data[!duplicated(salary_data), ]

# Rename columns for convenience
colnames(salary_data) <- c("YearsExperience", "Salary")

# Check cleaned data
head(salary_data)


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

# Basic statistics
mean(salary_data$YearsExperience)
median(salary_data$YearsExperience)
sd(salary_data$YearsExperience)

mean(salary_data$Salary)
median(salary_data$Salary)
sd(salary_data$Salary)

# Correlation between experience and salary
correlation <- cor(
  salary_data$YearsExperience,
  salary_data$Salary
)

print(paste("Correlation:", round(correlation, 4)))


# ============================================================
# 5. DATA VISUALIZATION
# ============================================================

# Scatter plot
ggplot(salary_data,
       aes(x = YearsExperience, y = Salary)) +
  geom_point(size = 3) +
  labs(
    title = "Salary vs Years of Experience",
    x = "Years of Experience",
    y = "Salary"
  ) +
  theme_minimal()


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

# Set seed for reproducibility
set.seed(123)

# Create random indices
train_indices <- sample(
  1:nrow(salary_data),
  size = 0.80 * nrow(salary_data)
)

# Training data
train_data <- salary_data[train_indices, ]

# Testing data
test_data <- salary_data[-train_indices, ]

# Check dimensions
dim(train_data)
dim(test_data)


# ============================================================
# 7. BUILD LINEAR REGRESSION MODEL
# ============================================================

# Fit simple linear regression model
model <- lm(
  Salary ~ YearsExperience,
  data = train_data
)

# Display model summary
summary(model)


# ============================================================
# 8. MODEL COEFFICIENTS
# ============================================================

# Extract coefficients
coefficients <- coef(model)

intercept <- coefficients[1]
slope <- coefficients[2]

print(paste("Intercept:", round(intercept, 2)))
print(paste("Slope:", round(slope, 2)))

# Regression equation
print(
  paste(
    "Salary =",
    round(intercept, 2),
    "+",
    round(slope, 2),
    "* YearsExperience"
  )
)


# ============================================================
# 9. VISUALIZE REGRESSION LINE
# ============================================================

ggplot(train_data,
       aes(x = YearsExperience, y = Salary)) +
  geom_point(size = 3) +
  geom_smooth(
    method = "lm",
    se = TRUE
  ) +
  labs(
    title = "Linear Regression: Salary Prediction",
    x = "Years of Experience",
    y = "Salary"
  ) +
  theme_minimal()


# ============================================================
# 10. PREDICTIONS ON TEST DATA
# ============================================================

# Predict salary for test data
predicted_salary <- predict(
  model,
  newdata = test_data
)

# Add predictions to test dataset
results <- data.frame(
  ActualSalary = test_data$Salary,
  PredictedSalary = predicted_salary
)

# Display predictions
print(results)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

# Calculate RMSE
rmse_value <- rmse(
  test_data$Salary,
  predicted_salary
)

# Calculate MAE
mae_value <- mae(
  test_data$Salary,
  predicted_salary
)

# Calculate R-squared on test data
ss_res <- sum(
  (test_data$Salary - predicted_salary)^2
)

ss_tot <- sum(
  (test_data$Salary - mean(test_data$Salary))^2
)

r_squared <- 1 - (ss_res / ss_tot)

print(paste("RMSE:", round(rmse_value, 2)))
print(paste("MAE:", round(mae_value, 2)))
print(paste("Test R-squared:", round(r_squared, 4)))


# ============================================================
# 12. TRAINING R-SQUARED
# ============================================================

training_r_squared <- summary(model)$r.squared

print(
  paste(
    "Training R-squared:",
    round(training_r_squared, 4)
  )
)


# ============================================================
# 13. ACTUAL VS PREDICTED SALARY
# ============================================================

results$Experience <- test_data$YearsExperience

ggplot(results,
       aes(x = ActualSalary,
           y = PredictedSalary)) +
  geom_point(size = 3) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Actual vs Predicted Salary",
    x = "Actual Salary",
    y = "Predicted Salary"
  ) +
  theme_minimal()


# ============================================================
# 14. RESIDUAL ANALYSIS
# ============================================================

residuals <- residuals(model)

# Residual vs fitted values
plot(
  fitted(model),
  residuals,
  main = "Residuals vs Fitted Values",
  xlab = "Fitted Values",
  ylab = "Residuals"
)

abline(
  h = 0,
  lty = 2
)


# ============================================================
# 15. NORMAL Q-Q PLOT
# ============================================================

qqnorm(
  residuals,
  main = "Normal Q-Q Plot of Residuals"
)

qqline(
  residuals,
  lty = 2
)


# ============================================================
# 16. RESIDUAL HISTOGRAM
# ============================================================

hist(
  residuals,
  main = "Distribution of Residuals",
  xlab = "Residuals",
  breaks = 10
)


# ============================================================
# 17. MODEL DIAGNOSTIC PLOTS
# ============================================================

par(mfrow = c(2, 2))

plot(model)

par(mfrow = c(1, 1))


# ============================================================
# 18. PREDICT SALARY FOR NEW EXPERIENCE VALUES
# ============================================================

# Create new observations
new_experience <- data.frame(
  YearsExperience = c(1, 3, 5, 7, 10, 12)
)

# Predict salary
new_predictions <- predict(
  model,
  newdata = new_experience,
  interval = "prediction"
)

# Combine experience and predictions
prediction_table <- cbind(
  new_experience,
  new_predictions
)

print(prediction_table)


# ============================================================
# 19. PREDICTION FOR A SINGLE EMPLOYEE
# ============================================================

experience_input <- data.frame(
  YearsExperience = 6
)

salary_prediction <- predict(
  model,
  newdata = experience_input,
  interval = "prediction"
)

print(
  paste(
    "Predicted salary for 6 years of experience:",
    round(salary_prediction[1, "fit"], 2)
  )
)


# ============================================================
# 20. SAVE RESULTS
# ============================================================

write.csv(
  results,
  "salary_prediction_results.csv",
  row.names = FALSE
)

write.csv(
  prediction_table,
  "salary_predictions_new_experience.csv",
  row.names = FALSE
)


# ============================================================
# 21. FINAL MODEL SUMMARY
# ============================================================

cat("\n============================================\n")
cat("       SALARY PREDICTION MODEL SUMMARY\n")
cat("============================================\n")

cat(
  "Regression Equation: Salary = ",
  round(intercept, 2),
  " + ",
  round(slope, 2),
  " × YearsExperience\n",
  sep = ""
)

cat(
  "Training R-squared: ",
  round(training_r_squared, 4),
  "\n",
  sep = ""
)

cat(
  "Test R-squared: ",
  round(r_squared, 4),
  "\n",
  sep = ""
)

cat(
  "RMSE: ",
  round(rmse_value, 2),
  "\n",
  sep = ""
)

cat(
  "MAE: ",
  round(mae_value, 2),
  "\n",
  sep = ""
)

cat("============================================\n")
cat("             PROJECT COMPLETED\n")
cat("============================================\n")