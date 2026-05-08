###############################################################################
# MA677 Final Project R Code
# Topic: Causal Inference and Marginal Structural Models
# Data: NHEFS complete-case data from the causaldata R package
###############################################################################

# 0. Install and load packages ---------------------------------------------
# If a package is not installed, this code installs it automatically.

required_packages <- c("causaldata", "dplyr", "ggplot2", "sandwich", "lmtest")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(causaldata)
library(dplyr)
library(ggplot2)
library(sandwich)
library(lmtest)

# 1. Load real data ---------------------------------------------------------
# The NHEFS data are real data used in Hernan and Robins' causal inference text.
# The complete-case data include the smoking cessation variable qsmk and
# weight change variable wt82_71.

data("nhefs_complete")
nhefs <- nhefs_complete

# Check the variable names in the real dataset
print(names(nhefs))

# 2. Clean analysis data ----------------------------------------------------
# qsmk: 1 if the participant quit smoking, 0 otherwise
# wt82_71: weight change from 1971 to 1982
# school: years of education
# This keeps the variables used in the treatment model and MSM.

analysis_dat <- nhefs %>%
  transmute(
    qsmk = qsmk,
    wt82_71 = wt82_71,
    sex = factor(sex),
    race = factor(race),
    age = age,
    school = school,
    smokeintensity = smokeintensity,
    smokeyrs = smokeyrs,
    exercise = factor(exercise),
    active = factor(active),
    wt71 = wt71
  ) %>%
  na.omit()

# Basic checks
cat("\nSample size after cleaning:\n")
print(dim(analysis_dat))

cat("\nTreatment variable summary, qsmk:\n")
print(table(analysis_dat$qsmk))

cat("\nOutcome summary, wt82_71:\n")
print(summary(analysis_dat$wt82_71))

# 3. Crude unweighted model -------------------------------------------------
# This simple model compares mean weight change between quitters and non-quitters
# without adjustment.

crude_model <- lm(wt82_71 ~ qsmk, data = analysis_dat)

cat("\nCrude model summary:\n")
print(summary(crude_model))

# 4. Treatment models for stabilized IP weights -----------------------------
# Denominator model: probability of treatment given measured covariates.
# Numerator model: marginal probability of treatment.

prob_denom_model <- glm(
  qsmk ~ sex + race + age + school + smokeintensity + smokeyrs +
    exercise + active + wt71,
  data = analysis_dat,
  family = binomial()
)

prob_num_model <- glm(
  qsmk ~ 1,
  data = analysis_dat,
  family = binomial()
)

# Predicted probabilities
p_denom <- predict(prob_denom_model, type = "response")
p_num <- predict(prob_num_model, type = "response")

# 5. Create stabilized inverse probability weights --------------------------
# For treated observations: P(A = 1) / P(A = 1 | L)
# For untreated observations: P(A = 0) / P(A = 0 | L)

analysis_dat <- analysis_dat %>%
  mutate(
    sw = ifelse(qsmk == 1, p_num / p_denom, (1 - p_num) / (1 - p_denom))
  )

cat("\nSummary of stabilized weights:\n")
print(summary(analysis_dat$sw))

cat("\nSelected quantiles of stabilized weights:\n")
print(quantile(analysis_dat$sw, probs = c(0.01, 0.05, 0.50, 0.95, 0.99)))

# 6. Plot the stabilized weights -------------------------------------------

weight_plot <- ggplot(analysis_dat, aes(x = sw)) +
  geom_histogram(bins = 40) +
  labs(
    title = "Distribution of Stabilized Inverse Probability Weights",
    x = "Stabilized weight",
    y = "Count"
  )

print(weight_plot)

ggsave(
  filename = "nhefs_stabilized_weight_distribution.png",
  plot = weight_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# 7. Optional weight truncation ---------------------------------------------
# Truncation reduces the influence of extremely large weights.
# Here weights are truncated at the 1st and 99th percentiles.

lower_cutoff <- quantile(analysis_dat$sw, 0.01)
upper_cutoff <- quantile(analysis_dat$sw, 0.99)

analysis_dat <- analysis_dat %>%
  mutate(
    sw_trunc = pmin(pmax(sw, lower_cutoff), upper_cutoff)
  )

cat("\nSummary of truncated stabilized weights:\n")
print(summary(analysis_dat$sw_trunc))

# 8. Fit marginal structural model -----------------------------------------
# The MSM estimates the marginal causal contrast between quitting and not quitting,
# using stabilized inverse probability weights.

msm_model <- lm(
  wt82_71 ~ qsmk,
  data = analysis_dat,
  weights = sw
)

cat("\nMSM with stabilized weights:\n")
print(summary(msm_model))

cat("\nMSM with stabilized weights and robust standard errors:\n")
print(coeftest(msm_model, vcov = vcovHC(msm_model, type = "HC0")))

# 9. Fit MSM with truncated weights ----------------------------------------

msm_model_trunc <- lm(
  wt82_71 ~ qsmk,
  data = analysis_dat,
  weights = sw_trunc
)

cat("\nMSM with truncated weights:\n")
print(summary(msm_model_trunc))

cat("\nMSM with truncated weights and robust standard errors:\n")
print(coeftest(msm_model_trunc, vcov = vcovHC(msm_model_trunc, type = "HC0")))

# 10. Compare estimates -----------------------------------------------------

results <- data.frame(
  Model = c(
    "Crude linear model",
    "MSM with stabilized weights",
    "MSM with truncated stabilized weights"
  ),
  Estimate_for_qsmk = c(
    coef(crude_model)["qsmk"],
    coef(msm_model)["qsmk"],
    coef(msm_model_trunc)["qsmk"]
  )
)

cat("\nComparison of estimated effect of quitting smoking on weight change:\n")
print(results)

# 11. Save output -----------------------------------------------------------

write.csv(results, "msm_nhefs_results.csv", row.names = FALSE)

cat("\nFiles saved:\n")
cat("msm_nhefs_results.csv\n")
cat("nhefs_stabilized_weight_distribution.png\n")

###############################################################################
# End of file
###############################################################################

