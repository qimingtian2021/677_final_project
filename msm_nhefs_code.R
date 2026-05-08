# MA677 Short Paper - Code Appendix
# Topic: Causal Inference, Inverse Probability Weighting, and Marginal Structural Models
# Data: Real NHEFS data used in Hernan and Robins, Causal Inference: What If
# Source: https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv

# Install packages if needed:
# install.packages(c("tidyverse", "sandwich", "lmtest", "broom"))

library(tidyverse)
library(sandwich)
library(lmtest)
library(broom)

# 1. Load real data --------------------------------------------------------
nhefs <- read_csv("https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv",
                  show_col_types = FALSE)

# 2. Clean analysis data ---------------------------------------------------
# qsmk: 1 if quit smoking between baseline and follow-up, 0 otherwise
# wt82_71: weight change from 1971 to 1982
# Restrict to complete cases for the variables used below.
analysis_dat <- nhefs %>%
  transmute(
    qsmk = qsmk,
    wt82_71 = wt82_71,
    sex = factor(sex),
    race = factor(race),
    age = age,
    education = factor(education),
    smokeintensity = smokeintensity,
    smokeyrs = smokeyrs,
    exercise = factor(exercise),
    active = factor(active),
    wt71 = wt71
  ) %>%
  drop_na()

# 3. Unweighted comparison -------------------------------------------------
unweighted_fit <- lm(wt82_71 ~ qsmk, data = analysis_dat)

# 4. Estimate stabilized inverse probability weights -----------------------
# Numerator: probability of observed treatment using a reduced model.
# Denominator: probability of observed treatment using baseline confounders.
num_model <- glm(qsmk ~ 1, family = binomial(), data = analysis_dat)
den_model <- glm(qsmk ~ sex + race + age + I(age^2) + education +
                   smokeintensity + I(smokeintensity^2) + smokeyrs +
                   I(smokeyrs^2) + exercise + active + wt71 + I(wt71^2),
                 family = binomial(), data = analysis_dat)

p_num <- predict(num_model, type = "response")
p_den <- predict(den_model, type = "response")

analysis_dat <- analysis_dat %>%
  mutate(
    sw = if_else(qsmk == 1, p_num / p_den, (1 - p_num) / (1 - p_den)),
    sw_trunc = pmin(pmax(sw, quantile(sw, 0.01)), quantile(sw, 0.99))
  )

# 5. Weight diagnostics ----------------------------------------------------
weight_summary <- analysis_dat %>%
  summarize(
    n = n(),
    mean_sw = mean(sw),
    sd_sw = sd(sw),
    min_sw = min(sw),
    p01_sw = quantile(sw, 0.01),
    median_sw = median(sw),
    p99_sw = quantile(sw, 0.99),
    max_sw = max(sw)
  )

print(weight_summary)

# 6. Marginal structural model --------------------------------------------
# The coefficient on qsmk estimates the marginal mean difference in weight gain
# comparing quitting smoking versus not quitting, under the identifying assumptions.
msm_fit <- lm(wt82_71 ~ qsmk, data = analysis_dat, weights = sw)
msm_fit_trunc <- lm(wt82_71 ~ qsmk, data = analysis_dat, weights = sw_trunc)

# Robust standard errors are used because weights change the estimating equation.
msm_results <- coeftest(msm_fit, vcov. = vcovHC(msm_fit, type = "HC0"))
msm_trunc_results <- coeftest(msm_fit_trunc, vcov. = vcovHC(msm_fit_trunc, type = "HC0"))

print(summary(unweighted_fit))
print(msm_results)
print(msm_trunc_results)

# 7. Optional covariate balance check --------------------------------------
# This function computes simple standardized mean differences for numeric covariates.
smd_numeric <- function(data, var, weight = NULL) {
  x <- data[[var]]
  a <- data$qsmk
  if (is.null(weight)) {
    m1 <- mean(x[a == 1]); m0 <- mean(x[a == 0])
    s <- sqrt((var(x[a == 1]) + var(x[a == 0])) / 2)
  } else {
    w <- data[[weight]]
    m1 <- weighted.mean(x[a == 1], w[a == 1]); m0 <- weighted.mean(x[a == 0], w[a == 0])
    s <- sd(x)
  }
  (m1 - m0) / s
}

balance_table <- tibble(
  variable = c("age", "smokeintensity", "smokeyrs", "wt71"),
  unweighted_smd = map_dbl(variable, ~ smd_numeric(analysis_dat, .x)),
  weighted_smd = map_dbl(variable, ~ smd_numeric(analysis_dat, .x, "sw"))
)

print(balance_table)
