library(AER)
library(stargazer)
library(car)
library(boot)
library(margins)
library(dynlm)

setwd("~/Desktop/Algothon")

# ===============================
# ARDL(1,1) ON LOG RETURNS IN R
# ===============================

# 1. Load and transform data
prices <- read.csv("prices.csv", header = TRUE)
prices <- as.matrix(prices)
prices <- t(prices)  # Ensure shape: 50 instruments x time

# 2. Compute log returns
log_prices <- log(prices)
returns <- t(apply(log_prices, 1, diff))  # shape: 50 x (T - 1)

# 3. Install and load dynlm
if (!require("dynlm")) install.packages("dynlm", dependencies = TRUE)
library(dynlm)

# 4. Initialize storage
n_inst <- nrow(returns)
coef_mat <- matrix(NA, n_inst, n_inst)
pval_mat <- matrix(NA, n_inst, n_inst)

# 5. Loop over target and regressor combinations
for (target in 1:n_inst) {
  for (reg in 1:n_inst) {
    if (target == reg) next
    
    y <- ts(returns[target, ])
    x <- ts(returns[reg, ])
    
    model <- dynlm(y ~ L(y, 1) + L(x, 1))
    sm <- summary(model)
    
    coef_mat[reg, target] <- coef(sm)["L(x, 1)", "Estimate"]
    pval_mat[reg, target] <- coef(sm)["L(x, 1)", "Pr(>|t|)"]
  }
}

# 6. Output to CSV
write.csv(coef_mat, "ardl_coefficients_log_returns_r.csv", row.names = FALSE)
write.csv(pval_mat, "ardl_pvalues_log_returns_r.csv", row.names = FALSE)

cat("ARDL results (log returns) saved to ardl_coefficients_log_returns_r.csv and ardl_pvalues_log_returns_r.csv\n")