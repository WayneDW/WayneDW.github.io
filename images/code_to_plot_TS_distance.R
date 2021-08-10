# Grid of X-axis values
x <- seq(0, 1, 0.001)

# Data
set.seed(1)
sigma = 0.15
alpha = 2
beta = 4
y1 <- 1 / sigma / sqrt(2*pi) * exp(-0.5 * (x-0.6)^2 / sigma^2)
y2 <- x^(alpha-1) * (1-x)^(beta-1) / gamma(alpha) / gamma(beta) * gamma(alpha+beta)
y3 <- rep(0, length(x))

plot(y1)
plot(y2)


# 
# Plot Gaussian density and beta density
plot(x, y1, type = "l", xlab="x-axis label",
     ylim = c(0, 3), xaxt='n', ann=FALSE, yaxt='n', bty="n")
title("Gaussian and Beta densities", line = -2)
lines(x, y3, type = "l", col = 2)

# Fill area between lines
polygon(c(x, rev(x)), c(y3, rev(y1)),
        col = "#6BD7AF", density=40)

polygon(c(x, rev(x)), c(y3, rev(y2)),
        col = "red", density = 40)




# Plot difference between the two densities
plot(x, y1-y2, type = "l",
     ylim = c(-2, 2), ylab = "y", xaxt='n', ann=FALSE, yaxt='n', bty="n")
title("Total variation distance", line = 0)


truncate_up = function(x) {
     return(max(0, x))
}

truncate_down = function(x) {
     return(min(0, x))
}
polygon(c(x, rev(x)), c(y3, rev(sapply(y1-y2, truncate_down))),
        col = "red", density = 40)

polygon(c(x, rev(x)), c(y3, rev(sapply(y1-y2, truncate_up))),
        col = "#6BD7AF", density = 40)