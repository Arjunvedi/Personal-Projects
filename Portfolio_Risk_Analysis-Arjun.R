library(quantmod)
library(ggplot2)
library(reshape2)
library(dplyr)
library(factoextra)

# Define portfolio and market index
symbols <- c("AAPL", "MSFT", "TSLA", "JPM", "XOM", "NVDA", "PG", "JNJ", "DUK", "BA")
market_index <- "^GSPC"

all_symbols <- c(symbols, market_index)
stock_data <- lapply(all_symbols, function(sym) {
  tryCatch({
    getSymbols(sym, src = "yahoo", from = "2018-01-01", to = "2023-01-01", auto.assign = FALSE)
  }, error = function(e) {
    message(paste("Error downloading:", sym))
    return(NULL)
  })
})
names(stock_data) <- all_symbols


prices <- do.call(merge, lapply(stock_data, function(data) {
  if (!is.null(data)) {
    Ad(data)
  } else {
    NULL
  }
}))
colnames(prices) <- c(symbols, "Market")  


log_returns <- diff(log(prices))  
log_returns <- na.omit(log_returns)  

scaled_returns <- scale(log_returns)

# Perform PCA
pca_result <- prcomp(scaled_returns, scale. = TRUE)

summary(pca_result)

pca_loadings <- pca_result$rotation  

pca_components <- pca_result$x      

# Scree plot and biplot for explained variance
fviz_eig(pca_result, 
         addlabels = TRUE,
         ylim = c(0, 70))

fviz_pca_biplot(pca_result,
                label="var")


# Extract top components 
top_components <- as.data.frame(pca_components[, 1:2])
colnames(top_components) <- c("PC1", "PC2")

# Analyze loadings for interpretation
top_loadings <- as.data.frame(pca_loadings[, 1:2])
colnames(top_loadings) <- c("PC1_Loading", "PC2_Loading")
top_loadings$Stock <- rownames(pca_loadings)
print(top_loadings)
print(head(top_components))
