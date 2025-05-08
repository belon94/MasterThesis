## Try to compute the DCC-GARCH(1,1) model with the following steps without taking into account the fat-tailed distribution.
### DCC-GARCH(1,1) model implementation with real data
### 29 ETF and 1 risk-free asset, so 30 variables with 4280 observations

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Optim

# Load data
etf_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/data/etfReturns.csv"
df = CSV.read(etf_rf, DataFrame)

etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", 
             "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", 
             "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", 
             "28278", "28279", "28280", "31372", "31466"]

# Data preprocessing - keeping mean imputation as requested
function preprocess_data(df, etf_names)
    # Handle missing values by replacing with the mean of the column
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)
    
    # Calculate mean returns for each asset
    μ = mean(returns, dims=1)
    
    return etf_returns, rf_returns, returns, vec(μ)
end

# GARCH(1,1) likelihood for parameter estimation
function garch_likelihood(params, returns, μ_i)
    ω, α, β = params
    n = length(returns)
    σ² = zeros(n)
    
    # Initialize with sample variance
    σ²[1] = var(returns .- μ_i)
    loglik = 0.0
    
    for t in 2:n
        # Use squared deviation from mean (r_t - μ)²
        ε²_t_1 = (returns[t-1] - μ_i)^2
        σ²[t] = max(ω + α * ε²_t_1 + β * σ²[t-1], 1e-8)
        loglik += -0.5 * (log(2π) + log(σ²[t]) + (returns[t] - μ_i)^2/σ²[t])
    end
    
    return -loglik
end

# Estimate GARCH parameters
function estimate_garch_params(returns, μ_i)
    # Initial parameters [ω, α, β]
    initial_params = [var(returns .- μ_i)*0.01, 0.1, 0.8]
    
    function obj(params)
        ω, α, β = params
        if ω ≤ 0 || α < 0 || β < 0 || α + β ≥ 1
            return Inf
        end
        return garch_likelihood(params, returns, μ_i)
    end
    
    result = optimize(obj, initial_params, BFGS())
    return Optim.minimizer(result)
end

# Fit univariate GARCH(1,1) and compute standardized residuals
function fit_univariate_garch(returns, μ_i)
    n = length(returns)
    ω, α, β = estimate_garch_params(returns, μ_i)
    
    # Conditional variance
    σ² = zeros(n)
    σ²[1] = var(returns .- μ_i)
    
    # Standardized residuals
    ε = returns .- μ_i
    ν = zeros(n)
    ν[1] = ε[1] / sqrt(σ²[1])
    
    for t in 2:n
        σ²[t] = ω + α * ε[t-1]^2 + β * σ²[t-1]
        ν[t] = ε[t] / sqrt(σ²[t])
    end
    
    # Forecast one step ahead variance
    σ²_next = ω + α * ε[n]^2 + β * σ²[n]
    
    return σ², ν, (ω, α, β), σ²_next
end

# Calculate R_bar (Bollerslev's CCC estimator)
function calculate_R_bar(standardized_residuals)
    T, n = size(standardized_residuals)
    R_bar = zeros(n, n)
    
    # R_bar = (1/T) * sum(ν_t * ν_t')
    for t in 1:T
        R_bar .+= standardized_residuals[t, :] * standardized_residuals[t, :]'
    end
    R_bar ./= T
    
    # Ensure R_bar is a correlation matrix
    D_inv = Diagonal(1.0 ./ sqrt.(diag(R_bar)))
    R_bar = D_inv * R_bar * D_inv
    
    return R_bar
end

# DCC likelihood for parameter estimation
function dcc_likelihood(params, standardized_residuals, R_bar)
    a, b = params
    T, n = size(standardized_residuals)
    
    # Initialize Q_t with unconditional correlation R_bar
    Q_prev = copy(R_bar)
    loglik = 0.0
    
    for t in 2:T
        # Calculate Q_t according to the DCC equation
        ν_prod = standardized_residuals[t-1, :] * standardized_residuals[t-1, :]'
        Q_t = R_bar * (1 - a - b) + a * ν_prod + b * Q_prev
        
        # Ensure Q_t is symmetric
        Q_t = (Q_t + Q_t') / 2
        
        # Compute correlation matrix from Q_t
        Q_diag_inv = Diagonal(1.0 ./ sqrt.(diag(Q_t)))
        R_t = Q_diag_inv * Q_t * Q_diag_inv
        
        # Add to log-likelihood (only correlation part)
        det_R = max(det(R_t), 1e-10)  # Ensure positive determinant
        ν_t = standardized_residuals[t, :]
        loglik += -0.5 * (log(det_R) + ν_t' * inv(R_t) * ν_t - ν_t' * ν_t)
        
        Q_prev = Q_t
    end
    
    return -loglik
end

# Estimate DCC parameters
function estimate_dcc_params(standardized_residuals, R_bar)
    function obj(params)
        a, b = params
        if a < 0 || b < 0 || a + b >= 1
            return Inf
        end
        return dcc_likelihood(params, standardized_residuals, R_bar)
    end
    
    initial_params = [0.01, 0.97]  # Common starting values for financial returns
    result = optimize(obj, initial_params, BFGS())
    return Optim.minimizer(result)
end

# Main DCC-GARCH function
function dcc_garch(returns, μ)
    T, n = size(returns)
    
    # First stage: Fit univariate GARCH models for each series
    σ² = zeros(T, n)
    σ²_next = zeros(n)
    ν = zeros(T, n)
    garch_params = Vector{Tuple{Float64, Float64, Float64}}(undef, n)
    
    for i in 1:n
        σ²[:, i], ν[:, i], garch_params[i], σ²_next[i] = fit_univariate_garch(returns[:, i], μ[i])
    end
    
    # Calculate R_bar based on standardized residuals
    R_bar = calculate_R_bar(ν)
    
    # Second stage: Estimate DCC parameters
    a, b = estimate_dcc_params(ν, R_bar)
    
    # Calculate dynamic correlation matrices
    Q = zeros(T, n, n)
    Q[1, :, :] = R_bar
    R = zeros(T, n, n)
    R[1, :, :] = R_bar
    H = zeros(T, n, n)
    
    # Initialize first conditional covariance matrix
    D_1 = Diagonal(sqrt.(σ²[1, :]))
    H[1, :, :] = D_1 * R[1, :, :] * D_1
    
    for t in 2:T
        # Update Q matrix according to DCC equation
        ν_prod = ν[t-1, :] * ν[t-1, :]'
        Q[t, :, :] = (1 - a - b) * R_bar + a * ν_prod + b * Q[t-1, :, :]
        
        # Ensure symmetry and positive definiteness
        Q[t, :, :] = (Q[t, :, :] + Q[t, :, :]') / 2
        
        # If Q is not positive definite, apply nearest PD correction
        eigvals_Q = eigvals(Q[t, :, :])
        if minimum(eigvals_Q) <= 0
            # Simple correction: add small constant to diagonal
            Q[t, :, :] += Diagonal(1e-4 * ones(n))
        end
        
        # Compute correlation matrix R from Q
        Q_diag_inv = Diagonal(1.0 ./ sqrt.(diag(Q[t, :, :])))
        R[t, :, :] = Q_diag_inv * Q[t, :, :] * Q_diag_inv
        
        # Compute conditional covariance matrix
        D_t = Diagonal(sqrt.(σ²[t, :]))
        H[t, :, :] = D_t * R[t, :, :] * D_t
    end
    
    # Calculate one-step-ahead correlation and covariance matrices
    Q_next = (1 - a - b) * R_bar + a * (ν[T, :] * ν[T, :]') + b * Q[T, :, :]
    Q_next = (Q_next + Q_next') / 2
    
    # Ensure positive definiteness of next-step Q
    eigvals_Q = eigvals(Q_next)
    if minimum(eigvals_Q) <= 0
        Q_next += Diagonal(1e-4 * ones(n))
    end
    
    Q_diag_inv = Diagonal(1.0 ./ sqrt.(diag(Q_next)))
    R_next = Q_diag_inv * Q_next * Q_diag_inv
    
    D_next = Diagonal(sqrt.(σ²_next))
    H_next = D_next * R_next * D_next
    
    return Dict(
        "correlations" => R,
        "next_correlation" => R_next,
        "volatilities" => σ²,
        "next_volatility" => σ²_next,
        "standardized_residuals" => ν,
        "garch_params" => garch_params,
        "dcc_params" => (a, b),
        "conditional_cov" => H,
        "next_cov" => H_next,
        "R_bar" => R_bar
    )
end

# Analysis function
function analyse_results(dcc_results, etf_names, returns, μ)
    n = length(etf_names) + 1  # Including risk-free
    
    println("\nMean returns:")
    for (i, etf) in enumerate(etf_names)
        println("ETF $etf: $(μ[i])")
    end
    println("Risk-free: $(μ[end])")
    
    println("\nGARCH(1,1) Parameters for each asset:")
    for (i, etf) in enumerate([etf_names; "rf"])
        ω, α, β = dcc_results["garch_params"][i]
        persistence = α + β
        uncond_var = ω / (1 - persistence)
        println("Asset $etf: ω = $(round(ω, digits=6)), α = $(round(α, digits=4)), " *
                "β = $(round(β, digits=4)), persistence = $(round(persistence, digits=4)), " *
                "unconditional variance = $(round(uncond_var, digits=6))")
    end
    
    a, b = dcc_results["dcc_params"]
    println("\nDCC Parameters:")
    println("a = $(round(a, digits=4))")
    println("b = $(round(b, digits=4))")
    println("Persistence (a + b) = $(round(a + b, digits=4))")
    
    println("\nUnconditional correlation matrix (R_bar):")
    R_bar = dcc_results["R_bar"]
    println("Average absolute correlation: $(round(mean(abs.(R_bar - Diagonal(diag(R_bar)))), digits=4))")
    
    T = size(returns, 1)
    println("\nFinal day correlations with risk-free rate:")
    for (i, etf) in enumerate(etf_names)
        corr = dcc_results["correlations"][T, i, end]
        println("ETF $etf: $(round(corr, digits=4))")
    end
    
    println("\nForecast for next day:")
    println("Volatilities:")
    for (i, etf) in enumerate([etf_names; "rf"])
        vol = sqrt(dcc_results["next_volatility"][i])
        println("Asset $etf: $(round(vol, digits=6))")
    end
end

# Execute the analysis
etf_returns, rf_returns, returns, μ = preprocess_data(df, etf_names)
dcc_results = dcc_garch(returns, μ)
analyse_results(dcc_results, etf_names, returns, μ)

# Function to examine correlation dynamics for specific pairs
function plot_correlation_dynamics(dcc_results, etf_names, pair_indices)
    T = size(dcc_results["correlations"], 1)
    correlations = dcc_results["correlations"]
    
    println("\nCorrelation Dynamics for Selected Pairs:")
    for (i, j) in pair_indices
        name_i = i <= length(etf_names) ? etf_names[i] : "rf"
        name_j = j <= length(etf_names) ? etf_names[j] : "rf"
        
        # Calculate correlation statistics
        corr_series = [correlations[t, i, j] for t in 1:T]
        avg_corr = mean(corr_series)
        min_corr = minimum(corr_series)
        max_corr = maximum(corr_series)
        
        println("$name_i - $name_j: avg = $(round(avg_corr, digits=4)), " *
                "min = $(round(min_corr, digits=4)), max = $(round(max_corr, digits=4))")
    end
end

#### Should we need to take into account the t-distribution, as the financial data are characterized by fat tails. 
#### We can capture the fail-tails returns and it is better for the 

### DCC-GARCH(1,1) model with Student's t-distribution
### 29 ETF and 1 risk-free asset, so 30 variables with 4280 observations

using CSV, DataFrames, Statistics, LinearAlgebra, Optim, SpecialFunctions

# Load data
etf_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/data/etfReturns.csv"
df = CSV.read(etf_rf, DataFrame)

etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", 
             "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", 
             "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", 
             "28278", "28279", "28280", "31372", "31466"]

# Data preprocessing with mean imputation
function preprocess_data(df, etf_names)
    # Handle missing values by replacing with the mean of the column
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)
    
    # Calculate mean returns for each asset
    μ = mean(returns, dims=1)
    
    return etf_returns, rf_returns, returns, vec(μ)
end

# Student's t log-density function
function logdensity_t(x, df, location, scale)
    # Log-density of the t-distribution
    logconst = lgamma((df + 1)/2) - lgamma(df/2) - 0.5*log(df*π) - log(scale)
    logpdf = logconst - ((df + 1)/2) * log(1 + ((x - location)/scale)^2/df)
    return logpdf
end

# GARCH(1,1) with Student's t likelihood for parameter estimation
function garch_t_likelihood(params, returns, μ_i)
    ω, α, β, df = params
    n = length(returns)
    σ² = zeros(n)
    
    # Initialize with sample variance
    σ²[1] = var(returns .- μ_i)
    loglik = 0.0
    
    for t in 2:n
        # Use squared deviation from mean
        ε²_t_1 = (returns[t-1] - μ_i)^2
        σ²[t] = max(ω + α * ε²_t_1 + β * σ²[t-1], 1e-8)
        
        # Student's t log-likelihood contribution
        σ_t = sqrt(σ²[t])
        loglik += logdensity_t(returns[t] - μ_i, df, 0.0, σ_t)
    end
    
    return -loglik
end

# Estimate GARCH parameters with Student's t errors
function estimate_garch_t_params(returns, μ_i)
    # Initial parameters [ω, α, β, df]
    initial_params = [var(returns .- μ_i)*0.01, 0.1, 0.8, 6.0]
    
    function obj(params)
        ω, α, β, df = params
        if ω ≤ 0 || α < 0 || β < 0 || α + β ≥ 1 || df ≤ 2.1
            return Inf
        end
        return garch_t_likelihood(params, returns, μ_i)
    end
    
    result = optimize(obj, initial_params, BFGS())
    return Optim.minimizer(result)
end

# Fit univariate GARCH(1,1) with Student's t errors
function fit_univariate_garch_t(returns, μ_i)
    n = length(returns)
    ω, α, β, df = estimate_garch_t_params(returns, μ_i)
    
    # Conditional variance
    σ² = zeros(n)
    σ²[1] = var(returns .- μ_i)
    
    # Standardized residuals with adjustment for t distribution
    # For t-distribution, variance = df/(df-2), so we adjust to get unit variance residuals
    scale_factor = sqrt((df - 2)/df)
    
    ε = returns .- μ_i
    ν = zeros(n)
    ν[1] = ε[1] / (sqrt(σ²[1]) * scale_factor)
    
    for t in 2:n
        σ²[t] = ω + α * ε[t-1]^2 + β * σ²[t-1]
        ν[t] = ε[t] / (sqrt(σ²[t]) * scale_factor)
    end
    
    # Forecast one step ahead variance
    σ²_next = ω + α * ε[n]^2 + β * σ²[n]
    
    return σ², ν, (ω, α, β, df), σ²_next
end

# Calculate R_bar (Bollerslev's CCC estimator) with robustness
function calculate_R_bar(standardized_residuals)
    T, n = size(standardized_residuals)
    R_bar = zeros(n, n)
    
    # R_bar = (1/T) * sum(ν_t * ν_t')
    for t in 1:T
        R_bar .+= standardized_residuals[t, :] * standardized_residuals[t, :]'
    end
    R_bar ./= T
    
    # Ensure R_bar is a proper correlation matrix
    D_inv = Diagonal(1.0 ./ sqrt.(diag(R_bar)))
    R_bar = D_inv * R_bar * D_inv
    
    # Ensure positive definiteness
    eigvals_R = eigvals(R_bar)
    if minimum(eigvals_R) <= 0
        R_bar += Diagonal(max(1e-4, -minimum(eigvals_R) + 1e-4) * ones(n))
        # Re-normalize to correlation matrix
        D_inv = Diagonal(1.0 ./ sqrt.(diag(R_bar)))
        R_bar = D_inv * R_bar * D_inv
    end
    
    return R_bar
end

# Multivariate Student's t log-density
function multivariate_t_logdensity(x, df, Σ)
    n = length(x)
    # Log-density of multivariate t
    logconst = lgamma((df + n)/2) - lgamma(df/2) - (n/2)*log(df*π) - 0.5*logdet(Σ)
    quad_form = x' * inv(Σ) * x
    logpdf = logconst - ((df + n)/2) * log(1 + quad_form/df)
    return logpdf
end

# DCC likelihood with Student's t distribution
function dcc_t_likelihood(params, standardized_residuals, R_bar, df_vec)
    a, b, df_dcc = params
    T, n = size(standardized_residuals)
    
    # Use  degrees of freedom from individual GARCH models as starting point
    # but estimate a common df for the multivariate model
    
    # Initialize Q_t with unconditional correlation R_bar
    Q_prev = copy(R_bar)
    loglik = 0.0
    
    for t in 2:T
        # Calculate Q_t according to the DCC equation
        ν_prod = standardized_residuals[t-1, :] * standardized_residuals[t-1, :]'
        Q_t = R_bar * (1 - a - b) + a * ν_prod + b * Q_prev
        
        # Ensure Q_t is symmetric and positive definite
        Q_t = (Q_t + Q_t') / 2
        eigvals_Q = eigvals(Q_t)
        if minimum(eigvals_Q) <= 0
            Q_t += Diagonal(max(1e-4, -minimum(eigvals_Q) + 1e-4) * ones(n))
        end
        
        # Compute correlation matrix from Q_t
        Q_diag_inv = Diagonal(1.0 ./ sqrt.(diag(Q_t)))
        R_t = Q_diag_inv * Q_t * Q_diag_inv
        
        # For multivariate t, we use the correlation matrix for the shape
        # and the degrees of freedom for tail behavior
        ν_t = standardized_residuals[t, :]
        
        # Add to log-likelihood using multivariate t
        loglik += multivariate_t_logdensity(ν_t, df_dcc, R_t)
        
        Q_prev = Q_t
    end
    
    return -loglik
end

# Estimate DCC parameters with Student's t distribution
function estimate_dcc_t_params(standardized_residuals, R_bar, df_vec)
    # Start with average df from univariate models, but let it be estimated
    avg_df = mean(df_vec)
    
    function obj(params)
        a, b, df_dcc = params
        if a < 0 || b < 0 || a + b >= 1 || df_dcc <= 2.1
            return Inf
        end
        return dcc_t_likelihood(params, standardized_residuals, R_bar, df_vec)
    end
    
    initial_params = [0.01, 0.97, avg_df]
    result = optimize(obj, initial_params, BFGS())
    return Optim.minimizer(result)
end

# Main DCC-GARCH with Student's t distribution
function dcc_garch_t(returns, μ)
    T, n = size(returns)
    
    # First stage: Fit univariate GARCH models with Student's t for each series
    σ² = zeros(T, n)
    σ²_next = zeros(n)
    ν = zeros(T, n)
    garch_params = Vector{Tuple{Float64, Float64, Float64, Float64}}(undef, n)
    df_vec = zeros(n)
    
    for i in 1:n
        σ²[:, i], ν[:, i], garch_params[i], σ²_next[i] = fit_univariate_garch_t(returns[:, i], μ[i])
        df_vec[i] = garch_params[i][4]  # Store degrees of freedom
    end
    
    # Calculate R_bar based on standardized residuals
    R_bar = calculate_R_bar(ν)
    
    # Second stage: Estimate DCC parameters with multivariate t
    a, b, df_dcc = estimate_dcc_t_params(ν, R_bar, df_vec)
    
    # Calculate dynamic correlation matrices
    Q = zeros(T, n, n)
    Q[1, :, :] = R_bar
    R = zeros(T, n, n)
    R[1, :, :] = R_bar
    H = zeros(T, n, n)
    
    # Initialize first conditional covariance matrix
    D_1 = Diagonal(sqrt.(σ²[1, :]))
    H[1, :, :] = D_1 * R[1, :, :] * D_1
    
    for t in 2:T
        # Update Q matrix according to DCC equation
        ν_prod = ν[t-1, :] * ν[t-1, :]'
        Q[t, :, :] = (1 - a - b) * R_bar + a * ν_prod + b * Q[t-1, :, :]
        
        # Ensure symmetry and positive definiteness
        Q[t, :, :] = (Q[t, :, :] + Q[t, :, :]') / 2
        eigvals_Q = eigvals(Q[t, :, :])
        if minimum(eigvals_Q) <= 0
            Q[t, :, :] += Diagonal(max(1e-4, -minimum(eigvals_Q) + 1e-4) * ones(n))
        end
        
        # Compute correlation matrix R from Q
        Q_diag_inv = Diagonal(1.0 ./ sqrt.(diag(Q[t, :, :])))
        R[t, :, :] = Q_diag_inv * Q[t, :, :] * Q_diag_inv
        
        # Compute conditional covariance matrix
        D_t = Diagonal(sqrt.(σ²[t, :]))
        H[t, :, :] = D_t * R[t, :, :] * D_t
    end
    
    # Calculate one-step-ahead correlation and covariance matrices
    Q_next = (1 - a - b) * R_bar + a * (ν[T, :] * ν[T, :]') + b * Q[T, :, :]
    Q_next = (Q_next + Q_next') / 2
    
    # Ensure positive definiteness of next-step Q
    eigvals_Q = eigvals(Q_next)
    if minimum(eigvals_Q) <= 0
        Q_next += Diagonal(max(1e-4, -minimum(eigvals_Q) + 1e-4) * ones(n))
    end
    
    Q_diag_inv = Diagonal(1.0 ./ sqrt.(diag(Q_next)))
    R_next = Q_diag_inv * Q_next * Q_diag_inv
    
    D_next = Diagonal(sqrt.(σ²_next))
    H_next = D_next * R_next * D_next
    
    return Dict(
        "correlations" => R,
        "next_correlation" => R_next,
        "volatilities" => σ²,
        "next_volatility" => σ²_next,
        "standardized_residuals" => ν,
        "garch_params" => garch_params,
        "dcc_params" => (a, b, df_dcc),
        "conditional_cov" => H,
        "next_cov" => H_next,
        "R_bar" => R_bar,
        "df_univariate" => df_vec
    )
end

# Enhanced analysis function for t-distribution
function analyse_results_t(dcc_results, etf_names, returns, μ)
    n = length(etf_names) + 1  # Including risk-free
    
    println("\nMean returns:")
    for (i, etf) in enumerate(etf_names)
        println("ETF $etf: $(μ[i])")
    end
    println("Risk-free: $(μ[end])")
    
    println("\nGARCH(1,1) Parameters with Student's t distribution for each asset:")
    for (i, etf) in enumerate([etf_names; "rf"])
        ω, α, β, df = dcc_results["garch_params"][i]
        persistence = α + β
        uncond_var = ω / (1 - persistence)
        println("Asset $etf: ω = $(round(ω, digits=6)), α = $(round(α, digits=4)), " *
                "β = $(round(β, digits=4)), df = $(round(df, digits=2)), " *
                "persistence = $(round(persistence, digits=4)), " *
                "unconditional variance = $(round(uncond_var, digits=6))")
    end
    
    a, b, df_dcc = dcc_results["dcc_params"]
    println("\nDCC Parameters with multivariate t-distribution:")
    println("a = $(round(a, digits=4))")
    println("b = $(round(b, digits=4))")
    println("Persistence (a + b) = $(round(a + b, digits=4))")
    println("Degrees of freedom (multivariate) = $(round(df_dcc, digits=2))")
    
    # Compare univariate and multivariate df
    println("\nUnivariate vs. Multivariate degrees of freedom:")
    println("Average univariate df = $(round(mean(dcc_results["df_univariate"]), digits=2))")
    println("Min univariate df = $(round(minimum(dcc_results["df_univariate"]), digits=2))")
    println("Max univariate df = $(round(maximum(dcc_results["df_univariate"]), digits=2))")
    println("Multivariate df = $(round(df_dcc, digits=2))")
    
    println("\nUnconditional correlation matrix (R_bar):")
    R_bar = dcc_results["R_bar"]
    println("Average absolute correlation: $(round(mean(abs.(R_bar - Diagonal(diag(R_bar)))), digits=4))")
    
    T = size(returns, 1)
    println("\nFinal day correlations with risk-free rate:")
    for (i, etf) in enumerate(etf_names)
        corr = dcc_results["correlations"][T, i, end]
        println("ETF $etf: $(round(corr, digits=4))")
    end
    
    println("\nForecast for next day:")
    println("Volatilities:")
    for (i, etf) in enumerate([etf_names; "rf"])
        vol = sqrt(dcc_results["next_volatility"][i])
        println("Asset $etf: $(round(vol, digits=6))")
    end
    
    # Tail risk analysis
    println("\nTail Risk Analysis:")
    println("Student's t-distribution captures fatter tails than Normal distribution.")
    println("Assets with lowest df (heaviest tails):")
    df_vec = [dcc_results["garch_params"][i][4] for i in 1:n]
    sorted_indices = sortperm(df_vec)
    for i in 1:min(5, n)
        idx = sorted_indices[i]
        asset_name = idx <= length(etf_names) ? etf_names[idx] : "rf"
        println("Asset $asset_name: df = $(round(df_vec[idx], digits=2))")
    end
end

# Function to compare VaR estimates between Normal and t distributions
function calculate_var_comparison(dcc_results, returns, μ, confidence_levels=[0.95, 0.99])
    T, n = size(returns)
    
    # Last day volatilities
    last_volatilities = sqrt.(dcc_results["volatilities"][T, :])
    next_volatilities = sqrt.(dcc_results["next_volatility"])
    
    println("\nValue at Risk (VaR) Comparison:")
    println("Normal vs. Student's t for next day returns:")
    
    for (i, asset) in enumerate([etf_names; "rf"])
        df = dcc_results["garch_params"][i][4]
        vol = next_volatilities[i]
        
        println("\nAsset $asset:")
        println("Forecast volatility: $(round(vol, digits=6))")
        println("Degrees of freedom: $(round(df, digits=2))")
        
        for cl in confidence_levels
            # Normal VaR
            normal_quantile = quantile(Normal(), 1-cl)
            normal_var = μ[i] + normal_quantile * vol
            
            # Student's t VaR
            t_quantile = quantile(TDist(df), 1-cl)
            # Adjust for different variance - t with df degrees of freedom has variance df/(df-2)
            scale_factor = sqrt((df-2)/df)
            t_var = μ[i] + t_quantile * vol / scale_factor
            
            # Relative difference
            rel_diff = (t_var - normal_var) / normal_var * 100
            
            println("$(cl*100)% VaR:")
            println("  Normal: $(round(normal_var, digits=6))")
            println("  Student's t: $(round(t_var, digits=6))")
            println("  Difference: $(round(rel_diff, digits=2))% (t is $(rel_diff < 0 ? "less" : "more") conservative)")
        end
    end
end

# Execute the analysis with Student's t distribution
etf_returns, rf_returns, returns, μ = preprocess_data(df, etf_names)
dcc_results_t = dcc_garch_t(returns, μ)
analyse_results_t(dcc_results_t, etf_names, returns, μ)

# Calculate Value at Risk comparisons
using Distributions
calculate_var_comparison(dcc_results_t, returns, μ)

# Function to examine correlation dynamics for specific pairs
function plot_correlation_dynamics(dcc_results, etf_names, pair_indices)
    T = size(dcc_results["correlations"], 1)
    correlations = dcc_results["correlations"]
    
    println("\nCorrelation Dynamics for Selected Pairs:")
    for (i, j) in pair_indices
        name_i = i <= length(etf_names) ? etf_names[i] : "rf"
        name_j = j <= length(etf_names) ? etf_names[j] : "rf"
        
        # Calculate correlation statistics
        corr_series = [correlations[t, i, j] for t in 1:T]
        avg_corr = mean(corr_series)
        min_corr = minimum(corr_series)
        max_corr = maximum(corr_series)
        std_corr = std(corr_series)
        
        println("$name_i - $name_j: avg = $(round(avg_corr, digits=4)), " *
                "min = $(round(min_corr, digits=4)), max = $(round(max_corr, digits=4)), " *
                "std = $(round(std_corr, digits=4))")
    end
end
