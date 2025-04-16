## to compute the MGARCH model, I decide to compute the DCC-GARCH model, which it is an extension for the CCC-MGARCH model
## because it has more flexibility with time-varying correlations and these correlations evolves over time and it is a function of past returns
## it also captures the market dynamics better than other MGARCH models . Compared to the CCC MGARCH model, 
## it does not require the assumption of constant correlations.
### to compute the DCC-GARCH model, The package ARCHModels.jl will be used 



using BayesFlux, Flux
using Random, Distributions
using StatsPlots, Optim
using ARCHModels, LinearAlgebra, DataFrames, CSV, Plots, Statistics
using MCMCChains, Bijectors

Random.seed!(6150533)


### A simulation of the DCC-GARCH model before testing it with the real data

# Sample size and parameters
n = 500
vol_stocks = (20^2)/252  # 20% annual volatility for stocks
vol_rf = (2^2)/252      # 2% annual volatility for risk-free rate
α = 0.5
β = 0.0

# Modified GARCH simulation function to handle different volatilities
function multivariate_garchNDraws(n::Int, N::Int, α::Float64, β::Float64, vol_stocks::Float64, vol_rf::Float64)
    # Initialize arrays
    ϵ = randn(Float64, n, N)
    L = similar(ϵ)
    σ_2 = zeros(n + 1, N)
    
    # Separate parameters for stocks and risk-free
    for i in 1:N
        # Different initialization for stocks vs risk-free
        if i < N  # Stocks
            ω = vol_stocks * (1 - α - β)
            L[1, i] = sqrt(vol_stocks) * ϵ[1, i]
            σ_2[1, i] = vol_stocks
        else      # Risk-free asset
            ω = vol_rf * (1 - α - β)
            L[1, i] = sqrt(vol_rf) * ϵ[1, i]
            σ_2[1, i] = vol_rf
        end
        
        # Generate series
        for j in 2:n
            if i < N  # Stocks
                σ_2[j, i] = vol_stocks * (1 - α - β) + α * L[j-1, i]^2 + β * σ_2[j-1, i]
            else      # Risk-free
                σ_2[j, i] = vol_rf * (1 - α - β) + α * L[j-1, i]^2 + β * σ_2[j-1, i]
            end
            L[j, i] = sqrt(σ_2[j, i]) * ϵ[j, i]
        end
        
        # One period ahead volatility
        if i < N
            σ_2[n+1, i] = vol_stocks * (1 - α - β) + α * L[n, i]^2 + β * σ_2[n, i]
        else
            σ_2[n+1, i] = vol_rf * (1 - α - β) + α * L[n, i]^2 + β * σ_2[n, i]
        end
    end
    
    return Dict("L" => L, "sigma_squared" => σ_2, "nextSigma" => σ_2[n+1, :])
end

# DCC parameters
function dcc_garch(returns, n_assets)
    # Parameters for DCC
    a = 0.01  # DCC alpha
    b = 0.97  # DCC beta
    
    # Standardize returns using GARCH volatilities
    volatilities = sqrt.(simulated_DCC_GARCH["sigma_squared"][1:end-1, :])
    std_returns = returns ./ volatilities
    
    # Initialize matrices
    Q_bar = cor(returns)
    Q_t = zeros(n, n_assets, n_assets)
    R_t = zeros(n, n_assets, n_assets)
    Q_t[1, :, :] = Q_bar
    
    # DCC recursion
    for t in 2:n
        # Q matrix update
        Q_t[t, :, :] = (1 - a - b) * Q_bar + 
                       a * (std_returns[t-1, :] * std_returns[t-1, :]') + 
                       b * Q_t[t-1, :, :]
        
        # Convert to correlation matrix
        Q_diag = sqrt.(diag(Q_t[t, :, :]))
        R_t[t, :, :] = Q_t[t, :, :] ./ (Q_diag * Q_diag')
    end
    
    return R_t
end

# Simulate for 29 stocks + 1 risk-free 
N = 30
simulated_DCC_GARCH = multivariate_garchNDraws(n, N, α, β, vol_stocks, vol_rf)

# Get returns and compute DCC correlations
returns = simulated_DCC_GARCH["L"]
dcc_correlations = dcc_garch(returns, N)

# Add mean returns for risk-free asset
returns[:, end] .+= 0.02/252  # Adding 2% annual risk-free rate

# Visualization
# Plot correlation between first stock and risk-free rate
plot(dcc_correlations[:, 1, N], 
     label="Stock 1 - Risk-free correlation",
     xlabel="Time",
     ylabel="Correlation",
     title="DCC-GARCH Dynamic Correlation")

# Plot volatilities
plot(sqrt.(simulated_DCC_GARCH["sigma_squared"][1:end-1, 1]), 
     label="Stock 1 volatility",
     xlabel="Time",
     ylabel="Volatility")
plot!(sqrt.(simulated_DCC_GARCH["sigma_squared"][1:end-1, N]), 
      label="Risk-free volatility")

### now I will implement the DCC-GARCH(1,1) model with the real data
### 29 ETF and 1 risk-free asset, so 30 variables with 4280 observations

# Load data
eft_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/data/etfReturns.csv"
df = CSV.read(eft_rf, DataFrame)

etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", 
             "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", 
             "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", 
             "28278", "28279", "28280", "31372", "31466"]



# Data preprocessing
function preprocess_data(df, etf_names)
    # Handle missing values , the missing value is replaced by the mean of the column. "Ask the professor if it is the best way to handle missing values"
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)
    
    return etf_returns, rf_returns, returns
end

# Process data
etf_returns, rf_returns, returns = preprocess_data(df, etf_names)

# GARCH(1,1) likelihood for parameter estimation
function garch_likelihood(params, returns)
    ω, α, β = params
    n = length(returns)
    σ² = zeros(n)
    σ²[1] = var(returns)
    loglik = 0.0
    
    for t in 2:n
        σ²[t] = max(ω + α * returns[t-1]^2 + β * σ²[t-1], 1e-6)
        loglik += -0.5 * (log(2π) + log(σ²[t]) + returns[t]^2/σ²[t])
    end
    
    return -loglik
end

# Estimate GARCH parameters
function estimate_garch_params(returns)
    initial_params = [var(returns)*0.01, 0.1, 0.8]
    
    function obj(params)
        ω, α, β = params
        if ω ≤ 0 || α < 0 || β < 0 || α + β ≥ 1
            return Inf
        end
        return garch_likelihood(params, returns)
    end
    
    result = optimize(obj, initial_params, BFGS())
    return Optim.minimizer(result)
end

# Fit univariate GARCH(1,1)
function fit_univariate_garch(returns)
    n = length(returns)
    ω, α, β = estimate_garch_params(returns)
    σ² = zeros(n + 1)
    σ²[1] = var(returns)
    
    for t in 2:n
        σ²[t] = ω + α * returns[t-1]^2 + β * σ²[t-1]
    end
    
    σ²[n+1] = ω + α * returns[n]^2 + β * σ²[n]
    return σ², (ω, α, β)
end

# DCC likelihood for parameter estimation
function dcc_likelihood(params, std_returns, Q_bar)
    a, b = params
    T, N = size(std_returns)
    Qt = similar(Q_bar)
    Qt .= Q_bar
    loglik = 0.0
    
    for t in 2:T
        Qt = (1 - a - b) * Q_bar + 
             a * (std_returns[t-1,:] * std_returns[t-1,:]') + 
             b * Qt
        
        Qt_diag = Diagonal(sqrt.(diag(Qt)))
        Rt = inv(Qt_diag) * Qt * inv(Qt_diag)
        
        loglik += -0.5 * (log(det(Rt)) + std_returns[t,:]' * inv(Rt) * std_returns[t,:])
    end
    
    return -loglik
end

# Estimate DCC parameters
function estimate_dcc_params(std_returns)
    Q_bar = cor(std_returns)
    
    function obj(params)
        a, b = params
        if a < 0 || b < 0 || (a + b) ≥ 1
            return Inf
        end
        return dcc_likelihood(params, std_returns, Q_bar)
    end
    
    initial_params = [0.01, 0.97]
    result = optimize(obj, initial_params, BFGS())
    return Optim.minimizer(result)
end

# Main DCC-GARCH function
function dcc_garch(returns)
    n, N = size(returns)
    
    # First stage: Fit univariate GARCH models
    volatilities = zeros(n + 1, N)
    std_returns = zeros(n, N)
    garch_params = Vector{Tuple{Float64, Float64, Float64}}(undef, N)
    
    for i in 1:N
        volatilities[:, i], garch_params[i] = fit_univariate_garch(returns[:, i])
        std_returns[:, i] = returns[:, i] ./ sqrt.(volatilities[1:end-1, i])
    end
    
    # Second stage: DCC estimation
    a, b = estimate_dcc_params(std_returns)
    
    Q_bar = cor(returns)
    Q_t = zeros(n, N, N)
    R_t = zeros(n, N, N)
    H_t = zeros(n, N, N)
    Q_t[1, :, :] = Q_bar
    
    for t in 2:n
        Q_t[t, :, :] = (1 - a - b) * Q_bar + 
                       a * (std_returns[t-1, :] * std_returns[t-1, :]') + 
                       b * Q_t[t-1, :, :]
        
        # Ensure positive definiteness
        Q_t[t, :, :] = (Q_t[t, :, :] + Q_t[t, :, :]') / 2
        
        # Compute correlation matrix
        Q_diag = Diagonal(sqrt.(diag(Q_t[t, :, :])))
        R_t[t, :, :] = inv(Q_diag) * Q_t[t, :, :] * inv(Q_diag)
        
        # Compute conditional covariance matrix
        D_t = Diagonal(sqrt.(volatilities[t, :]))
        H_t[t, :, :] = D_t * R_t[t, :, :] * D_t
    end
    
    return Dict(
        "correlations" => R_t,
        "volatilities" => volatilities,
        "std_returns" => std_returns,
        "garch_params" => garch_params,
        "dcc_params" => (a, b),
        "conditional_cov" => H_t
    )
end

# Analysis function
function analyse_results(dcc_results, etf_names, returns)
    n, N = size(returns)
    
    println("\nGARCH(1,1) Parameters for each asset:")
    for (i, etf) in enumerate(etf_names)
        ω, α, β = dcc_results["garch_params"][i]
        println("ETF $etf: ω = $ω, α = $α, β = $β")
    end
    
    a, b = dcc_results["dcc_params"]
    println("\nDCC Parameters:")
    println("a = $a")
    println("b = $b")
    
    println("\nAverage correlations with risk-free rate:")
    for (i, etf) in enumerate(etf_names)
        avg_corr = mean(dcc_results["correlations"][:, i, end])
        println("ETF $etf: $avg_corr")
    end
end



# Fit model
dcc_results = dcc_garch(returns)

analyse_results(dcc_results, etf_names, returns)

#### Now also I will compute the DCC-GARCH model with t-distribution
# GARCH(1,1) with t-distribution likelihood for parameter estimation
function t_garch_likelihood(params, returns)
    ω, α, β, df = params  # Added degrees of freedom parameter
    n = length(returns)
    σ² = zeros(n)
    σ²[1] = var(returns)
    loglik = 0.0
    
    # Constants for t-distribution
    logconst = lgamma((df + 1)/2) - lgamma(df/2) - 0.5*log(π*df)
    
    for t in 2:n
        σ²[t] = max(ω + α * returns[t-1]^2 + β * σ²[t-1], 1e-6)
        # t-distribution log-likelihood
        loglik += logconst - 0.5*log(σ²[t]) - 
                 ((df + 1)/2) * log(1 + (returns[t]^2)/(σ²[t]*df))
    end
    
    return -loglik
end

# Estimate GARCH parameters with t-distribution
function estimate_t_garch_params(returns)
    initial_params = [var(returns)*0.01, 0.1, 0.8, 5.0]  # Initial df = 5
    
    function obj(params)
        ω, α, β, df = params
        if ω ≤ 0 || α < 0 || β < 0 || α + β ≥ 1 || df <= 2  # df > 2 for finite variance
            return Inf
        end
        return t_garch_likelihood(params, returns)
    end
    
    result = optimize(obj, initial_params, BFGS())
    return Optim.minimizer(result)
end

# Fit univariate GARCH(1,1) with t-distribution
function fit_univariate_t_garch(returns)
    n = length(returns)
    ω, α, β, df = estimate_t_garch_params(returns)
    σ² = zeros(n + 1)
    σ²[1] = var(returns)
    
    for t in 2:n
        σ²[t] = ω + α * returns[t-1]^2 + β * σ²[t-1]
    end
    
    σ²[n+1] = ω + α * returns[n]^2 + β * σ²[n]
    return σ², (ω, α, β, df)
end

# Modified DCC likelihood with t-distribution
function t_dcc_likelihood(params, std_returns, Q_bar, df_vector)
    a, b = params
    T, N = size(std_returns)
    Qt = similar(Q_bar)
    Qt .= Q_bar
    loglik = 0.0
    
    for t in 2:T
        Qt = (1 - a - b) * Q_bar + 
             a * (std_returns[t-1,:] * std_returns[t-1,:]') + 
             b * Qt
        
        # Ensure positive definiteness
        Qt = (Qt + Qt') / 2
        
        Qt_diag = Diagonal(sqrt.(diag(Qt)))
        Rt = inv(Qt_diag) * Qt * inv(Qt_diag)
        
        # Multivariate t-distribution log-likelihood
        # Using average df for simplicity - could be more complex in a full implementation
        avg_df = mean(df_vector)
        v = std_returns[t,:]' * inv(Rt) * std_returns[t,:]
        
        loglik += lgamma((avg_df + N)/2) - lgamma(avg_df/2) - (N/2)*log(π*avg_df) -
                 0.5*log(det(Rt)) - ((avg_df + N)/2)*log(1 + v/avg_df)
    end
    
    return -loglik
end

# Main DCC-GARCH function with t-distribution
function t_dcc_garch(returns)
    n, N = size(returns)
    
    # First stage: Fit univariate t-GARCH models
    volatilities = zeros(n + 1, N)
    std_returns = zeros(n, N)
    t_garch_params = Vector{Tuple{Float64, Float64, Float64, Float64}}(undef, N)
    
    for i in 1:N
        volatilities[:, i], t_garch_params[i] = fit_univariate_t_garch(returns[:, i])
        std_returns[:, i] = returns[:, i] ./ sqrt.(volatilities[1:end-1, i])
    end
    
    # Extract degrees of freedom from each univariate model
    df_vector = [params[4] for params in t_garch_params]
    
    # Second stage: DCC estimation with t-distribution
    Q_bar = cor(returns)
    
    function obj(params)
        a, b = params
        if a < 0 || b < 0 || (a + b) ≥ 1
            return Inf
        end
        return t_dcc_likelihood(params, std_returns, Q_bar, df_vector)
    end
    
    initial_params = [0.01, 0.97]
    result = optimize(obj, initial_params, BFGS())
    a, b = Optim.minimizer(result)
    
    # Compute time-varying matrices
    Q_t = zeros(n, N, N)
    R_t = zeros(n, N, N)
    H_t = zeros(n, N, N)
    Q_t[1, :, :] = Q_bar
    
    for t in 2:n
        Q_t[t, :, :] = (1 - a - b) * Q_bar + 
                       a * (std_returns[t-1, :] * std_returns[t-1, :]') + 
                       b * Q_t[t-1, :, :]
        
        # Ensure positive definiteness
        Q_t[t, :, :] = (Q_t[t, :, :] + Q_t[t, :, :]') / 2
        
        # Compute correlation matrix
        Q_diag = Diagonal(sqrt.(diag(Q_t[t, :, :])))
        R_t[t, :, :] = inv(Q_diag) * Q_t[t, :, :] * inv(Q_diag)
        
        # Compute conditional covariance matrix
        D_t = Diagonal(sqrt.(volatilities[t, :]))
        H_t[t, :, :] = D_t * R_t[t, :, :] * D_t
    end
    
    return Dict(
        "correlations" => R_t,
        "volatilities" => volatilities,
        "std_returns" => std_returns,
        "garch_params" => t_garch_params,
        "dcc_params" => (a, b),
        "conditional_cov" => H_t,
        "degrees_of_freedom" => df_vector
    )
end

# Modified analysis function to include degrees of freedom
function analyse_t_results(t_dcc_results, etf_names, returns)
    n, N = size(returns)
    
    println("\nGARCH(1,1) with t-distribution Parameters for each asset:")
    for (i, etf) in enumerate(etf_names)
        ω, α, β, df = t_dcc_results["garch_params"][i]
        println("ETF $etf: ω = $ω, α = $α, β = $β, df = $df")
    end
    
    a, b = t_dcc_results["dcc_params"]
    println("\nDCC Parameters:")
    println("a = $a")
    println("b = $b")
    
    println("\nAverage correlations with risk-free rate:")
    for (i, etf) in enumerate(etf_names)
        avg_corr = mean(t_dcc_results["correlations"][:, i, end])
        println("ETF $etf: $avg_corr")
    end
    
    println("\nAverage degrees of freedom: ", mean(t_dcc_results["degrees_of_freedom"]))
end

# Fit t-distribution model
t_dcc_results = t_dcc_garch(returns)
analyse_t_results(t_dcc_results, etf_names, returns)