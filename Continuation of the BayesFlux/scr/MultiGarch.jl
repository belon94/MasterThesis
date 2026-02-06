## to compute the MGARCH model, I decide to compute the DCC-GARCH model, which it is an extension for the CCC-MGARCH model
## because it has more flexibility with time-varying correlations and these correlations evolves over time and it is a function of past returns
## it also captures the market dynamics better than other MGARCH models . Compared to the CCC MGARCH model, 
## it does not require the assumption of constant correlations.
### to compute the DCC-GARCH model, The package ARCHModels.jl will be used 


using BayesFlux, Flux
using Random, Distributions
using StatsPlots, Optim
using ARCHModels
using LinearAlgebra, DataFrames, CSV, Plots, Statistics
using MCMCChains, Bijectors
using SpecialFunctions

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
eft_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"
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

#############################

###############################################################################
#  Traditional DCC-GARCH Model - Multiple Assets
###############################################################################
using Random, Distributions, LinearAlgebra, Plots
using CSV, DataFrames, Statistics, Optim
using MCMCChains  # For diagnostics only

Random.seed!(1212)

# ───────────────────────── 1. Helper Functions ──────────────────────────────
sigmoid(x) = 1/(1+exp(-x))
transform_garch_params(params) = abs.(params) .+ 1e-6  # Ensure positivity
transform_dcc_params(a, b) = let a_=sigmoid(a); b_=sigmoid(b)*(1-a_); (a_, b_) end
nearest_pd(A) = (A + A')/2 + 1e-6I

# ───────────────────────── 2. Traditional DCC-GARCH Model ──────────────────────────
struct DCCGARCHModel
    N::Int          # Number of assets
    data::Matrix{Float64}  # Returns data (T x N)
    T::Int          # Number of observations
end

function DCCGARCHModel(data::Matrix{Float64})
    T, N = size(data)
    return DCCGARCHModel(N, data, T)
end

# GARCH(1,1) volatility for single asset
function garch11_volatility(returns::Vector{Float64}, ω::Float64, α::Float64, β::Float64)
    T = length(returns)
    σ² = zeros(T)
    σ²[1] = var(returns)  # Initial variance
    
    for t in 2:T
        σ²[t] = ω + α * returns[t-1]^2 + β * σ²[t-1]
    end
    
    return sqrt.(σ²)
end

# DCC log-likelihood function
function dcc_loglikelihood(params::Vector{Float64}, model::DCCGARCHModel)
    N, data, T = model.N, model.data, model.T
    
    # Parameter extraction
    param_idx = 1
    
    # GARCH parameters for each asset (ω, α, β for each)
    garch_params = reshape(params[param_idx:param_idx+3*N-1], 3, N)
    param_idx += 3*N
    
    # DCC parameters (a, b)
    dcc_a_raw, dcc_b_raw = params[param_idx], params[param_idx+1]
    dcc_a, dcc_b = transform_dcc_params(dcc_a_raw, dcc_b_raw)
    
    # Ensure GARCH parameters are positive and stationary
    ω = transform_garch_params(garch_params[1, :])
    α = transform_garch_params(garch_params[2, :])
    β = transform_garch_params(garch_params[3, :])
    
    # Check stationarity condition
    if any(α .+ β .>= 0.99)
        return -Inf
    end
    
    # Step 1: Estimate individual GARCH volatilities
    σ = zeros(T, N)
    for i in 1:N
        σ[:, i] = garch11_volatility(data[:, i], ω[i], α[i], β[i])
    end
    
    # Step 2: Compute standardized residuals
    z = data ./ σ
    
    # Step 3: DCC estimation
    # Unconditional correlation matrix
    Q̄ = cor(z)
    Q̄ = nearest_pd(Q̄)  # Ensure positive definite
    
    # Initialize Q
    Q = copy(Q̄)
    
    loglik = 0.0
    
    for t in 1:T
        # Current correlation matrix
        Q_diag_inv_sqrt = diagm(1 ./ sqrt.(max.(diag(Q), 1e-10)))
        R = Q_diag_inv_sqrt * Q * Q_diag_inv_sqrt
        R = nearest_pd(R)
        
        # Current covariance matrix
        D = diagm(σ[t, :])
        H = D * R * D
        H = nearest_pd(H)
        
        try
            # Log-likelihood contribution
            L = cholesky(H).L
            diff = data[t, :]
            quad = sum(abs2, L \ diff)
            loglik += -0.5 * (N * log(2π) + 2 * sum(log, diag(L)) + quad)
        catch
            return -Inf
        end
        
        # Update Q for next period
        if t < T
            z_t = z[t, :]
            Q = (1 - dcc_a - dcc_b) * Q̄ + dcc_a * (z_t * z_t') + dcc_b * Q
        end
    end
    
    return loglik
end

# ───────────────────────── 3. Estimation Function ──────────────────────────
function estimate_dcc_garch(model::DCCGARCHModel; maxiter=1000)
    N = model.N
    
    # Initial parameter values
    # GARCH parameters: ω, α, β for each asset
    initial_garch = vcat([0.01, 0.05, 0.9] for _ in 1:N...)
    
    # DCC parameters: a, b (in raw form for transformation)
    initial_dcc = [0.01, 0.95]
    
    initial_params = vcat(initial_garch, initial_dcc)
    
    println("Starting DCC-GARCH estimation...")
    println("Number of parameters: $(length(initial_params))")
    println("GARCH parameters per asset: 3 (ω, α, β)")
    println("DCC parameters: 2 (a, b)")
    
    # Objective function (negative log-likelihood)
    objective(params) = -dcc_loglikelihood(params, model)
    
    # Optimization
    result = optimize(objective, initial_params, 
                     LBFGS(), 
                     Optim.Options(iterations=maxiter, show_trace=true))
    
    if Optim.converged(result)
        println("✓ Estimation converged successfully")
    else
        println("⚠ Estimation did not converge")
    end
    
    return result
end

# ───────────────────────── 4. Results Analysis ──────────────────────────────
function analyze_results(result, model::DCCGARCHModel)
    N = model.N
    params = Optim.minimizer(result)
    
    # Extract parameters
    garch_params = reshape(params[1:3*N], 3, N)
    dcc_params_raw = params[3*N+1:end]
    
    # Transform parameters
    ω = transform_garch_params(garch_params[1, :])
    α = transform_garch_params(garch_params[2, :])
    β = transform_garch_params(garch_params[3, :])
    dcc_a, dcc_b = transform_dcc_params(dcc_params_raw...)
    
    println("\n" * "="^60)
    println("DCC-GARCH ESTIMATION RESULTS")
    println("="^60)
    println("Log-likelihood: $(round(-Optim.minimum(result), digits=4))")
    println("Number of observations: $(model.T)")
    println("Number of assets: $(model.N)")
    
    println("\nGARCH(1,1) Parameters:")
    for i in 1:N
        println("Asset $i:")
        println("  ω = $(round(ω[i], digits=6))")
        println("  α = $(round(α[i], digits=6))")  
        println("  β = $(round(β[i], digits=6))")
        println("  α + β = $(round(α[i] + β[i], digits=6))")
    end
    
    println("\nDCC Parameters:")
    println("  a = $(round(dcc_a, digits=6))")
    println("  b = $(round(dcc_b, digits=6))")
    println("  a + b = $(round(dcc_a + dcc_b, digits=6))")
    
    return Dict(
        "omega" => ω,
        "alpha" => α, 
        "beta" => β,
        "dcc_a" => dcc_a,
        "dcc_b" => dcc_b,
        "loglik" => -Optim.minimum(result)
    )
end

# ───────────────────────── 5. Forecasting Function ──────────────────────────
function forecast_volatility_correlation(model::DCCGARCHModel, params_dict, h::Int=1)
    N, data, T = model.N, model.data, model.T
    ω, α, β = params_dict["omega"], params_dict["alpha"], params_dict["beta"]
    dcc_a, dcc_b = params_dict["dcc_a"], params_dict["dcc_b"]
    
    # Compute final period volatilities and correlations
    σ = zeros(T, N)
    for i in 1:N
        σ[:, i] = garch11_volatility(data[:, i], ω[i], α[i], β[i])
    end
    
    z = data ./ σ
    Q̄ = cor(z)
    Q̄ = nearest_pd(Q̄)
    
    # Get final Q matrix
    Q = copy(Q̄)
    for t in 1:T-1
        z_t = z[t, :]
        Q = (1 - dcc_a - dcc_b) * Q̄ + dcc_a * (z_t * z_t') + dcc_b * Q
    end
    
    # Forecast volatilities (h-step ahead)
    σ_forecast = zeros(N)
    for i in 1:N
        σ²_T = σ[T, i]^2
        unconditional_var = ω[i] / (1 - α[i] - β[i])
        σ²_forecast = unconditional_var + (α[i] + β[i])^h * (σ²_T - unconditional_var)
        σ_forecast[i] = sqrt(σ²_forecast)
    end
    
    # Forecast correlations (mean revert to unconditional)
    Q_forecast = Q̄ + (dcc_a + dcc_b)^h * (Q - Q̄)
    Q_diag_inv_sqrt = diagm(1 ./ sqrt.(diag(Q_forecast)))
    R_forecast = Q_diag_inv_sqrt * Q_forecast * Q_diag_inv_sqrt
    
    return σ_forecast, R_forecast
end

# ───────────────────────── 6. Data Loading and Main Analysis ────────────────────────

# Load and preprocess data function
function load_and_preprocess_data(file_path::String, selected_assets::Vector{String})
    df = CSV.read(file_path, DataFrame)
    
    # Handle missing values
    for col in selected_assets
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    returns = Matrix{Float64}(df[!, selected_assets])
    μ = mean(returns, dims=1)
    
    return returns, vec(μ)
end

# Main analysis function
function run_dcc_garch_analysis(n_assets::Int)
    println("\n" * "="^60)
    println("TRADITIONAL DCC-GARCH ANALYSIS - $n_assets ASSETS")
    println("="^60)
    
    # Load data
    etf_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"  # Adjust path as needed
    etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", 
                 "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", 
                 "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", 
                 "28278", "28279", "28280", "31372", "31466"]
    
    selected_assets = etf_names[1:n_assets]
    
    try
        returns, μ_returns = load_and_preprocess_data(etf_rf, selected_assets)
        
        println("Selected assets: $selected_assets")
        println("Data dimensions: $(size(returns, 1)) observations, $n_assets assets")
        println("Average returns: $(round.(μ_returns, digits=6))")
        
        # Create model
        model = DCCGARCHModel(returns)
        
        # Estimate model
        result = estimate_dcc_garch(model)
        
        # Analyze results
        params_dict = analyze_results(result, model)
        
        # Generate forecasts
        println("\nGenerating 1-step ahead forecasts...")
        σ_forecast, R_forecast = forecast_volatility_correlation(model, params_dict, 1)
        
        println("\nForecasted Volatilities:")
        for i in 1:n_assets
            println("Asset $i: $(round(σ_forecast[i], digits=6))")
        end
        
        println("\nForecasted Correlation Matrix:")
        display(round.(R_forecast, digits=4))
        
        return model, result, params_dict
        
    catch e
        println("Error loading data: $e")
        println("Please ensure the data file path is correct")
        return nothing, nothing, nothing
    end
end

# ───────────────────────── 7. Run Analysis for Different Asset Counts ──────────────────────

# Run analysis for different numbers of assets
for n_assets in [2, 5, 10,30]
    model, result, params = run_dcc_garch_analysis(n_assets)
    
    if model !== nothing
        println("\n✓ $n_assets-asset analysis completed successfully")
    else
        println("\n✗ $n_assets-asset analysis failed")
    end
    
    println("\n" * "-"^60)
end

println("\nTraditional DCC-GARCH Analysis Complete!")
