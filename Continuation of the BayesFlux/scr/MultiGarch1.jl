###############################################################################
#   Multi-Asset DCC-GARCH Analysis for Different Portfolio Sizes
#   Comparing performance across 2, 5, 10, and 30 assets
###############################################################################

using Optim, LinearAlgebra, Statistics, Distributions
using CSV, DataFrames, Plots
using ForwardDiff, Random, Printf

Random.seed!(123)

# ══════════════════════════════════════════════════════════════════════════════
#                              YOUR DATA LOADING CODE
# ══════════════════════════════════════════════════════════════════════════════

# Load data
etf_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"
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
    # Also handle missing values in risk-free rate
    if any(ismissing, df[!, "rf"])
        df[!, "rf"] = coalesce.(df[!, "rf"], mean(skipmissing(df[!, "rf"])))
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns) # 30 assets: 29 ETFs + 1 risk-free
    # Calculate mean returns for each asset
    μ = mean(returns, dims=1)
    return etf_returns, rf_returns, returns, vec(μ)
end

# Preprocess the data
etf_returns, rf_returns, returns, μ_returns = preprocess_data(df, etf_names)

# ══════════════════════════════════════════════════════════════════════════════
#                              UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Mathematical helpers
logit(x) = log(x / (1 - x))
sigmoid(x) = 1 / (1 + exp(-x))
nearest_pd(A) = (A + A') / 2 + 1e-6 * I

# Transform parameters to ensure constraints
function transform_garch_params(θ)
    ω = exp(θ[1])  # ω > 0
    α = sigmoid(θ[2])  # 0 < α < 1
    β = sigmoid(θ[3]) * (1 - α)  # 0 < β, α + β < 1
    return ω, α, β
end

function transform_dcc_params(θ)
    a = sigmoid(θ[1]) * 0.95  # 0 < a < 0.95
    b = sigmoid(θ[2]) * (0.95 - a)  # 0 < b, a + b < 0.95
    return a, b
end

# ══════════════════════════════════════════════════════════════════════════════
#                              GARCH(1,1) IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

mutable struct GARCH11
    ω::Float64
    α::Float64
    β::Float64
    μ::Float64
    σ²::Vector{Float64}
    residuals::Vector{Float64}
    loglik::Float64
end

function garch11_loglikelihood(θ, returns::Vector{Float64})
    T = length(returns)
    μ = θ[1]
    ω, α, β = transform_garch_params(θ[2:4])
    
    # Check parameter constraints
    if α + β >= 1.0 || ω <= 0 || α <= 0 || β <= 0
        return -Inf
    end
    
    # Initialize conditional variance
    σ² = zeros(T)
    σ²[1] = var(returns)
    
    # Compute conditional variances
    for t in 2:T
        eps2_lag = (returns[t-1] - μ)^2
        σ²[t] = ω + α * eps2_lag + β * σ²[t-1]
        
        if σ²[t] <= 0
            return -Inf
        end
    end
    
    # Compute log-likelihood
    loglik = 0.0
    for t in 1:T
        eps = returns[t] - μ
        loglik -= 0.5 * (log(2π) + log(σ²[t]) + eps^2/σ²[t])
    end
    
    return loglik
end

function estimate_garch11(returns::Vector{Float64}; max_iter=1000)
    # Initial parameter guess
    μ_init = mean(returns)
    σ²_sample = var(returns)
    ω_init = log(0.1 * σ²_sample)
    α_init = logit(0.1)
    β_init = logit(0.8)
    θ₀ = [μ_init, ω_init, α_init, β_init]
    
    # Optimization
    objective = θ -> -garch11_loglikelihood(θ, returns)
    result = optimize(objective, θ₀, BFGS(), 
                     Optim.Options(iterations=max_iter, show_trace=false))
    
    if !Optim.converged(result)
        @warn "GARCH optimization did not converge"
    end
    
    θ_opt = Optim.minimizer(result)
    μ = θ_opt[1]
    ω, α, β = transform_garch_params(θ_opt[2:4])
    
    # Compute fitted values
    T = length(returns)
    σ² = zeros(T)
    σ²[1] = var(returns)
    
    for t in 2:T
        eps2_lag = (returns[t-1] - μ)^2
        σ²[t] = ω + α * eps2_lag + β * σ²[t-1]
    end
    
    # Standardized residuals
    residuals = (returns .- μ) ./ sqrt.(σ²)
    loglik = garch11_loglikelihood(θ_opt, returns)
    
    return GARCH11(ω, α, β, μ, σ², residuals, loglik)
end

# ══════════════════════════════════════════════════════════════════════════════
#                              DCC MODEL IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

mutable struct DCCModel
    garch_models::Vector{GARCH11}
    a::Float64
    b::Float64
    Q̄::Matrix{Float64}
    Q::Array{Float64,3}
    R::Array{Float64,3}
    H::Array{Float64,3}
    loglik::Float64
    residuals::Matrix{Float64}
    n_assets::Int
    estimation_time::Float64
end

function dcc_loglikelihood(θ, standardized_residuals::Matrix{Float64})
    T, N = size(standardized_residuals)
    a, b = transform_dcc_params(θ)
    
    if a + b >= 0.95 || a <= 0 || b <= 0
        return -Inf
    end
    
    Q̄ = cor(standardized_residuals)
    Q = zeros(N, N, T)
    Q[:, :, 1] = Q̄
    loglik = 0.0
    
    for t in 1:T
        if t > 1
            z_lag = standardized_residuals[t-1, :]
            Q[:, :, t] = (1 - a - b) .* Q̄ .+ a .* (z_lag * z_lag') .+ b .* Q[:, :, t-1]
        end
        
        Q_t = Symmetric(Q[:, :, t])
        D_inv = Diagonal(1 ./ sqrt.(max.(diag(Q_t), 1e-8)))
        R_t = Symmetric(D_inv * Q_t * D_inv)
        R_t = nearest_pd(R_t)
        
        try
            L = cholesky(R_t).L
            z_t = standardized_residuals[t, :]
            loglik -= 0.5 * (N * log(2π) + 2 * sum(log, diag(L)) + dot(z_t, R_t \ z_t))
        catch
            return -Inf
        end
    end
    
    return loglik
end

function estimate_dcc(returns::Matrix{Float64}; max_iter=1000, verbose=true)
    start_time = time()
    T, N = size(returns)
    
    if verbose
        println("Estimating DCC model for $N assets over $T periods...")
    end
    
    # Stage 1: Estimate individual GARCH models
    if verbose
        println("Stage 1: Estimating individual GARCH(1,1) models...")
    end
    garch_models = Vector{GARCH11}(undef, N)
    
    for i in 1:N
        if verbose
            print("  Asset $i... ")
        end
        garch_models[i] = estimate_garch11(returns[:, i])
        if verbose
            println("✓")
        end
    end
    
    # Extract standardized residuals
    residuals = hcat([model.residuals for model in garch_models]...)
    
    # Stage 2: Estimate DCC parameters
    if verbose
        println("Stage 2: Estimating DCC parameters...")
    end
    θ₀ = [logit(0.02), logit(0.95)]
    
    objective = θ -> -dcc_loglikelihood(θ, residuals)
    result = optimize(objective, θ₀, BFGS(), 
                     Optim.Options(iterations=max_iter, show_trace=false))
    
    if !Optim.converged(result)
        @warn "DCC optimization did not converge"
    end
    
    θ_opt = Optim.minimizer(result)
    a, b = transform_dcc_params(θ_opt)
    
    # Compute fitted DCC matrices
    Q̄ = cor(residuals)
    Q = zeros(N, N, T)
    R = zeros(N, N, T)
    H = zeros(N, N, T)
    
    Q[:, :, 1] = Q̄
    
    for t in 1:T
        if t > 1
            z_lag = residuals[t-1, :]
            Q[:, :, t] = (1 - a - b) .* Q̄ .+ a .* (z_lag * z_lag') .+ b .* Q[:, :, t-1]
        end
        
        Q_t = Symmetric(Q[:, :, t])
        D_inv = Diagonal(1 ./ sqrt.(max.(diag(Q_t), 1e-8)))
        R[:, :, t] = D_inv * Q_t * D_inv
        
        D_t = Diagonal(sqrt.([garch_models[i].σ²[t] for i in 1:N]))
        H[:, :, t] = D_t * R[:, :, t] * D_t
    end
    
    # Total log-likelihood
    stage1_loglik = sum(model.loglik for model in garch_models)
    stage2_loglik = dcc_loglikelihood(θ_opt, residuals)
    total_loglik = stage1_loglik + stage2_loglik
    
    estimation_time = time() - start_time
    
    if verbose
        println("  DCC parameters: a=$(round(a, digits=4)), b=$(round(b, digits=4))")
        println("  Total log-likelihood: $(round(total_loglik, digits=2))")
        println("  Estimation time: $(round(estimation_time, digits=2)) seconds")
    end
    
    return DCCModel(garch_models, a, b, Q̄, Q, R, H, total_loglik, residuals, N, estimation_time)
end

# ══════════════════════════════════════════════════════════════════════════════
#                              MULTI-ASSET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

function run_multi_asset_analysis(returns::Matrix{Float64}, asset_counts::Vector{Int}; 
                                 train_split::Float64=0.8, verbose::Bool=true)
    """Run DCC-GARCH analysis for multiple asset counts"""
    
    println("MULTI-ASSET DCC-GARCH ANALYSIS")
    println("="^60)
    println("Total available assets: $(size(returns, 2))")
    println("Asset counts to analyze: $asset_counts")
    println("Train-test split: $(Int(train_split*100))%-$(Int((1-train_split)*100))%")
    println()
    
    # Train-test split
    T_total = size(returns, 1)
    train_size = Int(floor(train_split * T_total))
    
    results = Dict()
    summary_stats = []
    
    for n_assets in asset_counts
        println("="^60)
        println("ANALYZING $n_assets ASSETS")
        println("="^60)
        
        if n_assets > size(returns, 2)
            println("⚠️  Requested $n_assets assets but only $(size(returns, 2)) available. Skipping...")
            continue
        end
        
        # Select subset of assets
        if n_assets == size(returns, 2)
            selected_returns = returns
        else
            selected_returns = returns[:, 1:n_assets]
        end
        
        train_returns = selected_returns[1:train_size, :]
        test_returns = selected_returns[train_size+1:end, :]
        
        println("Training data: $(size(train_returns))")
        println("Test data: $(size(test_returns))")
        println()
        
        # Estimate DCC model
        try
            dcc_model = estimate_dcc(train_returns, verbose=verbose)
            
            # Calculate model diagnostics
            aic = -2 * dcc_model.loglik + 2 * (4*n_assets + 2)
            bic = -2 * dcc_model.loglik + log(train_size) * (4*n_assets + 2)
            
            # Calculate average correlation
            avg_correlation = if n_assets > 1
                mean([dcc_model.R[i,j,t] for t in 1:train_size for i in 1:n_assets for j in 1:n_assets if i < j])
            else
                0.0
            end
            
            # Store results
            results[n_assets] = Dict(
                "model" => dcc_model,
                "train_returns" => train_returns,
                "test_returns" => test_returns,
                "loglik" => dcc_model.loglik,
                "aic" => aic,
                "bic" => bic,
                "avg_correlation" => avg_correlation,
                "estimation_time" => dcc_model.estimation_time,
                "dcc_params" => (dcc_model.a, dcc_model.b),
                "persistence" => dcc_model.a + dcc_model.b
            )
            
            # Add to summary stats
            push!(summary_stats, [
                n_assets,
                round(dcc_model.loglik, digits=2),
                round(aic, digits=2),
                round(bic, digits=2),
                round(dcc_model.a, digits=4),
                round(dcc_model.b, digits=4),
                round(dcc_model.a + dcc_model.b, digits=4),
                round(avg_correlation, digits=4),
                round(dcc_model.estimation_time, digits=2)
            ])
            
            println("✅ Successfully estimated DCC model for $n_assets assets")
            
        catch e
            println("❌ Failed to estimate DCC model for $n_assets assets")
            println("Error: $e")
            continue
        end
        
        println()
    end
    
    # Print summary table
    print_summary_table(summary_stats)
    
    # Generate comparison plots
    if length(results) > 1
        create_comparison_plots(results)
    end
    
    return results
end

function print_summary_table(summary_stats)
    """Print formatted summary table"""
    println("="^100)
    println("SUMMARY COMPARISON TABLE")
    println("="^100)
    
    headers = ["Assets", "Log-Lik", "AIC", "BIC", "DCC-a", "DCC-b", "Persist.", "Avg.Corr", "Time(s)"]
    
    # Print headers
    for (i, header) in enumerate(headers)
        print(rpad(header, 10))
        if i < length(headers)
            print(" │ ")
        end
    end
    println()
    
    # Print separator
    println("─"^(10*length(headers) + 3*(length(headers)-1)))
    
    # Print data rows
    for row in summary_stats
        for (i, val) in enumerate(row)
            print(rpad(string(val), 10))
            if i < length(row)
                print(" │ ")
            end
        end
        println()
    end
    println("="^100)
end

function create_comparison_plots(results)
    """Create comparison plots across different asset counts"""
    asset_counts = sort(collect(keys(results)))
    
    # Extract metrics for plotting
    logliks = [results[n]["loglik"] for n in asset_counts]
    aics = [results[n]["aic"] for n in asset_counts]
    bics = [results[n]["bic"] for n in asset_counts]
    dcc_a = [results[n]["dcc_params"][1] for n in asset_counts]
    dcc_b = [results[n]["dcc_params"][2] for n in asset_counts]
    persistence = [results[n]["persistence"] for n in asset_counts]
    avg_corrs = [results[n]["avg_correlation"] for n in asset_counts]
    times = [results[n]["estimation_time"] for n in asset_counts]
    
    # Create plots
    p1 = plot(asset_counts, logliks, marker=:circle, linewidth=2,
              title="Log-Likelihood", xlabel="Number of Assets", ylabel="Log-Likelihood",
              legend=false, color=:blue)
    
    p2 = plot(asset_counts, [aics bics], marker=[:circle :square], linewidth=2,
              title="Information Criteria", xlabel="Number of Assets", ylabel="IC Value",
              label=["AIC" "BIC"], colors=[:red :orange])
    
    p3 = plot(asset_counts, [dcc_a dcc_b persistence], marker=[:circle :square :diamond], linewidth=2,
              title="DCC Parameters", xlabel="Number of Assets", ylabel="Parameter Value",
              label=["DCC-a" "DCC-b" "Persistence"], colors=[:green :purple :brown])
    
    p4 = plot(asset_counts, avg_corrs, marker=:circle, linewidth=2,
              title="Average Correlation", xlabel="Number of Assets", ylabel="Correlation",
              legend=false, color=:teal)
    
    p5 = plot(asset_counts, times, marker=:circle, linewidth=2,
              title="Estimation Time", xlabel="Number of Assets", ylabel="Time (seconds)",
              legend=false, color=:red, yscale=:log10)
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, p5, layout=(3, 2), size=(1200, 1000),
                        plot_title="DCC-GARCH Model Comparison Across Asset Counts")
    
    display(combined_plot)
    
    return combined_plot
end

function analyze_correlation_dynamics(results)
    """Analyze correlation dynamics across different asset counts"""
    println("\nCORRELATION DYNAMICS ANALYSIS")
    println("="^50)
    
    for (n_assets, result) in results
        if n_assets == 1
            continue  # Skip single asset
        end
        
        model = result["model"]
        T = size(model.R, 3)
        
        # Calculate correlation statistics
        correlations = []
        for t in 1:T
            for i in 1:n_assets
                for j in (i+1):n_assets
                    push!(correlations, model.R[i, j, t])
                end
            end
        end
        
        avg_corr = mean(correlations)
        std_corr = std(correlations)
        min_corr = minimum(correlations)
        max_corr = maximum(correlations)
        
        println("$n_assets Assets:")
        println("  Average correlation: $(round(avg_corr, digits=4))")
        println("  Std deviation: $(round(std_corr, digits=4))")
        println("  Range: [$(round(min_corr, digits=4)), $(round(max_corr, digits=4))]")
        println()
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

# Run analysis for 2, 5, 10, and 30 assets
asset_counts = [2, 5, 10, 30]

println("Starting multi-asset DCC-GARCH analysis...")
println("Available data shape: $(size(returns))")
println()

# Run the comprehensive analysis
results = run_multi_asset_analysis(returns, asset_counts, train_split=0.8, verbose=true)

# Additional correlation dynamics analysis
analyze_correlation_dynamics(results)

# Save results summary
println("\nAnalysis complete! Results stored in 'results' dictionary.")
println("Access individual models: results[n_assets][\"model\"]")
println("Available asset counts: $(sort(collect(keys(results))))")