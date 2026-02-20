###############################################################################
#   DCC-GARCH Bayesian Neural Network 
#   - Train/Validation/Test + Rolling Window Forecasting
#   - Portfolio Optimization & VaR
#   - SGNHTS: Stochastic Gradient Nose-Hoover Thermostat only implemented
#   - remove the windowed Feedforward because of numerical instability
#   - MMAP to compare with the SGNHTS results   
###############################################################################

using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Statistics
using MCMCChains, Bijectors
using CSV, DataFrames
using Printf, Plots, StatsPlots

Random.seed!(1212)

const f0 = 0f0
const VAR_CONFIDENCE_LEVELS = [0.01, 0.05, 0.10]

# Critical values for VaR backtesting (chi-square distribution)
const CHI2_1_05 = 3.84  
const CHI2_1_01 = 6.63  
const CHI2_2_05 = 5.99  
const CHI2_2_01 = 9.21  

# Output folder
const OUTPUT_FOLDER = "bnn_dcc_results"

# ══════════════════════════════════════════════════════════════════════════════
#                              VAR RESULTS STRUCTURE & BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════

struct VaRResults
    PF::Float32; TUFF::Int32; LRTUFF::Float32; LRUC::Float32
    LRIND::Float32; LRCC::Float32; BASEL::Int32
end

function VaR_backtest(returns::Array{Float32,1}, VaR::Array{Float32,2}, alphas::Array{Float32,1})
    results = Dict{Float32, VaRResults}()
    
    for i in 1:length(alphas)
        hit = returns .< VaR[:, i]
        n1, n0 = sum(hit), length(hit) - sum(hit)
        PF = Float32(n1/length(hit))
        
        PF == 0 && continue
        
        alpha = alphas[i]
        limits = cumsum(pdf.(Binomial(length(returns), alpha), 1:50))
        green = count(limits .< 0.90)
        yellow = green + count((limits .> 0.90) .& (limits .< 0.99))
        
        TUFF, LRTUFF, LRUC, LRIND, LRCC = Int32(0), Float32(NaN), Float32(NaN), Float32(NaN), Float32(NaN)
        
        if n1 != 0
            first_hit = findfirst(hit)
            if !isnothing(first_hit)
                TUFF = Int32(first_hit)
                try
                    log_term1 = log(alpha * (1-alpha)^(TUFF-1))
                    log_term2 = log((1/TUFF) * (1-1/TUFF)^(TUFF-1))
                    isfinite(log_term1) && isfinite(log_term2) && (LRTUFF = Float32(-2 * log_term1 + 2 * log_term2))
                catch; end
            end
        end
        
        if n1 != 0
            try
                log_likelihood_unrestricted = n1*log(PF) + n0*log(1-PF)
                log_likelihood_restricted = n1*log(alpha) + n0*log(1-alpha)
                if isfinite(log_likelihood_unrestricted) && isfinite(log_likelihood_restricted)
                    LRUC = Float32(-2 * (log_likelihood_restricted - log_likelihood_unrestricted))
                end
            catch; end
        end
        
        if n1 != 0
            n00=n01=n10=n11=0
            for j in 1:(length(returns)-1)
                n00 += (hit[j]==0 && hit[j+1]==0); n01 += (hit[j]==0 && hit[j+1]==1)
                n10 += (hit[j]==1 && hit[j+1]==0); n11 += (hit[j]==1 && hit[j+1]==1)
            end
            
            try
                if (n00+n01) > 0 && (n10+n11) > 0
                    p01, p11, p2 = n01/(n00+n01), n11/(n10+n11), (n01+n11)/(n00+n01+n10+n11)
                    if n11 == 0
                        n01 > 0 && p01 > 0 && (LRIND = Float32(((1-p01)^n00)*(p01^n01)))
                    else
                        if p01 > 0 && p11 > 0 && p2 > 0 && (1-p2) > 0
                            log_unrestricted = n00*log(1-p01) + n01*log(p01) + n10*log(1-p11) + n11*log(p11)
                            log_restricted = (n00+n10)*log(1-p2) + (n01+n11)*log(p2)
                            isfinite(log_unrestricted) && isfinite(log_restricted) && (LRIND = Float32(-2 * (log_restricted - log_unrestricted)))
                        end
                    end
                end
            catch; end
        end
        
        LRCC = if isfinite(LRUC) && isfinite(LRIND)
            LRUC + LRIND
        elseif isfinite(LRUC)
            LRUC
        elseif isfinite(LRIND)
            LRIND
        else
            Float32(NaN)
        end
        
        BASEL = n1 >= yellow ? Int32(-1) : (n1 <= yellow && n1 > green ? Int32(0) : Int32(1))
        
        results[alpha] = VaRResults(PF, TUFF, LRTUFF, LRUC, LRIND, LRCC, BASEL)
    end
    
    return results
end

VaR_backtest(returns::Array{Float32,1}, VaR::Array{Float32,1}, alphas::Array{Float32,1}) = 
    VaR_backtest(returns, reshape(VaR, :, 1), alphas)

function enhanced_var_backtesting(portfolio_returns::Vector{Float64}, daily_portfolio_vars::Vector{Vector{Float64}}, confidence_levels::Vector{Float64} = VAR_CONFIDENCE_LEVELS)
    length(portfolio_returns) != length(daily_portfolio_vars) && return Dict{Float32, VaRResults}()
    
    n_obs, n_levels = length(portfolio_returns), length(confidence_levels)
    returns_f32, alphas_f32 = Float32.(portfolio_returns), Float32.(confidence_levels)
    var_matrix = Matrix{Float32}(undef, n_obs, n_levels)
    
    for t in 1:n_obs, (j, α) in enumerate(confidence_levels)
        var_matrix[t, j] = length(daily_portfolio_vars[t]) >= j ? Float32(-daily_portfolio_vars[t][j]) : Float32(NaN)
    end
    
    valid_rows = [!any(isnan, var_matrix[i, :]) for i in 1:n_obs]
    sum(valid_rows) < n_obs && (returns_f32 = returns_f32[valid_rows]; var_matrix = var_matrix[valid_rows, :])
    length(returns_f32) < 10 && return Dict{Float32, VaRResults}()
    
    try
        return VaR_backtest(returns_f32, var_matrix, alphas_f32)
    catch
        return Dict{Float32, VaRResults}()
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                              SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

sigmoid(x) = 1.0f0/(1.0f0+exp(-x))

σ(x) = 1f0/(1f0+exp(-x))
function transform_ab(a_raw, b_raw)
    a = σ(a_raw)
    b = (1f0 - a) * σ(b_raw)
    return (a, b)
end

nearest_pd(A) = (A + A')/2 + 1e-2*I

function prepare_sequential_data(data, indices)
    valid_indices = indices[indices .< size(data, 1)]
    X = data[valid_indices, :]'
    Y = data[valid_indices .+ 1, :]'
    return Float32.(X), Float32.(Y)
end

function prepare_rolling_window_data(data, test_indices, window_size, approach=:sequential)
    rolling_forecasts = []
    for test_idx in test_indices
        if approach == :sequential
            if test_idx > 1
                X_t = Float32.(vec(data[test_idx-1, :]))
                Y_t = Float32.(vec(data[test_idx, :]))
                push!(rolling_forecasts, (X_t, Y_t, test_idx))   
            end
        end
    end
    return rolling_forecasts
end

function preprocess_financial_data(df, etf_names)
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    if any(ismissing, df[!, "rf"])
        df[!, "rf"] = coalesce.(df[!, "rf"], mean(skipmissing(df[!, "rf"])))
    end
    etf_returns = Matrix{Float64}(df[!, etf_names]) * 100
    rf_returns = Vector{Float64}(df[!, "rf"]) * 100
    returns = hcat(etf_returns, rf_returns)
    μ_returns = mean(returns, dims=1)
    return returns, vec(μ_returns)
end

# ══════════════════════════════════════════════════════════════════════════════
#                    COVARIANCE MATRIX AND PORTFOLIO FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

function construct_minimum_variance_portfolio(cov_matrix)
    Σ = Matrix{Float64}(cov_matrix)
    N = size(Σ, 1)
    ones_vec = ones(Float64, N)
    F = cholesky(Symmetric(nearest_pd(Σ)))
    u = F \ (F' \ ones_vec)
    w = u / (ones_vec' * u)[1]
    return Float32.(w)
end

function calculate_portfolio_var(portfolio_weights, forecast_mean, cov_matrix; confidence_levels=(0.01, 0.05, 0.10))
    μp = Float64.(portfolio_weights' * forecast_mean)[1]
    Σp = Float64.(portfolio_weights' * Matrix{Float64}(cov_matrix) * portfolio_weights)[1]
    σp = sqrt(max(Σp, 0.0))
    z = Dict(0.01=>2.326, 0.05=>1.645, 0.10=>1.282)
    var_results = Dict{Float64, Float64}()
    for α in confidence_levels
        var_results[α] = -μp + z[α]*σp
    end
    return var_results, μp, σp
end

# ══════════════════════════════════════════════════════════════════════════════
#                              PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

function plot_enhanced_var_analysis(portfolio_returns::Vector{Float64}, daily_portfolio_vars::Vector{Vector{Float64}}, var_backtest_results::Dict{Float32, VaRResults}, asset_names::Vector{String})
    isempty(portfolio_returns) || isempty(daily_portfolio_vars) && return nothing
    
    try
        p1 = plot(title="Portfolio VaR vs Actual Returns (BNN)", xlabel="Day", ylabel="Return/VaR", size=(1000, 400))
        plot!(p1, 1:length(portfolio_returns), portfolio_returns, label="Actual Returns", color=:black, linewidth=1.5)
        
        colors = [:red, :orange, :blue]
        for (i, level) in enumerate(VAR_CONFIDENCE_LEVELS)
            if length(daily_portfolio_vars) > 0 && length(daily_portfolio_vars[1]) >= i
                var_series = [length(vars) >= i ? -vars[i] : NaN for vars in daily_portfolio_vars]
                var_series = filter(!isnan, var_series)
                
                if !isempty(var_series) && length(var_series) == length(portfolio_returns)
                    plot!(p1, 1:length(var_series), var_series, label="$(round(Int, 100*level))% VaR", color=colors[i], linewidth=2, linestyle=:dash)
                    violations = portfolio_returns .< var_series
                    any(violations) && scatter!(p1, findall(violations), portfolio_returns[violations], color=colors[i], markersize=3, alpha=0.7, label="$(round(Int, 100*level))% Violations")
                end
            end
        end
        
        p2 = bar(title="VaR Backtesting Results (BNN)", xlabel="Confidence Level", ylabel="Test Statistic", size=(1000, 400))
        if !isempty(var_backtest_results)
            levels, lruc_values, lrind_values, lrcc_values = String[], Float64[], Float64[], Float64[]
            for (alpha, result) in sort(collect(var_backtest_results), by=x->x[1])
                push!(levels, "$(round(Int, 100*alpha))%")
                push!(lruc_values, isfinite(result.LRUC) ? Float64(result.LRUC) : 0.0)
                push!(lrind_values, isfinite(result.LRIND) ? Float64(result.LRIND) : 0.0)
                push!(lrcc_values, isfinite(result.LRCC) ? Float64(result.LRCC) : 0.0)
            end
            
            x_pos = 1:length(levels)
            bar!(p2, x_pos .- 0.25, lruc_values, width=0.2, label="UC Test", alpha=0.8)
            bar!(p2, x_pos, lrind_values, width=0.2, label="IND Test", alpha=0.8)
            bar!(p2, x_pos .+ 0.25, lrcc_values, width=0.2, label="CC Test", alpha=0.8)
            
            hline!(p2, [CHI2_1_05], label="5% Critical (UC/IND: $(CHI2_1_05))", linestyle=:dash, color=:red, linewidth=2)
            hline!(p2, [CHI2_1_01], label="1% Critical (UC/IND: $(CHI2_1_01))", linestyle=:dash, color=:darkred, linewidth=2)
            hline!(p2, [CHI2_2_05], label="5% Critical (CC: $(CHI2_2_05))", linestyle=:dot, color=:orange, linewidth=2)
            hline!(p2, [CHI2_2_01], label="1% Critical (CC: $(CHI2_2_01))", linestyle=:dot, color=:darkorange, linewidth=2)
            xticks!(p2, x_pos, levels)
        end
        
        p3 = histogram(portfolio_returns, bins=30, alpha=0.7, title="Return Distribution with VaR Levels (BNN)", xlabel="Portfolio Returns", ylabel="Frequency", label="Return Distribution", color=:lightblue, size=(1000, 400))
        
        for (i, α) in enumerate(VAR_CONFIDENCE_LEVELS)
            if haskey(var_backtest_results, Float32(α))
                result = var_backtest_results[Float32(α)]
                var_cutoff = quantile(portfolio_returns, α)
                line_color = result.BASEL == 1 ? :green : (result.BASEL == 0 ? :orange : :red)
                vline!(p3, [var_cutoff], label="$(round(Int,100*α))% VaR (PF: $(round(100*result.PF, digits=1))%)", linewidth=3, linestyle=:dash, color=line_color)
            end
        end
        
        return plot(p1, p2, p3, layout=(3,1), size=(1000, 1200))
    catch e
        println("Warning: Enhanced VaR plotting failed: $e")
        return nothing
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                         UPDATED CSV SAVING - SAVES BOTH RNN AND LSTM
# ══════════════════════════════════════════════════════════════════════════════

function save_results_to_csv(all_results::Dict{Int64, Dict{String, Dict{Symbol, Any}}}, output_folder::String = OUTPUT_FOLDER)
    println("\nSaving results to CSV files...")
    
    !isdir(output_folder) && mkdir(output_folder)
    
    for (n_assets, data_dict) in all_results
        println("  Saving results for $n_assets assets...")
        
        if haskey(data_dict, "portfolio_results") && !isempty(data_dict["portfolio_results"])
            portfolio_data = data_dict["portfolio_results"]
            
            # Save results for EACH model (RNN and LSTM) separately
            for (model_type, model_name_short) in [(:rnn, "RNN"), (:lstm, "LSTM")]
                if haskey(portfolio_data, model_type) && haskey(portfolio_data[model_type], "forecasts")
                    forecasts = portfolio_data[model_type]["forecasts"]
                    model_name_full = portfolio_data[model_type]["model_name"]
                    
                    println("    Processing $(model_name_full)...")
                    
                    # 1. Save Portfolio Returns and VaR Values
                    if haskey(forecasts, "portfolio_returns") && haskey(forecasts, "portfolio_var_1")
                        n_obs = length(forecasts["portfolio_returns"])
                        
                        var_df = DataFrame(
                            Day = 1:n_obs,
                            Actual_Returns = forecasts["portfolio_returns"],
                            VaR_1_percent = forecasts["portfolio_var_1"],
                            VaR_5_percent = forecasts["portfolio_var_5"],
                            VaR_10_percent = forecasts["portfolio_var_10"],
                            Portfolio_Volatility = forecasts["portfolio_volatility"],
                            N_Assets = fill(n_assets, n_obs),
                            Model = fill(model_name_full, n_obs)
                        )
                        
                        csv_path = joinpath(output_folder, "portfolio_var_returns_$(model_name_short)_$(n_assets)_assets.csv")
                        CSV.write(csv_path, var_df)
                        println("      ✓ Portfolio VaR vs Returns saved: $csv_path")
                    end
                    
                    # 2. Save VaR Backtesting Results
                    if haskey(forecasts, "var_backtest_results") && !isempty(forecasts["var_backtest_results"])
                        backtest_df = DataFrame(
                            Confidence_Level = Float32[],
                            Violation_Rate = Float32[],
                            TUFF = Int32[],
                            LRTUFF = Float32[],
                            LRUC = Float32[],
                            LRIND = Float32[],
                            LRCC = Float32[],
                            Basel_Zone = String[],  # Changed to String for clarity
                            N_Assets = Int[],
                            Model = String[]
                        )
                        
                        for (alpha, result) in sort(collect(forecasts["var_backtest_results"]), by=x->x[1])
                            basel_str = result.BASEL == 1 ? "Green" : (result.BASEL == 0 ? "Yellow" : "Red")
                            push!(backtest_df, (
                                alpha, result.PF, result.TUFF, result.LRTUFF,
                                result.LRUC, result.LRIND, result.LRCC, basel_str,
                                n_assets, model_name_full
                            ))
                        end
                        
                        csv_path = joinpath(output_folder, "var_backtest_$(model_name_short)_$(n_assets)_assets.csv")
                        CSV.write(csv_path, backtest_df)
                        println("      ✓ VaR backtesting results saved: $csv_path")
                    end
                    
                    # 3. Save Portfolio Weights
                    if haskey(forecasts, "mv_weights")
                        n_obs = size(forecasts["mv_weights"], 2)
                        n_assets_actual = size(forecasts["mv_weights"], 1)
                        
                        weights_df = DataFrame()
                        for i in 1:n_assets_actual
                            weights_df[!, "Asset_$(i)_Weight"] = forecasts["mv_weights"][i, :]
                        end
                        weights_df[!, "Day"] = 1:n_obs
                        weights_df[!, "N_Assets"] = fill(n_assets, n_obs)
                        weights_df[!, "Model"] = fill(model_name_full, n_obs)
                        
                        csv_path = joinpath(output_folder, "mvp_weights_$(model_name_short)_$(n_assets)_assets.csv")
                        CSV.write(csv_path, weights_df)
                        println("      ✓ MVP weights saved: $csv_path")
                    end
                end
            end
        end
        
        # 4. Save Model Performance Summary (both models together)
        if haskey(data_dict, "model_results")
            model_results = data_dict["model_results"]
            
            summary_df = DataFrame(
                Model = String[],
                Metric = String[],
                Value = Any[],
                N_Assets = Int[]
            )
            
            for (model_type, result) in model_results
                model_name_full = result["model_name"]
                
                # Add all metrics for this model
                push!(summary_df, (model_name_full, "R_hat", result["r_hat"], n_assets))
                push!(summary_df, (model_name_full, "DCC_alpha", result["dcc_params"][1], n_assets))
                push!(summary_df, (model_name_full, "DCC_beta", result["dcc_params"][2], n_assets))
                push!(summary_df, (model_name_full, "Converged", result["converged"], n_assets))
                push!(summary_df, (model_name_full, "Training_RMSE", result["train"]["rmse"], n_assets))
                push!(summary_df, (model_name_full, "Validation_RMSE", result["validation"]["rmse"], n_assets))
                push!(summary_df, (model_name_full, "Test_RMSE", result["test"]["rmse"], n_assets))
                push!(summary_df, (model_name_full, "Coverage_90", result["test"]["coverage_90"], n_assets))
                push!(summary_df, (model_name_full, "Coverage_95", result["test"]["coverage_95"], n_assets))
            end
            
            csv_path = joinpath(output_folder, "model_summary_$(n_assets)_assets.csv")
            CSV.write(csv_path, summary_df)
            println("    ✓ Model summary saved: $csv_path")
        end
        
        # 5. Save Portfolio Performance Comparison (both models together)
        if haskey(data_dict, "portfolio_results")
            portfolio_data = data_dict["portfolio_results"]
            
            perf_df = DataFrame(
                Model = String[],
                Avg_Portfolio_Return = Float64[],
                Portfolio_Volatility = Float64[],
                Avg_VaR_1 = Float64[],
                Avg_VaR_5 = Float64[],
                Avg_VaR_10 = Float64[],
                N_Assets = Int[]
            )
            
            for (model_type, result_dict) in portfolio_data
                if haskey(result_dict, "performance")
                    perf = result_dict["performance"]
                    push!(perf_df, (
                        result_dict["model_name"],
                        perf["avg_portfolio_return"],
                        perf["portfolio_volatility"],
                        perf["avg_var_1"],
                        perf["avg_var_5"],
                        perf["avg_var_10"],
                        n_assets
                    ))
                end
            end
            
            csv_path = joinpath(output_folder, "portfolio_performance_$(n_assets)_assets.csv")
            CSV.write(csv_path, perf_df)
            println("    ✓ Portfolio performance saved: $csv_path")
        end
    end
    
    println("\n✓ All results saved in '$output_folder' directory")
end

# ══════════════════════════════════════════════════════════════════════════════
#                              LIKELIHOOD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

struct DCCGarchRNNSequential{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
end
DCCGarchRNNSequential(nc::NetConstructor{T,F}, prior::D, N::Int) where {T,F,D<:Distribution} =
    DCCGarchRNNSequential{T,F,D}(2, nc, prior, N)

function (ℓ::DCCGarchRNNSequential)(x::Matrix{T}, y::Matrix{T}, θnet::AbstractVector, θlike::AbstractVector) where {T}
    θnet, θlike = T.(θnet), T.(θlike)
    a, b = transform_ab(θlike...)
    net = ℓ.nc(θnet)
    N, Tsteps = ℓ.N, size(x, 2)
    rnn_layer, dense_layer = net[1], net[2]
    hidden_size = size(rnn_layer.cell.Wi, 1)
    h = zeros(T, hidden_size)
    outs = map(1:Tsteps) do t
        input_t = view(x, :, t)
        h = tanh.(rnn_layer.cell.Wi * input_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
        return dense_layer(h)
    end
    return compute_dcc_likelihood(outs, y, a, b, N, Tsteps, ℓ.prior)
end

struct DCCGarchLSTMSequential{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
end
DCCGarchLSTMSequential(nc::NetConstructor{T,F}, prior::D, N::Int) where {T,F,D<:Distribution} =
    DCCGarchLSTMSequential{T,F,D}(2, nc, prior, N)

function (ℓ::DCCGarchLSTMSequential)(x::Matrix{T}, y::Matrix{T}, θnet::AbstractVector, θlike::AbstractVector) where {T}
    θnet, θlike = T.(θnet), T.(θlike)
    a, b = transform_ab(θlike...)
    net = ℓ.nc(θnet)
    N, Tsteps = ℓ.N, size(x, 2)
    lstm_layer, dense_layer = net[1], net[2]
    hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
    h, c = zeros(T, hidden_size), zeros(T, hidden_size)
    outs = map(1:Tsteps) do t
        input_t = view(x, :, t)
        gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
        i_gate = sigmoid.(gates[1:hidden_size])
        f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
        g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
        o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
        c = f_gate .* c .+ i_gate .* g_gate
        h = o_gate .* tanh.(c)
        return dense_layer(h)
    end
    return compute_dcc_likelihood(outs, y, a, b, N, Tsteps, ℓ.prior)
end

function compute_dcc_likelihood(outs, y, a, b, N, Tsteps, prior)
    T = eltype(y)
    μ     = hcat(map(o -> o[1:N], outs)...)
    logσ2 = hcat(map(o -> o[N+1:2N], outs)...)
    σ = exp.(logσ2 ./ 2)
    z = (y .- μ) ./ σ
    Q̄ = Tsteps > 1 ? (z * z') / Tsteps : Matrix{T}(I, N, N)
    d = 1 ./ sqrt.(max.(diag(Q̄), T(1e-10)))
    Q̄ = Symmetric(Diagonal(d) * Q̄ * Diagonal(d))
    Q, logl = Q̄, zero(T)
    for t in 1:Tsteps
        d = 1 ./ sqrt.(max.(diag(Q), T(1e-10)))
        R = Symmetric(Diagonal(d) * Q * Diagonal(d))
        D = Diagonal(view(σ, :, t))
        H = nearest_pd(D * R * D)
        L = cholesky(Symmetric(H)).L
        diff = view(y, :, t) .- view(μ, :, t)
        quad = sum(abs2, L \ diff)
        logl -= 0.5 * (N * log(2π) + 2 * sum(log, diag(L)) + quad)
        if t < Tsteps
            zt = view(z, :, t)
            Q = (1 - a - b) .* Q̄ .+ a .* (zt * zt') .+ b .* Q
        end
    end
    return logl
end

# ══════════════════════════════════════════════════════════════════════════════
#                              MODEL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

build_rnn_model(N::Int, hidden_size::Int) = Chain(RNN(N => hidden_size), Dense(hidden_size => 2N))
build_lstm_model(N::Int, hidden_size::Int) = Chain(LSTM(N => hidden_size), Dense(hidden_size => 2N))

# ══════════════════════════════════════════════════════════════════════════════
#                         TRAINING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

function train_model_with_validation(model_type::Symbol, X_train, Y_train, X_val, Y_val, N::Int; 
        hidden_size::Int=32, window_size::Int=22, prior_std::Float32=0.15f0,
        mcmc_samples::Int=10000, chains::Int=4)

    println("Training $model_type model with $N assets...")
    if model_type == :rnn
        net = build_rnn_model(N, hidden_size)
        likelihood_type = DCCGarchRNNSequential
    elseif model_type == :lstm
        net = build_lstm_model(N, hidden_size)
        likelihood_type = DCCGarchLSTMSequential
    else
        error("Unknown model type: $model_type")
    end

    nc = destruct(net)
    like = likelihood_type(nc, Normal(0f0, prior_std), N)
    prior = GaussianPrior(nc, prior_std)
    init = InitialiseAllSame(Normal(0f0, prior_std), like, prior)

    bnn_train = BNN(X_train, Y_train, like, prior, init)
    bnn_val   = BNN(X_val,   Y_val,   like, prior, init)

    println("Finding MAP estimate...")
    θmap = find_mode(bnn_train, 50, 1000, FluxModeFinder(bnn_train, Flux.ADAM()))

    println("Starting MCMC sampling...")
    step_size = N <= 5 ? 1f-4 : 2f-5
    sampler = SGNHTS(step_size, 1f0; xi=1f0^2, μ=10f0)
    ch_raw = mcmc(bnn_train, chains, mcmc_samples, sampler)

    burn_in = mcmc_samples ÷ 2
    ch_burn_raw, ch2d = apply_burnin_and_flatten(ch_raw, burn_in)

    val_loglik = compute_validation_likelihood(bnn_val, ch2d)

    return bnn_train, bnn_val, ch2d, ch_burn_raw, net, val_loglik
end

function compute_validation_likelihood(bnn_val, chain2d)
    n_samples = min(size(chain2d, 2), 100)
    idxs = rand(1:size(chain2d, 2), n_samples)
    val_ll = map(idxs) do i
        θnet = chain2d[1:end-2, i]
        θlike = chain2d[end-1:end, i]
        bnn_val.like(bnn_val.x, bnn_val.y, θnet, θlike)
    end
    
    finite_ll = filter(isfinite, val_ll)
    return length(finite_ll) > 0 ? mean(finite_ll) : -Inf
end

# ══════════════════════════════════════════════════════════════════════════════
#                         CHAIN HELPERS 
# ══════════════════════════════════════════════════════════════════════════════

function apply_burnin_and_flatten(ch, burn_in)
    if ndims(ch) == 2
        ch_b = ch[:, burn_in+1:end]
        return ch_b, ch_b
    elseif ndims(ch) == 3
        P, S, C = size(ch)
        S_b = max(S - burn_in, 0)
        @assert S_b > 0 "Burn-in removes all samples; reduce burn_in or increase samples."
        ch_b = ch[:, burn_in+1:end, :]
        ch2d = reshape(permutedims(ch_b, (1,2,3)), P, S_b*C)
        return ch_b, ch2d
    else
        error("Unsupported chain dims: $(ndims(ch))")
    end
end

function naive_prediction(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    T_steps = size(x, 2)
    output_size = 2 * N * T_steps
    
    yhats = Array{T, 2}(undef, output_size, size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        yh = vec(net(x))
        
        if all(isfinite, yh)
            yhats[:, i] = yh
        end
    end
    return yhats
end

function compute_rhat_bayesflux(bnn, ch)
    net_params = ch[1:end-2, :]
    yhats = naive_prediction(bnn, net_params)
    
    if any(!isfinite, yhats)
        println("Warning: Non-finite predictions detected in R-hat computation")
        finite_cols = [all(isfinite, yhats[:, i]) for i in 1:size(yhats, 2)]
        if sum(finite_cols) < 10
            return NaN
        end
        yhats = yhats[:, finite_cols]
    end
    
    chain_yhat = Chains(yhats')
    return maximum(summarystats(chain_yhat)[:, :rhat])
end

get_θnet(chain2d) = chain2d[1:end-2, :]
get_θlike(chain2d) = chain2d[end-1:end, :]

# ══════════════════════════════════════════════════════════════════════════════
#                        FORWARD PASSES
# ══════════════════════════════════════════════════════════════════════════════

function rnn_single_forecast(net, X_t, N)
    rnn_layer, dense_layer = net[1], net[2]
    hidden_size = size(rnn_layer.cell.Wi, 1)
    h = zeros(Float32, hidden_size)
    h = tanh.(rnn_layer.cell.Wi * X_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
    return dense_layer(h)
end

function lstm_single_forecast(net, X_t, N)
    lstm_layer, dense_layer = net[1], net[2]
    hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
    h, c = zeros(Float32, hidden_size), zeros(Float32, hidden_size)
    gates = lstm_layer.cell.Wi * X_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
    i_gate = sigmoid.(gates[1:hidden_size])
    f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
    g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
    o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
    c = f_gate .* c .+ i_gate .* g_gate
    h = o_gate .* tanh.(c)
    return dense_layer(h)
end

function rnn_forward_pass(net, x, N)
    rnn_layer, dense_layer = net[1], net[2]
    hidden_size = size(rnn_layer.cell.Wi, 1)
    h = zeros(Float32, hidden_size)
    return map(1:size(x, 2)) do t
        input_t = view(x, :, t)
        h = tanh.(rnn_layer.cell.Wi * input_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
        return dense_layer(h)[1:N]
    end
end

function lstm_forward_pass(net, x, N)
    lstm_layer, dense_layer = net[1], net[2]
    hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
    h, c = zeros(Float32, hidden_size), zeros(Float32, hidden_size)
    return map(1:size(x, 2)) do t
        input_t = view(x, :, t)
        gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
        i_gate = sigmoid.(gates[1:hidden_size])
        f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
        g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
        o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
        c = f_gate .* c .+ i_gate .* g_gate
        h = o_gate .* tanh.(c)
        return dense_layer(h)[1:N]
    end
end

function get_moments(net, x, N, likelihood_type)
    n_obs = size(x, 2)
    results = if likelihood_type <: DCCGarchRNNSequential
        rnn_moments(net, x, N)
    elseif likelihood_type <: DCCGarchLSTMSequential
        lstm_moments(net, x, N)
    else
        map(t -> net(view(x, :, t)), 1:n_obs)
    end
    μ = hcat([r[1:N] for r in results]...)
    σ = exp.(hcat([r[N+1:2N] for r in results]...) ./ 2)
    return μ, σ
end

function rnn_moments(net, x, N)
    rnn_layer, dense_layer = net[1], net[2]
    hidden_size = size(rnn_layer.cell.Wi, 1)
    h = zeros(Float32, hidden_size)
    return map(1:size(x, 2)) do t
        input_t = view(x, :, t)
        h = tanh.(rnn_layer.cell.Wi * input_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
        return dense_layer(h)
    end
end

function lstm_moments(net, x, N)
    lstm_layer, dense_layer = net[1], net[2]
    hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
    h, c = zeros(Float32, hidden_size), zeros(Float32, hidden_size)
    return map(1:size(x, 2)) do t
        input_t = view(x, :, t)
        gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
        i_gate = sigmoid.(gates[1:hidden_size])
        f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
        g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
        o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
        c = f_gate .* c .+ i_gate .* g_gate
        h = o_gate .* tanh.(c)
        return dense_layer(h)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                      DCC STATE COMPUTATION (PER SAMPLE)
# ══════════════════════════════════════════════════════════════════════════════

function compute_dcc_training_state_per_sample(bnn_train, θnet, θlike, N)
    a, b = transform_ab(θlike...)
    net = bnn_train.like.nc(θnet)
    μ, σ = get_moments(net, bnn_train.x, N, typeof(bnn_train.like))
    z = (bnn_train.y .- μ) ./ (σ .+ 1f-6)
    Tsteps = size(z, 2)
    Qbar = (z * z') / Tsteps
    d = 1 ./ sqrt.(max.(diag(Qbar), 1f-10))
    Qbar = Symmetric(Diagonal(d) * Qbar * Diagonal(d))
    Q = copy(Qbar)
    for t in 1:Tsteps
        zt = view(z, :, t)
        Q = (1 - a - b) .* Qbar .+ a .* (zt * zt') .+ b .* Q
    end
    z_last = view(z, :, Tsteps)
    return (Float32.(Qbar), Float32.(Q), Float32.(z_last), Float32(a), Float32(b))
end

function one_step_dcc_covariance(μ_pred, logσ2_pred, a, b, Q_prev, z_prev, Qbar)
    N = length(μ_pred)
    σ = exp.(logσ2_pred ./ 2)
    Q_next = (1f0 - a - b) .* Qbar .+ a .* (z_prev * z_prev') .+ b .* Q_prev
    d = 1f0 ./ sqrt.(max.(diag(Q_next), 1f-10))
    R_next = Symmetric(Diagonal(d) * Q_next * Diagonal(d))
    D = Diagonal(σ)
    H = nearest_pd(Matrix(D * R_next * D))
    return H, Q_next
end

# ══════════════════════════════════════════════════════════════════════════════
#                      FULLY BAYESIAN ROLLING FORECAST
# ══════════════════════════════════════════════════════════════════════════════

function rolling_covariance_forecast_with_portfolio_analysis(bnn_train, chain2d, rolling_data, model_type::Symbol, N::Int)
    println("Performing FULLY BAYESIAN rolling window forecasting with portfolio analysis and VaR backtesting...")
    n_test = length(rolling_data)
    n_samples = min(size(chain2d, 2), 100)
    sample_indices = 1:n_samples

    println("  Initializing DCC states for $n_samples posterior samples...")
    dcc_states = []
    for idx in sample_indices
        θnet = chain2d[1:end-2, idx]
        θlike = chain2d[end-1:end, idx]
        state = compute_dcc_training_state_per_sample(bnn_train, θnet, θlike, N)
        push!(dcc_states, state)
    end

    sample_results = Dict{String, Any}(
        "means" => Array{Float32}(undef, N, n_test, n_samples),
        "covariances" => Array{Float32}(undef, N, N, n_test, n_samples),
        "weights" => Array{Float32}(undef, N, n_test, n_samples),
        "portfolio_returns" => Array{Float32}(undef, n_test, n_samples),
        "portfolio_vars_1" => Array{Float32}(undef, n_test, n_samples),
        "portfolio_vars_5" => Array{Float32}(undef, n_test, n_samples),
        "portfolio_vars_10" => Array{Float32}(undef, n_test, n_samples),
        "portfolio_volatility" => Array{Float32}(undef, n_test, n_samples)
    )

    for (t_idx, (X_t, Y_t, _)) in enumerate(rolling_data)
        if t_idx % 25 == 0
            println("  Processing forecast $t_idx/$n_test")
        end

        for (s_idx, chain_idx) in enumerate(sample_indices)
            θnet = chain2d[1:end-2, chain_idx]
            θlike = chain2d[end-1:end, chain_idx]
            Qbar, Q_prev, z_prev, a, b = dcc_states[s_idx]

            net = bnn_train.like.nc(θnet)
            out = model_type == :rnn ? rnn_single_forecast(net, X_t, N) :
                  model_type == :lstm ? lstm_single_forecast(net, X_t, N) : net(X_t)
            
            if !all(isfinite, out) || length(out) < 2*N
                continue
            end

            μ_pred = out[1:N]
            logσ2_pred = clamp.(out[N+1:2N], -10f0, 5f0)

            H, Q_next = one_step_dcc_covariance(μ_pred, logσ2_pred, a, b, Q_prev, z_prev, Qbar)

            w = construct_minimum_variance_portfolio(H)
            if !all(isfinite, w)
                continue
            end

            var_results, μp, σp = calculate_portfolio_var(w, μ_pred, H)
            if !isfinite(σp) || any(!isfinite, values(var_results))
                continue
            end

            sample_results["means"][:, t_idx, s_idx] = μ_pred
            sample_results["covariances"][:, :, t_idx, s_idx] = Float32.(H)
            sample_results["weights"][:, t_idx, s_idx] = w
            sample_results["portfolio_volatility"][t_idx, s_idx] = Float32(σp)
            sample_results["portfolio_vars_1"][t_idx, s_idx] = Float32(var_results[0.01])
            sample_results["portfolio_vars_5"][t_idx, s_idx] = Float32(var_results[0.05])
            sample_results["portfolio_vars_10"][t_idx, s_idx] = Float32(var_results[0.10])
            sample_results["portfolio_returns"][t_idx, s_idx] = Float32(w' * Y_t)

            σ = exp.(logσ2_pred ./ 2f0)
            z_next = (Y_t .- μ_pred) ./ (σ .+ 1f-6)
            dcc_states[s_idx] = (Qbar, Q_next, z_next, a, b)
        end
    end

    forecasts = Dict{String, Any}(
        "means" => mean(sample_results["means"], dims=3)[:, :, 1],
        "covariance_matrices" => mean(sample_results["covariances"], dims=4)[:, :, :, 1],
        "mv_weights" => mean(sample_results["weights"], dims=3)[:, :, 1],
        "portfolio_returns" => mean(sample_results["portfolio_returns"], dims=2)[:, 1],
        "portfolio_var_1" => mean(sample_results["portfolio_vars_1"], dims=2)[:, 1],
        "portfolio_var_5" => mean(sample_results["portfolio_vars_5"], dims=2)[:, 1],
        "portfolio_var_10" => mean(sample_results["portfolio_vars_10"], dims=2)[:, 1],
        "portfolio_volatility" => mean(sample_results["portfolio_volatility"], dims=2)[:, 1],
        "actuals" => Array{Float32}(undef, N, n_test)
    )

    for (t_idx, (_, Y_t, _)) in enumerate(rolling_data)
        forecasts["actuals"][:, t_idx] = Y_t
    end

    if !isempty(forecasts["portfolio_returns"])
        portfolio_returns_vec = Vector{Float64}(forecasts["portfolio_returns"])
        daily_portfolio_vars_vec = Vector{Vector{Float64}}()
        
        for t in 1:n_test
            push!(daily_portfolio_vars_vec, [
                Float64(forecasts["portfolio_var_1"][t]),
                Float64(forecasts["portfolio_var_5"][t]),
                Float64(forecasts["portfolio_var_10"][t])
            ])
        end
        
        var_backtest_results = enhanced_var_backtesting(portfolio_returns_vec, daily_portfolio_vars_vec, VAR_CONFIDENCE_LEVELS)
        forecasts["var_backtest_results"] = var_backtest_results
        
        if !isempty(var_backtest_results)
            println("VaR Backtesting Results (Critical Values: UC/IND at 5%=$CHI2_1_05, 1%=$CHI2_1_01; CC at 5%=$CHI2_2_05, 1%=$CHI2_2_01):")
            for (alpha, result) in sort(collect(var_backtest_results), by=x->x[1])
                level_pct = round(Int, 100*alpha)
                basel_status = result.BASEL == 1 ? "Green" : (result.BASEL == 0 ? "Yellow" : "Red")
                uc_reject_5 = isfinite(result.LRUC) && result.LRUC > CHI2_1_05 ? "*" : ""
                uc_reject_1 = isfinite(result.LRUC) && result.LRUC > CHI2_1_01 ? "**" : ""
                ind_reject_5 = isfinite(result.LRIND) && result.LRIND > CHI2_1_05 ? "*" : ""
                ind_reject_1 = isfinite(result.LRIND) && result.LRIND > CHI2_1_01 ? "**" : ""
                cc_reject_5 = isfinite(result.LRCC) && result.LRCC > CHI2_2_05 ? "*" : ""
                cc_reject_1 = isfinite(result.LRCC) && result.LRCC > CHI2_2_01 ? "**" : ""
                
                println("  $level_pct%: PF=$(round(100*result.PF,digits=1))% $basel_status | UC=$(round(result.LRUC,digits=2))$uc_reject_5$uc_reject_1 IND=$(round(result.LRIND,digits=2))$ind_reject_5$ind_reject_1 CC=$(round(result.LRCC,digits=2))$cc_reject_5$cc_reject_1")
            end
            println("  * = Reject at 5%, ** = Reject at 1%")
        end
    end

    return forecasts
end

function rolling_window_forecast(bnn_train, chain2d, rolling_data, model_type::Symbol, N::Int)
    println("Performing rolling window forecasting (stochastic samples)...")
    n_test = length(rolling_data)
    n_samples = min(size(chain2d, 2), 200)
    
    forecasts = Dict{String, Any}(
        "means" => Array{Float32}(undef, N, n_test),
        "vars" => Array{Float32}(undef, N, n_test),
        "samples" => Array{Float32}(undef, N, n_test, n_samples),
        "actuals" => Array{Float32}(undef, N, n_test),
        "loglik" => Array{Float32}(undef, n_test)
    )
    sample_indices = rand(1:size(chain2d, 2), n_samples)
    
    for (t_idx, (X_t, Y_t, _)) in enumerate(rolling_data)
        if t_idx % 50 == 0
            println("  Processing forecast $t_idx/$n_test")
        end
        forecasts["actuals"][:, t_idx] = Y_t
        forecast_samples = Array{Float32}(undef, N, n_samples)
        logliks = fill(-Inf32, n_samples)
        
        for (s_idx, chain_idx) in enumerate(sample_indices)
            θnet = chain2d[1:end-2, chain_idx]
            θlike = chain2d[end-1:end, chain_idx]
            mean_pred, var_pred, sample_pred, loglik = single_point_forecast(
                θnet, θlike, X_t, Y_t, bnn_train.like, model_type, N)
            
            if all(isfinite, mean_pred) && all(isfinite, var_pred) && all(isfinite, sample_pred) && isfinite(loglik)
                forecast_samples[:, s_idx] = sample_pred
                logliks[s_idx] = loglik
            end
        end
        
        valid_mask = [all(isfinite, forecast_samples[:, i]) for i in 1:n_samples]
        valid_samples_filtered = forecast_samples[:, valid_mask]
        
        if size(valid_samples_filtered, 2) > 0
            forecasts["means"][:, t_idx] = mean(valid_samples_filtered, dims=2)[:, 1]
            forecasts["vars"][:, t_idx] = var(valid_samples_filtered, dims=2)[:, 1]
        else
            error("No valid samples available for time step $t_idx")
        end
        
        forecasts["samples"][:, t_idx, :] = forecast_samples
        
        finite_logliks = logliks[isfinite.(logliks)]
        forecasts["loglik"][t_idx] = length(finite_logliks) > 0 ? mean(finite_logliks) : -Inf32
    end
    return forecasts
end

function single_point_forecast(θnet, θlike, X_t, Y_t, likelihood, model_type::Symbol, N::Int)
    net = likelihood.nc(θnet)
    output = model_type == :rnn ? rnn_single_forecast(net, X_t, N) :
             model_type == :lstm ? lstm_single_forecast(net, X_t, N) : net(X_t)
    μ_pred = output[1:N]
    logσ2_pred = clamp.(output[N+1:2N], -10.0f0, 5.0f0)
    σ_pred = exp.(logσ2_pred ./ 2.0f0)
    
    H = Diagonal((σ_pred .^ 2)) + 1e-6f0*I
    
    L = cholesky(Symmetric(H)).L
    sample_pred = μ_pred + L * randn(Float32, N)
    diff = Y_t - μ_pred
    quad = sum(abs2, L \ diff)
    loglik = -0.5f0 * (N * log(2π) + 2 * sum(log, diag(L)) + quad)
    
    return μ_pred, σ_pred .^ 2, sample_pred, loglik
end

# ══════════════════════════════════════════════════════════════════════════════
#                              EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

function compute_coverage_probability(y_true, y_samples, confidence_level)
    α = 1 - confidence_level
    lower_q, upper_q = α/2, 1 - α/2
    N, n_test, n_samp = size(y_samples)
    total, inside = 0, 0
    
    for i in 1:N, t in 1:n_test
        samples_it = y_samples[i, t, :]
        valid_samples = samples_it[isfinite.(samples_it)]
        
        if length(valid_samples) < 5 || !isfinite(y_true[i, t])
            continue
        end
        
        lo = quantile(valid_samples, lower_q)
        hi = quantile(valid_samples, upper_q)
        
        if isfinite(lo) && isfinite(hi)
            inside += (lo <= y_true[i, t] <= hi) ? 1 : 0
            total += 1
        end
    end
    
    return total > 0 ? inside / total : NaN
end

function compute_forecast_quantiles(y_true, y_samples, quantiles)
    N, n_test, n_samples = size(y_samples)
    n_q = length(quantiles)
    observed = zeros(Float64, n_q)
    
    for qi in 1:n_q
        q = quantiles[qi]
        below, count = 0, 0
        
        for i in 1:N, t in 1:n_test
            samples = y_samples[i, t, :]
            valid = samples[isfinite.(samples)]
            
            if length(valid) < 5 || !isfinite(y_true[i, t])
                continue
            end
            
            thr = quantile(valid, q)
            if isfinite(thr)
                below += y_true[i, t] < thr ? 1 : 0
                count += 1
            end
        end
        
        observed[qi] = count > 0 ? below / count : NaN
    end
    
    return observed
end

function generate_predictions(bnn, chain2d)
    N = bnn.like.N
    n_samples = size(chain2d, 2)
    n_obs = size(bnn.x, 2)
    yhats = Array{Float32}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i in 1:n_samples
        θnet = chain2d[1:end-2, i]
        net  = bnn.like.nc(θnet)
        predictions = if typeof(bnn.like) <: DCCGarchRNNSequential
            rnn_forward_pass(net, bnn.x, N)
        elseif typeof(bnn.like) <: DCCGarchLSTMSequential
            lstm_forward_pass(net, bnn.x, N)
        else
            map(t -> net(view(bnn.x, :, t))[1:N], 1:n_obs)
        end
        
        pred_vec = vcat(predictions...)
        if all(isfinite, pred_vec)
            yhats[:, i] = pred_vec
        end
    end
    
    return yhats
end

function compute_observed_quantiles(y_true, y_samples, quantiles, N)
    n_q = length(quantiles)
    observed = zeros(Float64, n_q)
    y_true_vec = vec(y_true)
    total_points = min(size(y_samples, 1), length(y_true_vec))
    y_samples_trim = y_samples[1:total_points, :]
    y_true_vec = y_true_vec[1:total_points]
    
    for i in 1:n_q
        q = quantiles[i]
        below, count = 0, 0
        
        for j in 1:total_points
            val = y_true_vec[j]
            if !isfinite(val)
                continue
            end
            
            point_samples = y_samples_trim[j, :]
            valid = point_samples[isfinite.(point_samples)]
            
            if length(valid) < 5
                continue
            end
            
            thr = quantile(valid, q)
            if isfinite(thr)
                below += val < thr ? 1 : 0
                count += 1
            end
        end
        
        observed[i] = count > 0 ? below / count : NaN
    end
    
    return observed
end

function evaluate_in_sample(bnn, chain2d, set_name::String)
    N = bnn.like.N
    yhats = generate_predictions(bnn, chain2d)
    pred_mean = mean(yhats, dims=2)[:, 1]
    n_obs = size(bnn.y, 2)
    y_true_flat = vec(bnn.y)
    
    valid_mask = isfinite.(y_true_flat) .& isfinite.(pred_mean)
    if sum(valid_mask) > 0
        rmse = sqrt(mean((y_true_flat[valid_mask] .- pred_mean[valid_mask]).^2))
    else
        rmse = Inf
    end
    
    logliks = [
        bnn.like(bnn.x, bnn.y, chain2d[1:end-2, i], chain2d[end-1:end, i])
        for i in 1:min(100, size(chain2d, 2))
    ]
    finite_logliks = filter(isfinite, logliks)
    avg_loglik = length(finite_logliks) > 0 ? mean(finite_logliks) : -Inf
    
    posterior_samples = generate_posterior_predictive(bnn, chain2d, bnn.x, bnn.y)
    t_q = 0.05:0.05:0.95
    o_q = compute_observed_quantiles(bnn.y', posterior_samples, t_q, N)
    mad_qq = mean(abs.(o_q[isfinite.(o_q)] .- t_q[isfinite.(o_q)]))
    
    return Dict(
        "set_name" => set_name,
        "rmse" => rmse,
        "loglik" => avg_loglik,
        "mad_qq" => mad_qq,
        "qq_observed" => o_q,
        "qq_target" => t_q
    )
end

function generate_posterior_predictive(bnn, chain2d, x, y)
    N = bnn.like.N
    n_samples = min(size(chain2d, 2), 200)
    n_obs = size(x, 2)
    pred_samples = Array{Float32}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i in 1:n_samples
        θnet = chain2d[1:end-2, i]
        net = bnn.like.nc(θnet)
        μ_all, σ_all = get_moments(net, x, N, typeof(bnn.like))
        σ_all = clamp.(σ_all, 1e-6f0, 10f0)
        samples = similar(μ_all)
        
        for t in 1:n_obs
            samples[:, t] = μ_all[:, t] + σ_all[:, t] .* randn(Float32, N)
        end
        
        sample_vec = vec(samples)
        if all(isfinite, sample_vec)
            pred_samples[:, i] = sample_vec
        end
    end
    
    return pred_samples
end

function evaluate_out_of_sample(forecasts, set_name::String)
    y_true = forecasts["actuals"]
    y_pred_mean = forecasts["means"]
    y_pred_var = haskey(forecasts, "vars") ? forecasts["vars"] : similar(y_pred_mean)
    logliks = haskey(forecasts, "loglik") ? forecasts["loglik"] : fill(Float32(NaN), size(y_true, 2))
    
    valid_mask = isfinite.(y_true) .& isfinite.(y_pred_mean)
    rmse = sqrt(mean((y_true[valid_mask] .- y_pred_mean[valid_mask]).^2))
    
    finite_logliks = logliks[isfinite.(logliks)]
    avg_loglik = length(finite_logliks) > 0 ? mean(finite_logliks) : NaN
    
    coverage_90 = haskey(forecasts, "samples") ? compute_coverage_probability(y_true, forecasts["samples"], 0.90) : NaN
    coverage_95 = haskey(forecasts, "samples") ? compute_coverage_probability(y_true, forecasts["samples"], 0.95) : NaN
    
    t_q = 0.05:0.05:0.95
    o_q = haskey(forecasts, "samples") ? compute_forecast_quantiles(y_true, forecasts["samples"], t_q) : fill(NaN, length(t_q))
    mad_qq = mean(abs.(o_q[isfinite.(o_q)] .- t_q[isfinite.(o_q)]))
    
    return Dict(
        "set_name" => set_name,
        "rmse" => rmse,
        "loglik" => avg_loglik,
        "coverage_90" => coverage_90,
        "coverage_95" => coverage_95,
        "mad_qq" => mad_qq,
        "qq_observed" => o_q,
        "qq_target" => t_q
    )
end

function evaluate_portfolio_performance(forecasts, model_name::String)
    println("\n" * "="^60)
    println("PORTFOLIO PERFORMANCE ANALYSIS: $model_name")
    println("="^60)
    
    finite_returns = forecasts["portfolio_returns"][isfinite.(forecasts["portfolio_returns"])]
    finite_volatility = forecasts["portfolio_volatility"][isfinite.(forecasts["portfolio_volatility"])]
    finite_var_1 = forecasts["portfolio_var_1"][isfinite.(forecasts["portfolio_var_1"])]
    finite_var_5 = forecasts["portfolio_var_5"][isfinite.(forecasts["portfolio_var_5"])]
    finite_var_10 = forecasts["portfolio_var_10"][isfinite.(forecasts["portfolio_var_10"])]
    
    avg_portfolio_return = length(finite_returns) > 0 ? mean(finite_returns) : 0.0
    portfolio_volatility = length(finite_volatility) > 0 ? mean(finite_volatility) : 0.0
    avg_var_1 = length(finite_var_1) > 0 ? mean(finite_var_1) : 0.0
    avg_var_5 = length(finite_var_5) > 0 ? mean(finite_var_5) : 0.0
    avg_var_10 = length(finite_var_10) > 0 ? mean(finite_var_10) : 0.0
    
    println("Portfolio Performance Summary:")
    println("  Avg Realized Return: $(round(avg_portfolio_return, digits=4))")
    println("  Avg Predicted Volatility: $(round(portfolio_volatility, digits=4))")
    println("  VaR (avg across horizon): 1%=$(round(avg_var_1, digits=4)), 5%=$(round(avg_var_5, digits=4)), 10%=$(round(avg_var_10, digits=4))")
    
    if haskey(forecasts, "var_backtest_results") && !isempty(forecasts["var_backtest_results"])
        println("\nVaR Backtesting Results (Critical Values: UC/IND at 5%=$CHI2_1_05, 1%=$CHI2_1_01; CC at 5%=$CHI2_2_05, 1%=$CHI2_2_01):")
        for (alpha, result) in sort(collect(forecasts["var_backtest_results"]), by=x->x[1])
            level_pct = round(Int, 100*alpha)
            basel_status = result.BASEL == 1 ? "Green" : (result.BASEL == 0 ? "Yellow" : "Red")
            uc_reject = isfinite(result.LRUC) ? (result.LRUC > CHI2_1_01 ? "**" : (result.LRUC > CHI2_1_05 ? "*" : "")) : ""
            ind_reject = isfinite(result.LRIND) ? (result.LRIND > CHI2_1_01 ? "**" : (result.LRIND > CHI2_1_05 ? "*" : "")) : ""
            cc_reject = isfinite(result.LRCC) ? (result.LRCC > CHI2_2_01 ? "**" : (result.LRCC > CHI2_2_05 ? "*" : "")) : ""
            
            println("  $level_pct%: PF=$(round(100*result.PF,digits=1))% $basel_status Zone | UC=$(round(result.LRUC,digits=2))$uc_reject IND=$(round(result.LRIND,digits=2))$ind_reject CC=$(round(result.LRCC,digits=2))$cc_reject")
        end
        println("  * = Reject at 5%, ** = Reject at 1%")
    end
    
    return Dict(
        "avg_portfolio_return" => avg_portfolio_return,
        "portfolio_volatility" => portfolio_volatility,
        "avg_var_1" => avg_var_1,
        "avg_var_5" => avg_var_5,
        "avg_var_10" => avg_var_10,
        "var_backtest_results" => haskey(forecasts, "var_backtest_results") ? forecasts["var_backtest_results"] : Dict{Float32, VaRResults}()
    )
end

function evaluate_model_comprehensive(bnn_train, bnn_val, chain2d, ch_raw, rolling_forecasts, model_name::String)
    N = bnn_train.like.N
    train_results = evaluate_in_sample(bnn_train, chain2d, "Training")
    val_results   = evaluate_in_sample(bnn_val,   chain2d, "Validation")
    test_results  = evaluate_out_of_sample(rolling_forecasts, "Test")
    
    r_hat = compute_rhat_bayesflux(bnn_train, ch_raw)
    
    θlike_mean = vec(mean(get_θlike(chain2d); dims=2))
    dcc_params = transform_ab(θlike_mean...)
    
    results = Dict(
        "model_name" => model_name,
        "n_assets" => N,
        "r_hat" => r_hat,
        "dcc_params" => dcc_params,
        "converged" => isfinite(r_hat) ? (r_hat < 1.2) : false,
        "train" => train_results,
        "validation" => val_results,
        "test" => test_results
    )
    return results
end

function print_portfolio_comparison(portfolio_results, n_assets)
    println("\n" * "="^80)
    println("PORTFOLIO PERFORMANCE COMPARISON ($n_assets Assets)")
    println("="^80)
    println("\n┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    println("│ Model               │ Avg Return  │ Volatility  │ VaR 1%      │ VaR 5%      │ VaR 10%     │")
    println("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM")]
        if haskey(portfolio_results, model_type)
            perf = portfolio_results[model_type]["performance"]
            avg = lpad(string(round(perf["avg_portfolio_return"], digits=4)), 11)
            vol = lpad(string(round(perf["portfolio_volatility"], digits=4)), 11)
            v1  = lpad(string(round(perf["avg_var_1"], digits=4)), 11)
            v5  = lpad(string(round(perf["avg_var_5"], digits=4)), 11)
            v10 = lpad(string(round(perf["avg_var_10"], digits=4)), 11)
            println("│ $(rpad(name, 19)) │ $avg │ $vol │ $v1 │ $v5 │ $v10 │")
        end
    end
    println("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")
end

function print_comprehensive_results_summary(results, n_assets)
    println("\n" * "="^80)
    println("COMPREHENSIVE RESULTS SUMMARY ($n_assets Assets)")
    println("="^80)
    println("\n┌─────────────────────┬─────────┬──────────────┬─────────────────────────────────────────────────┐")
    println("│ Model               │ R-hat   │ DCC (α, β)   │ RMSE (Train/Val/Test)                           │")
    println("├─────────────────────┼─────────┼──────────────┼─────────────────────────────────────────────────┤")
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM")]
        if haskey(results, model_type)
            r = results[model_type]
            α, β = r["dcc_params"]
            train_rmse = r["train"]["rmse"]
            val_rmse = r["validation"]["rmse"]
            test_rmse = r["test"]["rmse"]
            println("│ $(rpad(name, 19)) │ $(lpad(string(round(r["r_hat"], digits=3)), 7)) │ $(lpad(string(round(α, digits=3)), 4)), $(lpad(string(round(β, digits=3)), 4)) │ $(lpad(string(round(train_rmse, digits=6)), 7)) / $(lpad(string(round(val_rmse, digits=6)), 7)) / $(lpad(string(round(test_rmse, digits=6)), 7)) │")
        end
    end
    println("└─────────────────────┴─────────┴──────────────┴─────────────────────────────────────────────────┘")
end

function print_var_backtesting_summary(portfolio_results, n_assets)
    println("\n" * "="^80)
    println("VAR BACKTESTING SUMMARY ($n_assets Assets)")
    println("="^80)
    println("Critical Values: UC/IND tests - 5%: $CHI2_1_05, 1%: $CHI2_1_01 | CC test - 5%: $CHI2_2_05, 1%: $CHI2_2_01")
    println("Legend: * = Reject at 5%, ** = Reject at 1%")
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM")]
        if haskey(portfolio_results, model_type)
            perf = portfolio_results[model_type]["performance"]
            if haskey(perf, "var_backtest_results") && !isempty(perf["var_backtest_results"])
                println("\n$name VaR Backtesting:")
                for (alpha, result) in sort(collect(perf["var_backtest_results"]), by=x->x[1])
                    level_pct = round(Int, 100*alpha)
                    basel_icon = result.BASEL == 1 ? "Green" : (result.BASEL == 0 ? "Yellow" : "Red")
                    uc_reject = isfinite(result.LRUC) ? (result.LRUC > CHI2_1_01 ? "**" : (result.LRUC > CHI2_1_05 ? "*" : "")) : ""
                    ind_reject = isfinite(result.LRIND) ? (result.LRIND > CHI2_1_01 ? "**" : (result.LRIND > CHI2_1_05 ? "*" : "")) : ""
                    cc_reject = isfinite(result.LRCC) ? (result.LRCC > CHI2_2_01 ? "**" : (result.LRCC > CHI2_2_05 ? "*" : "")) : ""
                    
                    println("  $level_pct%: PF=$(round(100*result.PF,digits=1))% TUFF=$(result.TUFF) $basel_icon | UC=$(round(result.LRUC,digits=2))$uc_reject IND=$(round(result.LRIND,digits=2))$ind_reject CC=$(round(result.LRCC,digits=2))$cc_reject")
                end
            end
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

function comprehensive_dcc_analysis_with_portfolio(data_path::String, n_assets::Int=30)
    println("COMPREHENSIVE DCC-GARCH ANALYSIS WITH PORTFOLIO OPTIMIZATION & VAR BACKTESTING")
    println("Using FULLY BAYESIAN approach - propagating uncertainty through DCC parameters")

    train_index = 110:3450
    val_index   = 3451:3700
    test_index  = 3701:3949

    println("Data splits:")
    println("  Training: $(first(train_index)) to $(last(train_index)) ($(length(train_index)) observations)")
    println("  Validation: $(first(val_index)) to $(last(val_index)) ($(length(val_index)) observations)")
    println("  Test: $(first(test_index)) to $(last(test_index)) ($(length(test_index)) observations)")

    df = CSV.read(data_path, DataFrame)
    etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414",
                 "16418", "16421", "16423", "16424", "16426", "16433", "16437",
                 "16452", "16460", "24697", "27635", "28272", "28273", "28274",
                 "28275", "28276", "28277", "28278", "28279", "28280", "31372", "31466"]

    returns, μ_returns = preprocess_financial_data(df, etf_names)

    if n_assets < 30
        selected_indices = 1:n_assets
        returns = returns[:, selected_indices]
        μ_returns = μ_returns[selected_indices]
        selected_assets = etf_names[1:min(n_assets, length(etf_names))]
    else
        selected_assets = vcat(etf_names, ["rf"])
    end

    y = Float32.(returns)

    println("Selected assets: $(length(selected_assets)) total")
    println("Data dimensions: $(size(y)) (observations × assets)")

    X_train_seq, Y_train_seq = prepare_sequential_data(y, train_index)
    X_val_seq,   Y_val_seq   = prepare_sequential_data(y, val_index)
    window_size = 22
    rolling_data_seq = prepare_rolling_window_data(y, test_index, window_size, :sequential)

    println("Training data prepared:")
    println("  Sequential: X=$(size(X_train_seq)), Y=$(size(Y_train_seq))")
    println("  Rolling forecasts: $(length(rolling_data_seq)) time points")

    hidden_size = max(16, min(64, 8 * n_assets))
    models_to_run = [
        (:rnn,     X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, rolling_data_seq, "Sequential RNN"),
        (:lstm,    X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, rolling_data_seq, "Sequential LSTM")
    ]

    results = Dict{Symbol, Any}()
    portfolio_results = Dict{Symbol, Any}()

    for (model_type, X_train, Y_train, X_val, Y_val, rolling_data, model_name) in models_to_run
        println("\n" * "="^60)
        println("TRAINING AND EVALUATING $model_name WITH PORTFOLIO ANALYSIS & VAR BACKTESTING")
        println("="^60)
        bnn_train, bnn_val, chain2d, ch_raw, net, val_loglik = train_model_with_validation(
            model_type, X_train, Y_train, X_val, Y_val, n_assets; hidden_size=hidden_size, mcmc_samples=10000, chains=4)
        println("Training complete. Validation log-likelihood: $(round(val_loglik, digits=4))")

        portfolio_forecasts = rolling_covariance_forecast_with_portfolio_analysis(
            bnn_train, chain2d, rolling_data, model_type, n_assets)
        portfolio_performance = evaluate_portfolio_performance(portfolio_forecasts, model_name)

        portfolio_results[model_type] = Dict(
            "forecasts" => portfolio_forecasts,
            "performance" => portfolio_performance,
            "model_name" => model_name
        )

        regular_forecasts = rolling_window_forecast(bnn_train, chain2d, rolling_data, model_type, n_assets)
        result = evaluate_model_comprehensive(bnn_train, bnn_val, chain2d, ch_raw, regular_forecasts, model_name)
        result["val_loglik"] = val_loglik
        results[model_type] = result

        println("\n$model_name Results Summary:")
        println("  R-hat: $(round(result["r_hat"], digits=4))")
        println("  DCC params (α, β): $(round.(result["dcc_params"], digits=4))")
        println("  Converged: $(result["converged"] ? "✓" : "✗")")
        println("  Training RMSE: $(round(result["train"]["rmse"], digits=6))")
        println("  Validation RMSE: $(round(result["validation"]["rmse"], digits=6))")
        println("  Test RMSE: $(round(result["test"]["rmse"], digits=6))")
        
        if haskey(portfolio_forecasts, "portfolio_returns") && haskey(portfolio_forecasts, "var_backtest_results")
            portfolio_returns_vec = Vector{Float64}(portfolio_forecasts["portfolio_returns"])
            daily_portfolio_vars_vec = Vector{Vector{Float64}}()
            
            n_test = length(portfolio_returns_vec)
            for t in 1:n_test
                push!(daily_portfolio_vars_vec, [
                    Float64(portfolio_forecasts["portfolio_var_1"][t]),
                    Float64(portfolio_forecasts["portfolio_var_5"][t]),
                    Float64(portfolio_forecasts["portfolio_var_10"][t])
                ])
            end
            
            var_plot = plot_enhanced_var_analysis(portfolio_returns_vec, daily_portfolio_vars_vec, 
                                                portfolio_forecasts["var_backtest_results"], selected_assets)
            
            if var_plot !== nothing
                display(var_plot)
                plot_filename = joinpath(OUTPUT_FOLDER, "bnn_var_analysis_$(model_type)_$(n_assets)_assets.png")
                savefig(var_plot, plot_filename)
                println("  VaR analysis plot saved: $plot_filename")
            end
        end
        
        if haskey(portfolio_forecasts, "portfolio_volatility")
            mvp_vol_plot = plot(1:length(portfolio_forecasts["portfolio_volatility"]), 
                              portfolio_forecasts["portfolio_volatility"],
                              title="MVP Volatility Evolution - $model_name ($n_assets Assets)", 
                              xlabel="Day", ylabel="Portfolio Volatility", 
                              linewidth=2, color=:purple, label="MVP Volatility",
                              size=(1000, 400))
            display(mvp_vol_plot)
            vol_filename = joinpath(OUTPUT_FOLDER, "bnn_mvp_evolution_$(model_type)_$(n_assets)_assets.png")
            savefig(mvp_vol_plot, vol_filename)
            println("  MVP evolution plot saved: $vol_filename")
        end
    end

    print_comprehensive_results_summary(results, n_assets)
    print_portfolio_comparison(portfolio_results, n_assets)
    print_var_backtesting_summary(portfolio_results, n_assets)
    
    all_results = Dict(n_assets => Dict("model_results" => results, "portfolio_results" => portfolio_results))
    save_results_to_csv(all_results, OUTPUT_FOLDER)

    return results, portfolio_results
end

# ══════════════════════════════════════════════════════════════════════════════
#                              RUN THE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

data_path = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"

asset_counts = [2, 5, 10, 15, 29, 30]
all_results = Dict{Int64, Tuple{Dict{Symbol, Any}, Dict{Symbol, Any}}}()

for n_assets in asset_counts
    println("RUNNING ANALYSIS FOR $n_assets ASSETS")
    
    results, portfolio_results = comprehensive_dcc_analysis_with_portfolio(data_path, n_assets)
    all_results[n_assets] = (results, portfolio_results)
    
    println("\nAnalysis complete for $n_assets assets! Results saved and plots generated.")
end

println("ALL ANALYSES COMPLETE!")
println("Results generated for asset counts: $(collect(keys(all_results)))")


####################Parameter counting for models####################

# Model building functions (copy from your code)
build_rnn_model(N::Int, hidden_size::Int) = Chain(RNN(N => hidden_size), Dense(hidden_size => 2N))
build_lstm_model(N::Int, hidden_size::Int) = Chain(LSTM(N => hidden_size), Dense(hidden_size => 2N))

# Parameter counting function
function count_parameters(net)
    total = 0
    for layer in net
        for param in Flux.params(layer)
            total += length(param)
        end
    end
    return total + 2  # Add 2 for DCC parameters (a, b)
end

# Calculate for all your models
function print_parameter_counts()
    asset_counts = [2, 5, 10, 15, 29, 30]
    
    println("Parameter Counts:")
    println("="^60)
    
    for n_assets in asset_counts
        hidden_size = max(16, min(32, 4 * n_assets))
        
        # RNN model
        rnn_net = build_rnn_model(n_assets, hidden_size)
        rnn_params = count_parameters(rnn_net)
        
        # LSTM model  
        lstm_net = build_lstm_model(n_assets, hidden_size)
        lstm_params = count_parameters(lstm_net)
        
        println("N=$n_assets (hidden=$hidden_size):")
        println("  RNN:  $rnn_params parameters")
        println("  LSTM: $lstm_params parameters")
        println()
    end
end

# Run it
print_parameter_counts()