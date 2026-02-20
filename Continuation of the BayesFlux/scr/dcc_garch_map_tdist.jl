###############################################################################
#   DCC-GARCH Bayesian Neural Network with t-Distribution 
#   - MAP to compare with the SGNHTS results   
###############################################################################

using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Statistics
using CSV, DataFrames
using Printf, Plots, StatsPlots
using SpecialFunctions

Random.seed!(1212)

const f0 = 0f0
const VAR_CONFIDENCE_LEVELS = [0.01, 0.05, 0.10]

# Critical values for VaR backtesting (chi-square distribution)
const CHI2_1_05 = 3.84
const CHI2_1_01 = 6.63
const CHI2_2_05 = 5.99
const CHI2_2_01 = 9.21

# Output folder
const OUTPUT_FOLDER = "map_tdist_results"

# ══════════════════════════════════════════════════════════════════════════════
#                         VAR RESULTS STRUCTURE & BACKTESTING
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
#                             SHARED UTILITIES
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
#                  COVARIANCE MATRIX AND PORTFOLIO FUNCTIONS
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

# CORRECTED FUNCTION: Fixed VaR calculation to match theoretical formula
function calculate_portfolio_var(portfolio_weights, forecast_mean, cov_matrix, nu::Float64; confidence_levels=(0.01, 0.05, 0.10))
    μp = Float64.(portfolio_weights' * forecast_mean)[1]
    Σp = Float64.(portfolio_weights' * Matrix{Float64}(cov_matrix) * portfolio_weights)[1]
    σp = sqrt(max(Σp, 0.0)) * sqrt((nu - 2) / nu)  # Corrected: (ν-2)/ν scaling
    
    var_results = Dict{Float64, Float64}()
    for α in confidence_levels
        t_quantile = quantile(TDist(nu), α)  # This is negative for α < 0.5
        var_results[α] = -(μp + t_quantile * σp)  # Corrected: negative of entire expression
    end
    return var_results, μp, σp
end

# ══════════════════════════════════════════════════════════════════════════════
#                      ENHANCED PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

function plot_enhanced_var_analysis(portfolio_returns::Vector{Float64}, daily_portfolio_vars::Vector{Vector{Float64}}, var_backtest_results::Dict{Float32, VaRResults}, asset_names::Vector{String}, model_name::String)
    isempty(portfolio_returns) || isempty(daily_portfolio_vars) && return nothing
    
    try
        p1 = plot(title="Portfolio VaR vs Actual Returns ($model_name - MAP t-dist)", xlabel="Day", ylabel="Return/VaR (% points)", size=(1000, 400))
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
        
        p2 = bar(title="VaR Backtesting Results ($model_name - MAP t-dist)", xlabel="Confidence Level", ylabel="Test Statistic", size=(1000, 400))
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
        
        p3 = histogram(portfolio_returns, bins=30, alpha=0.7, title="Return Distribution with VaR Levels ($model_name - MAP t-dist)", xlabel="Portfolio Returns", ylabel="Frequency", label="Return Distribution", color=:lightblue, size=(1000, 400))
        
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
#                         LIKELIHOOD FUNCTIONS (T-DISTRIBUTION)
# ══════════════════════════════════════════════════════════════════════════════

struct DCCGarchRNNSequential{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
    nu::Float32
end
DCCGarchRNNSequential(nc::NetConstructor{T,F}, prior::D, N::Int, nu::Float32=5.0f0) where {T,F,D<:Distribution} =
    DCCGarchRNNSequential{T,F,D}(2, nc, prior, N, nu)

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
    return compute_dcc_likelihood_tdist(outs, y, a, b, N, Tsteps, ℓ.prior, ℓ.nu)
end

struct DCCGarchLSTMSequential{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
    nu::Float32
end
DCCGarchLSTMSequential(nc::NetConstructor{T,F}, prior::D, N::Int, nu::Float32=5.0f0) where {T,F,D<:Distribution} =
    DCCGarchLSTMSequential{T,F,D}(2, nc, prior, N, nu)

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
    return compute_dcc_likelihood_tdist(outs, y, a, b, N, Tsteps, ℓ.prior, ℓ.nu)
end

function compute_dcc_likelihood_tdist(outs, y, a, b, N, Tsteps, prior, nu)
    T = eltype(y)
    μ     = hcat(map(o -> o[1:N], outs)...)
    logσ2 = hcat(map(o -> o[N+1:2N], outs)...)
    σ = exp.(logσ2 ./ 2)
    z = (y .- μ) ./ σ
    Q̄ = Tsteps > 1 ? (z * z') / Tsteps : Matrix{T}(I, N, N)
    d = 1 ./ sqrt.(max.(diag(Q̄), T(1e-10)))
    Q̄ = Symmetric(Diagonal(d) * Q̄ * Diagonal(d))
    Q, logl = Q̄, zero(T)
    
    nu_T = T(nu)
    log_gamma_term = loggamma((nu_T + N)/2) - loggamma(nu_T/2)
    log_const = -(N/2) * log(nu_T * π)
    
    for t in 1:Tsteps
        d = 1 ./ sqrt.(max.(diag(Q), T(1e-10)))
        R = Symmetric(Diagonal(d) * Q * Diagonal(d))
        D = Diagonal(view(σ, :, t))
        H = nearest_pd(D * R * D)
        L = cholesky(Symmetric(H)).L
        diff = view(y, :, t) .- view(μ, :, t)
        quad = sum(abs2, L \ diff)
        
        logl += log_gamma_term + log_const - sum(log, diag(L)) - ((nu_T + N)/2) * log(1 + quad/nu_T)
        
        if t < Tsteps
            zt = view(z, :, t)
            Q = (1 - a - b) .* Q̄ .+ a .* (zt * zt') .+ b .* Q
        end
    end
    return logl
end

# ══════════════════════════════════════════════════════════════════════════════
#                         MODEL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

build_rnn_model(N::Int, hidden_size::Int) = Chain(RNN(N => hidden_size), Dense(hidden_size => 2N))
build_lstm_model(N::Int, hidden_size::Int) = Chain(LSTM(N => hidden_size), Dense(hidden_size => 2N))

# ══════════════════════════════════════════════════════════════════════════════
#                         TRAINING (MAP-ONLY)
# ══════════════════════════════════════════════════════════════════════════════

function train_model_map(model_type::Symbol, X_train, Y_train, X_val, Y_val, N::Int; 
        hidden_size::Int=32, prior_std::Float32=0.15f0, nu::Float32=5.0f0)

    println("Training $model_type model with $N assets (MAP estimation, t-distribution with ν=$nu)...")
    
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
    like = likelihood_type(nc, Normal(0f0, prior_std), N, nu)
    prior = GaussianPrior(nc, prior_std)
    init = InitialiseAllSame(Normal(0f0, prior_std), like, prior)

    bnn_train = BNN(X_train, Y_train, like, prior, init)
    bnn_val   = BNN(X_val,   Y_val,   like, prior, init)

    println("Finding MAP estimate...")
    opt = FluxModeFinder(bnn_train, Flux.RMSProp())
    θmap = find_mode(bnn_train, 10, 1000, opt)
    println("MAP estimation complete.")

    return bnn_train, bnn_val, θmap, net
end

# ══════════════════════════════════════════════════════════════════════════════
#                 FORWARD PASSES & SINGLE-STEP FORECASTS
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
#                 DCC STATE FROM TRAINING - MAP VERSION
# ══════════════════════════════════════════════════════════════════════════════

function compute_dcc_training_state_map(bnn_train, θmap)
    N = bnn_train.like.N
    θnet_map = θmap[1:end-2]
    θlike_map = θmap[end-1:end]
    a, b = transform_ab(θlike_map...)
    net = bnn_train.like.nc(θnet_map)
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
#          ROLLING PORTFOLIO FORECASTING WITH VAR (MAP VERSION)
# ══════════════════════════════════════════════════════════════════════════════

function rolling_covariance_forecast_map(bnn_train, θmap, rolling_data, model_type::Symbol, N::Int, nu::Float64)
    println("Performing MAP-based rolling window forecasting with portfolio analysis (t-distribution, ν=$nu)...")
    n_test = length(rolling_data)

    net = bnn_train.like.nc(θmap[1:end-2])
    
    Qbar, Q_prev, z_prev, a_map, b_map = compute_dcc_training_state_map(bnn_train, θmap)

    forecasts = Dict{String, Any}(
        "means" => Array{Float32}(undef, N, n_test),
        "covariance_matrices" => Array{Float32}(undef, N, N, n_test),
        "mv_weights" => Array{Float32}(undef, N, n_test),
        "portfolio_returns" => Array{Float32}(undef, n_test),  
        "portfolio_var_1" => Array{Float32}(undef, n_test),
        "portfolio_var_5" => Array{Float32}(undef, n_test),
        "portfolio_var_10" => Array{Float32}(undef, n_test),
        "portfolio_volatility" => Array{Float32}(undef, n_test),
        "actuals" => Array{Float32}(undef, N, n_test)
    )

    for (t_idx, (X_t, Y_t, _)) in enumerate(rolling_data)
        if t_idx % 25 == 0
            println("  Processing forecast $t_idx/$n_test")
        end
        forecasts["actuals"][:, t_idx] = Y_t

        out = model_type == :rnn ? rnn_single_forecast(net, X_t, N) : lstm_single_forecast(net, X_t, N)
        
        if !all(isfinite, out)
            error("Non-finite prediction from MAP network at step $t_idx")
        end

        μ_pred = out[1:N]
        logσ2_pred = clamp.(out[N+1:2N], -10f0, 5f0)

        H, Q_next = one_step_dcc_covariance(μ_pred, logσ2_pred, a_map, b_map, Q_prev, z_prev, Qbar)

        w = construct_minimum_variance_portfolio(H)
        
        var_results, μp, σp = calculate_portfolio_var(w, μ_pred, H, nu)
        
        forecasts["means"][:, t_idx] = μ_pred
        forecasts["covariance_matrices"][:, :, t_idx] = Float32.(H)
        forecasts["mv_weights"][:, t_idx] = w
        forecasts["portfolio_volatility"][t_idx] = Float32(σp)
        forecasts["portfolio_var_1"][t_idx] = Float32(var_results[0.01])
        forecasts["portfolio_var_5"][t_idx] = Float32(var_results[0.05])
        forecasts["portfolio_var_10"][t_idx] = Float32(var_results[0.10])
        forecasts["portfolio_returns"][t_idx] = Float32(w' * Y_t)

        σ_step = exp.(logσ2_pred ./ 2f0) .+ 1f-6
        z_prev = (Y_t .- μ_pred) ./ σ_step
        Q_prev = Q_next
    end

    if !isempty(forecasts["portfolio_returns"])
        portfolio_returns_vec = Vector{Float64}(forecasts["portfolio_returns"])
        daily_portfolio_vars_vec = [[Float64(forecasts["portfolio_var_1"][t]),
                                     Float64(forecasts["portfolio_var_5"][t]),
                                     Float64(forecasts["portfolio_var_10"][t])] for t in 1:n_test]
        
        var_backtest_results = enhanced_var_backtesting(portfolio_returns_vec, daily_portfolio_vars_vec, VAR_CONFIDENCE_LEVELS)
        forecasts["var_backtest_results"] = var_backtest_results
    end

    return forecasts
end

# ══════════════════════════════════════════════════════════════════════════════
#                         EVALUATION METRICS (MAP VERSION)
# ══════════════════════════════════════════════════════════════════════════════

function evaluate_model_map(bnn_train, bnn_val, bnn_test_forecasts, θmap, model_name::String, nu::Float64)
    N = bnn_train.like.N
    net = bnn_train.like.nc(θmap[1:end-2])

    μ_train, _ = get_moments(net, bnn_train.x, N, typeof(bnn_train.like))
    train_rmse = sqrt(mean((vec(bnn_train.y) .- vec(μ_train)).^2))

    μ_val, _ = get_moments(net, bnn_val.x, N, typeof(bnn_val.like))
    val_rmse = sqrt(mean((vec(bnn_val.y) .- vec(μ_val)).^2))

    test_rmse = sqrt(mean((vec(bnn_test_forecasts["actuals"]) .- vec(bnn_test_forecasts["means"])).^2))

    dcc_params = transform_ab(θmap[end-1:end]...)

    results = Dict(
        "model_name" => model_name,
        "n_assets" => N,
        "nu" => nu,
        "dcc_params" => dcc_params,
        "train_rmse" => train_rmse,
        "validation_rmse" => val_rmse,
        "test_rmse" => test_rmse
    )
    return results
end

function evaluate_portfolio_performance(forecasts::Dict{String, Any}, model_name::String)
    if !haskey(forecasts, "portfolio_returns") || !haskey(forecasts, "portfolio_volatility")
        println("Warning: Missing required forecast keys for portfolio evaluation")
        return Dict(
            "model_name" => model_name,
            "mean_return" => NaN,
            "volatility" => NaN,
            "sharpe_ratio" => NaN,
            "avg_predicted_volatility" => NaN
        )
    end
    
    portfolio_returns = forecasts["portfolio_returns"]
    portfolio_volatility = forecasts["portfolio_volatility"]
    
    mean_return = mean(portfolio_returns)
    volatility = std(portfolio_returns)
    sharpe_ratio = volatility > 0 ? mean_return / volatility : NaN
    
    return Dict(
        "model_name" => model_name,
        "mean_return" => mean_return,
        "volatility" => volatility,
        "sharpe_ratio" => sharpe_ratio,
        "avg_predicted_volatility" => mean(portfolio_volatility)
    )
end

function print_map_results_summary(results, n_assets, nu)
    println("\n" * "="^80)
    println("COMPREHENSIVE RESULTS SUMMARY (MAP, $n_assets Assets, t-distribution ν=$nu)")
    println("="^80)
    println("\n┌─────────────────────┬──────────────┬─────────────────────────────────────────────────┐")
    println("│ Model               │ DCC (α, β)   │ RMSE (Train/Val/Test)                           │")
    println("├─────────────────────┼──────────────┼─────────────────────────────────────────────────┤")
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM")]
        if haskey(results, model_type)
            r = results[model_type]
            α, β = r["dcc_params"]
            train_rmse = r["train_rmse"]
            val_rmse = r["validation_rmse"]
            test_rmse = r["test_rmse"]
            println("│ $(rpad(name, 19)) │ $(lpad(string(round(α, digits=3)), 4)), $(lpad(string(round(β, digits=3)), 4)) │ $(lpad(string(round(train_rmse, digits=6)), 7)) / $(lpad(string(round(val_rmse, digits=6)), 7)) / $(lpad(string(round(test_rmse, digits=6)), 7)) │")
        end
    end
    println("└─────────────────────┴──────────────┴─────────────────────────────────────────────────┘")
end

function print_portfolio_comparison(portfolio_results::Dict, n_assets::Int)
    println("\n" * "="^80)
    println("PORTFOLIO PERFORMANCE COMPARISON ($n_assets Assets)")
    println("="^80)
    println("\n┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┐")
    println("│ Model               │ Mean Return │ Volatility  │ Sharpe Ratio│ Avg Pred Volatility │")
    println("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤")
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM")]
        if haskey(portfolio_results, model_type)
            result_dict = portfolio_results[model_type]
            if haskey(result_dict, "performance")
                perf = result_dict["performance"]
                println("│ $(rpad(name, 19)) │ $(lpad(string(round(perf["mean_return"], digits=4)), 11)) │ $(lpad(string(round(perf["volatility"], digits=4)), 11)) │ $(lpad(string(round(perf["sharpe_ratio"], digits=4)), 12)) │ $(lpad(string(round(perf["avg_predicted_volatility"], digits=4)), 19)) │")
            end
        end
    end
    println("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┘")
end

function print_var_backtesting_summary(portfolio_results::Dict, n_assets::Int)
    println("\n" * "="^100)
    println("VAR BACKTESTING SUMMARY ($n_assets Assets)")
    println("="^100)
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM")]
        if haskey(portfolio_results, model_type)
            result_dict = portfolio_results[model_type]
            if haskey(result_dict, "forecasts")
                forecasts = result_dict["forecasts"]
                if haskey(forecasts, "var_backtest_results")
                    var_results = forecasts["var_backtest_results"]
                    
                    println("\n$name:")
                    println("─"^100)
                    println("┌────────────┬──────────┬─────────┬──────────┬──────────┬──────────┬──────────┬────────────┐")
                    println("│ Conf Level │ PF (%)   │ TUFF    │ LR-TUFF  │ LR-UC    │ LR-IND   │ LR-CC    │ Basel Zone │")
                    println("├────────────┼──────────┼─────────┼──────────┼──────────┼──────────┼──────────┼────────────┤")
                    
                    for (alpha, result) in sort(collect(var_results), by=x->x[1])
                        pf_pct = round(100 * result.PF, digits=2)
                        basel_str = result.BASEL == 1 ? "Green" : (result.BASEL == 0 ? "Yellow" : "Red")
                        
                        tuff_str = result.TUFF > 0 ? string(result.TUFF) : "N/A"
                        lrtuff_str = isfinite(result.LRTUFF) ? string(round(result.LRTUFF, digits=3)) : "N/A"
                        lruc_str = isfinite(result.LRUC) ? string(round(result.LRUC, digits=3)) : "N/A"
                        lrind_str = isfinite(result.LRIND) ? string(round(result.LRIND, digits=3)) : "N/A"
                        lrcc_str = isfinite(result.LRCC) ? string(round(result.LRCC, digits=3)) : "N/A"
                        
                        println("│ $(rpad(string(round(Int, 100*alpha)) * "%", 10)) │ $(lpad(string(pf_pct), 8)) │ $(lpad(tuff_str, 7)) │ $(lpad(lrtuff_str, 8)) │ $(lpad(lruc_str, 8)) │ $(lpad(lrind_str, 8)) │ $(lpad(lrcc_str, 8)) │ $(lpad(basel_str, 10)) │")
                    end
                    println("└────────────┴──────────┴─────────┴──────────┴──────────┴──────────┴──────────┴────────────┘")
                end
            end
        end
    end
end

function save_var_backtest_to_csv(portfolio_results::Dict, n_assets::Int, nu::Float64, filename_prefix::String="var_backtest_map_tdist")
    all_data = []
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM")]
        if haskey(portfolio_results, model_type)
            result_dict = portfolio_results[model_type]
            if haskey(result_dict, "forecasts")
                forecasts = result_dict["forecasts"]
                if haskey(forecasts, "var_backtest_results")
                    var_results = forecasts["var_backtest_results"]
                    
                    for (alpha, result) in var_results
                        row = Dict(
                            "Model" => name,
                            "N_Assets" => n_assets,
                            "Nu" => nu,
                            "Confidence_Level" => alpha,
                            "PF_Percent" => 100 * result.PF,
                            "TUFF" => result.TUFF,
                            "LR_TUFF" => result.LRTUFF,
                            "LR_UC" => result.LRUC,
                            "LR_IND" => result.LRIND,
                            "LR_CC" => result.LRCC,
                            "Basel_Zone" => result.BASEL == 1 ? "Green" : (result.BASEL == 0 ? "Yellow" : "Red")
                        )
                        push!(all_data, row)
                    end
                end
            end
        end
    end
    
    if !isempty(all_data)
        df = DataFrame(all_data)
        csv_filename = joinpath(OUTPUT_FOLDER, "$(filename_prefix)_$(n_assets)_assets.csv")
        CSV.write(csv_filename, df)
        println("VaR backtesting results saved to: $csv_filename")
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                        MISSING SAVE FUNCTIONS - NOW ADDED
# ══════════════════════════════════════════════════════════════════════════════

function save_model_performance_summary(results::Dict, n_assets::Int, nu::Float64)
    println("Saving model performance summary...")
    all_data = []
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM")]
        if haskey(results, model_type)
            result = results[model_type]
            row = Dict(
                "Model" => name,
                "N_Assets" => n_assets,
                "Nu" => nu,
                "Estimation_Method" => "MAP",
                "Distribution" => "t-Distribution",
                "Training_RMSE" => result["train_rmse"],
                "Validation_RMSE" => result["validation_rmse"],
                "Test_RMSE" => result["test_rmse"],
                "DCC_alpha" => result["dcc_params"][1],
                "DCC_beta" => result["dcc_params"][2]
            )
            push!(all_data, row)
        end
    end
    
    if !isempty(all_data)
        df = DataFrame(all_data)
        csv_filename = joinpath(OUTPUT_FOLDER, "model_performance_summary_$(n_assets)_assets.csv")
        CSV.write(csv_filename, df)
        println("  ✓ Model performance summary saved: $csv_filename")
    end
end

function save_portfolio_weights(portfolio_results::Dict, n_assets::Int)
    println("Saving portfolio weights...")
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM")]
        if haskey(portfolio_results, model_type)
            result_dict = portfolio_results[model_type]
            if haskey(result_dict, "forecasts")
                forecasts = result_dict["forecasts"]
                if haskey(forecasts, "mv_weights")
                    weights = forecasts["mv_weights"]
                    n_days = size(weights, 2)
                    
                    weights_df = DataFrame()
                    for i in 1:n_assets
                        weights_df[!, "Asset_$(i)_Weight"] = weights[i, :]
                    end
                    weights_df[!, "Day"] = 1:n_days
                    weights_df[!, "Model"] = fill(name, n_days)
                    weights_df[!, "Estimation_Method"] = fill("MAP", n_days)
                    weights_df[!, "Distribution"] = fill("t-Distribution", n_days)
                    
                    csv_filename = joinpath(OUTPUT_FOLDER, "mvp_weights_$(model_type)_$(n_assets)_assets.csv")
                    CSV.write(csv_filename, weights_df)
                    println("  ✓ Portfolio weights saved: $csv_filename")
                end
            end
        end
    end
end

function save_portfolio_var_returns(portfolio_results::Dict, n_assets::Int, nu::Float64)
    println("Saving Portfolio VaR vs Returns data...")
    
    for (model_type, result_dict) in portfolio_results
        if haskey(result_dict, "forecasts")
            forecasts = result_dict["forecasts"]
            model_name = result_dict["model_name"]
            model_name_short = model_type == :rnn ? "RNN" : "LSTM"
            
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
                    Nu = fill(nu, n_obs),
                    Estimation_Method = fill("MAP", n_obs),
                    Distribution = fill("t-Distribution", n_obs),
                    Model = fill(model_name, n_obs)
                )
                
                csv_path = joinpath(OUTPUT_FOLDER, "portfolio_var_returns_map_tdist_$(model_name_short)_$(n_assets)_assets.csv")
                CSV.write(csv_path, var_df)
                println("  ✓ Portfolio VaR vs Returns saved: $csv_path")
            else
                println("  Warning: Missing portfolio returns or VaR data for $(model_name)")
            end
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                             MAIN ANALYSIS (MAP VERSION)
# ══════════════════════════════════════════════════════════════════════════════

function comprehensive_dcc_analysis_map(data_path::String, n_assets::Int=30, nu::Float64=5.0)
    println("COMPREHENSIVE DCC-GARCH ANALYSIS (MAP-BASED, t-distribution with ν=$nu)")
    println("Including Portfolio Construction, VaR Calculation & Backtesting")

    !isdir(OUTPUT_FOLDER) && mkdir(OUTPUT_FOLDER)

    train_index = 110:3450
    val_index   = 3451:3700
    test_index  = 3701:3949
    
    df = CSV.read(data_path, DataFrame)
    etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414",
                 "16418", "16421", "16423", "16424", "16426", "16433", "16437",
                 "16452", "16460", "24697", "27635", "28272", "28273", "28274",
                 "28275", "28276", "28277", "28278", "28279", "28280", "31372", "31466"]

    returns, _ = preprocess_financial_data(df, etf_names)

    if n_assets < 30
        returns = returns[:, 1:n_assets]
        selected_assets = etf_names[1:n_assets]
    else
        selected_assets = vcat(etf_names, ["rf"])
    end
    y = Float32.(returns)

    X_train_seq, Y_train_seq = prepare_sequential_data(y, train_index)
    X_val_seq,   Y_val_seq   = prepare_sequential_data(y, val_index)
    rolling_data_seq = prepare_rolling_window_data(y, test_index, 22, :sequential)

    hidden_size = max(16, min(32, 4 * n_assets))
    models_to_run = [
        (:rnn,  X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, rolling_data_seq, "Sequential RNN"),
        (:lstm, X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, rolling_data_seq, "Sequential LSTM")
    ]

    results = Dict{Symbol, Any}()
    portfolio_results = Dict{Symbol, Any}()

    for (model_type, X_train, Y_train, X_val, Y_val, rolling_data, model_name) in models_to_run
        println("TRAINING AND EVALUATING $model_name (MAP, t-dist)")
        
        bnn_train, bnn_val, θmap, _ = train_model_map(
            model_type, X_train, Y_train, X_val, Y_val, n_assets; hidden_size=hidden_size, nu=Float32(nu))

        portfolio_forecasts = rolling_covariance_forecast_map(
            bnn_train, θmap, rolling_data, model_type, n_assets, nu)
        
        portfolio_performance = evaluate_portfolio_performance(portfolio_forecasts, model_name)

        portfolio_results[model_type] = Dict(
            "forecasts" => portfolio_forecasts,
            "performance" => portfolio_performance,
            "model_name" => model_name
        )

        result = evaluate_model_map(bnn_train, bnn_val, portfolio_forecasts, θmap, model_name, nu)
        results[model_type] = result
        
        if haskey(portfolio_forecasts, "portfolio_returns") && haskey(portfolio_forecasts, "var_backtest_results")
            portfolio_returns_vec = Vector{Float64}(portfolio_forecasts["portfolio_returns"])
            daily_vars_vec = [[Float64(portfolio_forecasts["portfolio_var_1"][t]),
                               Float64(portfolio_forecasts["portfolio_var_5"][t]),
                               Float64(portfolio_forecasts["portfolio_var_10"][t])] for t in 1:length(portfolio_returns_vec)]
            
            var_plot = plot_enhanced_var_analysis(portfolio_returns_vec, daily_vars_vec, 
                                                  portfolio_forecasts["var_backtest_results"], selected_assets, model_name)
            
            if var_plot !== nothing
                display(var_plot)
                plot_filename = joinpath(OUTPUT_FOLDER, "map_tdist_var_analysis_$(model_type)_$(n_assets)_assets.png")
                savefig(var_plot, plot_filename)
                println("  VaR analysis plot saved: $plot_filename")
            end
        end
        
        if haskey(portfolio_forecasts, "portfolio_volatility")
            mvp_vol_plot = plot(1:length(portfolio_forecasts["portfolio_volatility"]), 
                                portfolio_forecasts["portfolio_volatility"],
                                title="MVP Volatility Evolution (MAP t-dist) - $model_name ($n_assets Assets)", 
                                xlabel="Day", ylabel="Portfolio Volatility (% points)", 
                                linewidth=2, color=:purple, label="MVP Volatility",
                                size=(1000, 400))
            display(mvp_vol_plot)
            vol_filename = joinpath(OUTPUT_FOLDER, "map_tdist_mvp_evolution_$(model_type)_$(n_assets)_assets.png")
            savefig(mvp_vol_plot, vol_filename)
            println("  MVP evolution plot saved: $vol_filename")
        end
    end

    print_map_results_summary(results, n_assets, nu)
    print_portfolio_comparison(portfolio_results, n_assets) 
    print_var_backtesting_summary(portfolio_results, n_assets)
    
    # Save all results to CSV files
    save_var_backtest_to_csv(portfolio_results, n_assets, nu, "var_backtest_map_tdist")
    save_model_performance_summary(results, n_assets, nu)  # NEW: Added this function
    save_portfolio_weights(portfolio_results, n_assets)     # NEW: Added this function
    save_portfolio_var_returns(portfolio_results, n_assets, nu)  # NEW: Added this function

    return results, portfolio_results
end

# ══════════════════════════════════════════════════════════════════════════════
#                             RUN THE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

data_path = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"

asset_counts = [2, 5, 10, 15, 29, 30]
nu_value = 5.0
all_results_map = Dict{Int64, Tuple{Dict{Symbol, Any}, Dict{Symbol, Any}}}()

for n_assets in asset_counts
    println("RUNNING MAP-BASED ANALYSIS FOR $n_assets ASSETS (t-distribution, ν=$nu_value)")
   
    results, portfolio_results = comprehensive_dcc_analysis_map(data_path, n_assets, nu_value)
    all_results_map[n_assets] = (results, portfolio_results)
    
    println("\nAnalysis complete for $n_assets assets! Results saved and plots generated.")
end

println("ALL MAP-BASED ANALYSES COMPLETE (t-distribution)!")
println("Results generated for asset counts: $(collect(keys(all_results_map)))")
println("All results saved in folder: $OUTPUT_FOLDER")