###############################################################################
#   DCC-GARCH Bayesian Neural Network - Complete Multi-Asset Analysis (Fixed)
#   - Train/Validation/Test + Rolling Window Forecasting
#   - Robust RMSE & Coverage + Portfolio Optimization & VaR
#   - Fixed unicode character, variable scoping, and portfolio metrics issues
###############################################################################

using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Statistics
using MCMCChains, Bijectors
using CSV, DataFrames
using Printf

Random.seed!(1212)

const f0 = 0f0

# ══════════════════════════════════════════════════════════════════════════════
#                              SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

sigmoid(x) = 1.0f0/(1.0f0+exp(-x))

# Enforce 0 < a < 1 and 0 < b < 1-a  ⇒  a + b < 1 (DCC stationarity)
σ(x) = 1f0/(1f0+exp(-x))
function transform_ab(a_raw, b_raw)
    a = σ(a_raw)
    b = (1f0 - a) * σ(b_raw)
    return (a, b)
end

# Simple nearest PD - FIXED
nearest_pd(A) = (A + A')/2 + 1e-4*I

# Data preparation
function prepare_sequential_data(data, indices)
    # Sequential: X[t] -> Y[t+1]
    valid_indices = indices[indices .< size(data, 1)]
    X = data[valid_indices, :]'
    Y = data[valid_indices .+ 1, :]'
    return Float32.(X), Float32.(Y)
end

function prepare_windowed_data(data, indices, window_size)
    n, p = size(data)
    valid_indices = indices[indices .> window_size]
    n_samples = length(valid_indices)
    X = Array{Float32}(undef, p * window_size, n_samples)
    Y = Array{Float32}(undef, p, n_samples)
    for (idx, i) in enumerate(valid_indices)
        X[:, idx] = reshape(data[i-window_size:i-1, :]', :)
        Y[:, idx] = data[i, :]
    end
    return X, Y
end

function prepare_rolling_window_data(data, test_indices, window_size, approach=:sequential)
    rolling_forecasts = []
    for test_idx in test_indices
        if approach == :sequential
            if test_idx > 1
                X_t = Float32.(vec(data[test_idx-1, :]))
                Y_t = Float32.(vec(data[test_idx, :]))
                push!(rolling_forecasts, (X_t, Y_t, test_idx))   # FIXED: push!
            end
        else
            if test_idx > window_size
                X_t = Float32.(reshape(data[test_idx-window_size:test_idx-1, :]', :))
                Y_t = Float32.(vec(data[test_idx, :]))
                push!(rolling_forecasts, (X_t, Y_t, test_idx))
            end
        end
    end
    return rolling_forecasts
end

# Simple preprocessing (note: for rigorous backtesting, compute imputation/scaling on TRAIN only)
function preprocess_financial_data(df, etf_names)
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    if any(ismissing, df[!, "rf"])
        df[!, "rf"] = coalesce.(df[!, "rf"], mean(skipmissing(df[!, "rf"])))
    end
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)
    μ_returns = mean(returns, dims=1)
    return returns, vec(μ_returns)
end

# ══════════════════════════════════════════════════════════════════════════════
#                    COVARIANCE MATRIX AND PORTFOLIO FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Numerically stable min-variance weights (no explicit inverse)
function construct_minimum_variance_portfolio(cov_matrix)
    Σ = Matrix{Float64}(cov_matrix)
    N = size(Σ, 1)
    ones_vec = ones(Float64, N)
    F = cholesky(Symmetric(nearest_pd(Σ) + 1e-9I))
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

struct DCCGarchWindowed{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
end
DCCGarchWindowed(nc::NetConstructor{T,F}, prior::D, N::Int) where {T,F,D<:Distribution} =
    DCCGarchWindowed{T,F,D}(2, nc, prior, N)

function (ℓ::DCCGarchWindowed)(x::Matrix{T}, y::Matrix{T}, θnet::AbstractVector, θlike::AbstractVector) where {T}
    θnet, θlike = T.(θnet), T.(θlike)
    a, b = transform_ab(θlike...)
    net = ℓ.nc(θnet)
    N, Tsteps = ℓ.N, size(x, 2)
    outs = map(t -> net(view(x, :, t)), 1:Tsteps)
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

function build_windowed_model(N::Int, window_size::Int, hidden_layers::Vector{Int})
    layers = Any[]
    input_size = N * window_size
    for (i, h) in enumerate(hidden_layers)
        push!(layers, Dense(i == 1 ? input_size : hidden_layers[i-1], h, relu))
    end
    push!(layers, Dense(hidden_layers[end], 2N))
    return Chain(layers...)
end

# ══════════════════════════════════════════════════════════════════════════════
#                         NAIVE PREDICTION FUNCTION WITH VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

function naive_prediction(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    # Calculate the correct output size: N assets × T time steps × 2 (mean + log-variance)
    N = bnn.like.N
    T_steps = size(x, 2)
    output_size = 2 * N * T_steps  # Full network output size
    
    yhats = Array{T, 2}(undef, output_size, size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        yh = vec(net(x))
        
        # Validation: Check if predictions are finite
        if all(isfinite, yh)
            yhats[:, i] = yh
        else
            # Use zeros for invalid predictions
            yhats[:, i] = zeros(T, output_size)
        end
    end
    return yhats
end

# ══════════════════════════════════════════════════════════════════════════════
#                         TRAINING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

function train_model_with_validation(model_type::Symbol, X_train, Y_train, X_val, Y_val, N::Int; 
        hidden_size::Int=32, window_size::Int=22, prior_std::Float32=0.15f0,
        mcmc_samples::Int=25000, chains::Int=4)

    println("Training $model_type model with $N assets...")
    if model_type == :rnn
        net = build_rnn_model(N, hidden_size)
        likelihood_type = DCCGarchRNNSequential
    elseif model_type == :lstm
        net = build_lstm_model(N, hidden_size)
        likelihood_type = DCCGarchLSTMSequential
    elseif model_type == :windowed
        net = build_windowed_model(N, window_size, [64, 32, 16])
        likelihood_type = DCCGarchWindowed
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
    _ = find_mode(bnn_train, 50, 1000, FluxModeFinder(bnn_train, Flux.ADAM()))

    println("Starting MCMC sampling...")
    step_size = N <= 5 ? 1f-4 : 2f-5
    sampler = SGNHTS(step_size, 1f0; xi=1f0^2, μ=10f0)
    ch_raw = mcmc(bnn_train, chains, mcmc_samples, sampler)

    burn_in = mcmc_samples ÷ 2
    ch_burn_raw, ch2d = apply_burnin_and_flatten(ch_raw, burn_in)

    # Validation log-likelihood on subset
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
    
    # Validation filter: Only use finite log-likelihoods
    finite_ll = filter(isfinite, val_ll)
    return length(finite_ll) > 0 ? mean(finite_ll) : -Inf
end

# ══════════════════════════════════════════════════════════════════════════════
#                         SIMPLIFIED CHAIN HELPERS 
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

# Updated R-hat computation using BayesFlux.jl approach with validation
function compute_rhat_bayesflux(bnn, ch)
    # Extract only network parameters (excluding likelihood parameters)
    net_params = ch[1:end-2, :]
    yhats = naive_prediction(bnn, net_params)
    
    # Validation: Check if predictions are valid before computing R-hat
    if any(!isfinite, yhats)
        println("Warning: Non-finite predictions detected in R-hat computation")
        # Filter out non-finite columns
        finite_cols = [all(isfinite, yhats[:, i]) for i in 1:size(yhats, 2)]
        if sum(finite_cols) < 10  # Need minimum samples
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
#                        FORWARD PASSES & SINGLE-STEP FORECASTS
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
#                      DCC STATE FROM TRAINING (Q̄, Q_T, z_T)
# ══════════════════════════════════════════════════════════════════════════════

function compute_dcc_training_state(bnn_train, chain2d)
    N = bnn_train.like.N
    θnet_mean = vec(mean(get_θnet(chain2d); dims=2))
    θlike_mean = vec(mean(get_θlike(chain2d); dims=2))
    a, b = transform_ab(θlike_mean...)
    net = bnn_train.like.nc(θnet_mean)
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
#                      ENHANCED ROLLING PORTFOLIO FORECASTING WITH VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

function rolling_covariance_forecast_with_portfolio_analysis(bnn_train, chain2d, rolling_data, model_type::Symbol, N::Int)
    println("Performing rolling window forecasting with portfolio analysis...")
    n_test = length(rolling_data)
    n_draws = min(size(chain2d, 2), 100)
    idxs = 1:n_draws

    # Precompute training DCC state
    Qbar, Q_prev, z_prev, a_mean, b_mean = compute_dcc_training_state(bnn_train, chain2d)

    forecasts = Dict(
        "means" => Array{Float32}(undef, N, n_test),
        "covariance_matrices" => Array{Float32}(undef, N, N, n_test),
        "mv_weights" => Array{Float32}(undef, N, n_test),
        "portfolio_returns" => Array{Float32}(undef, n_test),  # realized
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

        # Posterior mean output for current step with validation
        μ_acc = zeros(Float32, N)
        logσ2_acc = zeros(Float32, N)
        valid = 0
        
        for j in idxs
            θnet = chain2d[1:end-2, j]
            θlike = chain2d[end-1:end, j]
            net = bnn_train.like.nc(θnet)
            out = model_type == :rnn ? rnn_single_forecast(net, X_t, N) :
                  model_type == :lstm ? lstm_single_forecast(net, X_t, N) : net(X_t)
            
            # Validation: Check if predictions are finite
            if all(isfinite, out) && length(out) >= 2*N
                μ_acc .+= out[1:N]
                logσ2_acc .+= out[N+1:2N]
                valid += 1
            end
        end
        
        # Use valid predictions or fallback to zeros
        if valid > 0
            μ_pred = μ_acc ./ valid
            logσ2_pred = clamp.(logσ2_acc ./ valid, -10f0, 5f0)
        else
            μ_pred = zeros(Float32, N)
            logσ2_pred = zeros(Float32, N)
        end

        # DCC one-step covariance using (Q_prev, z_prev)
        H, Q_next = one_step_dcc_covariance(μ_pred, logσ2_pred, a_mean, b_mean, Q_prev, z_prev, Qbar)

        # Portfolio construction with validation
        w = ones(Float32, N) ./ N  # Initialize w in proper scope
        try
            w = construct_minimum_variance_portfolio(H)
            # Validate portfolio weights
            if !all(isfinite, w)
                w = ones(Float32, N) ./ N  # Equal weights fallback
            end
        catch
            w = ones(Float32, N) ./ N  # Equal weights fallback
        end

        σ = exp.(logσ2_pred ./ 2f0)
        R = H  # already covariance
        
        # Portfolio metrics with validation - FIXED SCOPING
        var_results = Dict(0.01=>0.0, 0.05=>0.0, 0.10=>0.0)  # Initialize in proper scope
        σp = 0.0f0  # Initialize in proper scope
        μp = 0.0f0  # Initialize in proper scope
        try
            var_results, μp, σp = calculate_portfolio_var(w, μ_pred, R)
            if !isfinite(σp)
                σp = 0.0f0
            end
            if any(!isfinite, values(var_results))
                var_results = Dict(0.01=>0.0, 0.05=>0.0, 0.10=>0.0)
            end
        catch
            var_results = Dict(0.01=>0.0, 0.05=>0.0, 0.10=>0.0)
            σp = 0.0f0
        end

        # Store results
        forecasts["means"][:, t_idx] = μ_pred
        forecasts["covariance_matrices"][:, :, t_idx] = Float32.(R)
        forecasts["mv_weights"][:, t_idx] = w
        forecasts["portfolio_volatility"][t_idx] = Float32(σp)
        forecasts["portfolio_var_1"][t_idx] = Float32(var_results[0.01])
        forecasts["portfolio_var_5"][t_idx] = Float32(var_results[0.05])
        forecasts["portfolio_var_10"][t_idx] = Float32(var_results[0.10])
        # Realized portfolio return (use actual Y_t)
        forecasts["portfolio_returns"][t_idx] = Float32(w' * Y_t)

        # Update DCC state for next step using realized standardized residual
        σ_step = σ .+ 1f-6
        z_prev = (Y_t .- μ_pred) ./ σ_step
        Q_prev = Q_next
    end

    return forecasts
end

# Original probabilistic rolling forecast with validation (restored)
function rolling_window_forecast(bnn_train, chain2d, rolling_data, model_type::Symbol, N::Int)
    println("Performing rolling window forecasting (stochastic samples)...")
    n_test = length(rolling_data)
    n_samples = min(size(chain2d, 2), 200)
    forecasts = Dict(
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
            
            # Validation: Check if all outputs are finite
            if all(isfinite, mean_pred) && all(isfinite, var_pred) && all(isfinite, sample_pred) && isfinite(loglik)
                forecast_samples[:, s_idx] = sample_pred
                logliks[s_idx] = loglik
            else
                # Use fallback values for invalid predictions
                forecast_samples[:, s_idx] = zeros(Float32, N)
                logliks[s_idx] = -Inf32
            end
        end
        
        # Sample validation: Filter valid samples
        valid_mask = [all(isfinite, forecast_samples[:, i]) for i in 1:n_samples]
        valid_samples_filtered = forecast_samples[:, valid_mask]
        
        if size(valid_samples_filtered, 2) > 0
            forecasts["means"][:, t_idx] = mean(valid_samples_filtered, dims=2)[:, 1]
            forecasts["vars"][:, t_idx] = var(valid_samples_filtered, dims=2)[:, 1]
        else
            # Fallback to zeros if no valid samples
            forecasts["means"][:, t_idx] = zeros(Float32, N)
            forecasts["vars"][:, t_idx] = zeros(Float32, N)
        end
        
        forecasts["samples"][:, t_idx, :] = forecast_samples
        
        # Likelihood filtering: Only use finite log-likelihoods
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
    
    # Diagonal covariance for robustness
    H = Diagonal((σ_pred .^ 2)) + 1e-6f0*I
    sample_pred = zeros(Float32, N)
    loglik = 0f0
    
    try
        L = cholesky(Symmetric(H)).L
        sample_pred = μ_pred + L * randn(Float32, N)
        diff = Y_t - μ_pred
        quad = sum(abs2, L \ diff)
        loglik = -0.5f0 * (N * log(2π) + 2 * sum(log, diag(L)) + quad)
    catch
        # Fallback for invalid covariance
        sample_pred = μ_pred + σ_pred .* randn(Float32, N)
        diff = Y_t - μ_pred
        quad = sum(abs2, diff ./ σ_pred)
        loglik = -0.5f0 * (N * log(2π) + 2 * sum(log, σ_pred) + quad)
    end
    
    return μ_pred, σ_pred .^ 2, sample_pred, loglik
end

# ══════════════════════════════════════════════════════════════════════════════
#                              EVALUATION METRICS WITH VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

function compute_coverage_probability(y_true, y_samples, confidence_level)
    α = 1 - confidence_level
    lower_q, upper_q = α/2, 1 - α/2
    N, n_test, n_samp = size(y_samples)
    total, inside = 0, 0
    
    for i in 1:N, t in 1:n_test
        samples_it = y_samples[i, t, :]
        valid_samples = samples_it[isfinite.(samples_it)]
        
        # Coverage validation: Need minimum samples and finite true value
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
            
            # Quantile validation: Need minimum samples and finite true value
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
        # Prediction validation: Check for finite values
        if all(isfinite, pred_vec)
            yhats[:, i] = pred_vec
        else
            # Use zeros for invalid predictions
            yhats[:, i] = zeros(Float32, N * n_obs)
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
            # Quantile validation: Skip non-finite true values
            if !isfinite(val)
                continue
            end
            
            point_samples = y_samples_trim[j, :]
            valid = point_samples[isfinite.(point_samples)]
            
            # Need minimum valid samples
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
    
    # Validation: Filter finite values for RMSE
    valid_mask = isfinite.(y_true_flat) .& isfinite.(pred_mean)
    if sum(valid_mask) > 0
        rmse = sqrt(mean((y_true_flat[valid_mask] .- pred_mean[valid_mask]).^2))
    else
        rmse = Inf
    end
    
    # Validation: Filter finite log-likelihoods
    logliks = [
        bnn.like(bnn.x, bnn.y, chain2d[1:end-2, i], chain2d[end-1:end, i])
        for i in 1:min(100, size(chain2d, 2))
    ]
    finite_logliks = filter(isfinite, logliks)
    avg_loglik = length(finite_logliks) > 0 ? mean(finite_logliks) : -Inf
    
    # Enhanced evaluation with quantiles
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
        # Predictive validation: Check for finite values
        if all(isfinite, sample_vec)
            pred_samples[:, i] = sample_vec
        else
            # Use zeros for invalid samples
            pred_samples[:, i] = zeros(Float32, N * n_obs)
        end
    end
    
    return pred_samples
end

function evaluate_out_of_sample(forecasts, set_name::String)
    y_true = forecasts["actuals"]
    y_pred_mean = forecasts["means"]
    y_pred_var = haskey(forecasts, "vars") ? forecasts["vars"] : similar(y_pred_mean)
    logliks = haskey(forecasts, "loglik") ? forecasts["loglik"] : fill(Float32(NaN), size(y_true, 2))
    
    # Validation: Filter finite values for metrics
    valid_mask = isfinite.(y_true) .& isfinite.(y_pred_mean)
    rmse = sqrt(mean((y_true[valid_mask] .- y_pred_mean[valid_mask]).^2))
    
    # Likelihood validation
    finite_logliks = logliks[isfinite.(logliks)]
    avg_loglik = length(finite_logliks) > 0 ? mean(finite_logliks) : NaN
    
    # Coverage and quantile validation
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
    
    # Portfolio validation: Filter finite values
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
    println("  Avg Realized Return: $(round(avg_portfolio_return * 100, digits=4))%")
    println("  Avg Predicted Volatility: $(round(portfolio_volatility * 100, digits=4))%")
    println("  VaR (avg across horizon): 1%=$(round(avg_var_1*100, digits=4))%, 5%=$(round(avg_var_5*100, digits=4))%, 10%=$(round(avg_var_10*100, digits=4))%")
    
    return Dict(
        "avg_portfolio_return" => avg_portfolio_return,
        "portfolio_volatility" => portfolio_volatility,
        "avg_var_1" => avg_var_1,
        "avg_var_5" => avg_var_5,
        "avg_var_10" => avg_var_10
    )
end

function evaluate_model_comprehensive(bnn_train, bnn_val, chain2d, ch_raw, rolling_forecasts, model_name::String)
    N = bnn_train.like.N
    train_results = evaluate_in_sample(bnn_train, chain2d, "Training")
    val_results   = evaluate_in_sample(bnn_val,   chain2d, "Validation")
    test_results  = evaluate_out_of_sample(rolling_forecasts, "Test")
    
    # Use BayesFlux.jl approach for R-hat computation with validation
    r_hat = compute_rhat_bayesflux(bnn_train, ch_raw)
    
    # DCC params posterior mean
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

# Pretty printing
function print_portfolio_comparison(portfolio_results, n_assets)
    println("\n" * "="^80)
    println("PORTFOLIO PERFORMANCE COMPARISON ($n_assets Assets)")
    println("="^80)
    println("\n┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    println("│ Model               │ Avg Return% │ Volatility% │ VaR 1%      │ VaR 5%      │ VaR 10%     │")
    println("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM"), (:windowed, "Windowed FF")]
        if haskey(portfolio_results, model_type)
            perf = portfolio_results[model_type]["performance"]
            avg = lpad(string(round(perf["avg_portfolio_return"] * 100, digits=4)), 11)
            vol = lpad(string(round(perf["portfolio_volatility"] * 100, digits=4)), 11)
            v1  = lpad(string(round(perf["avg_var_1"] * 100, digits=4)), 11)
            v5  = lpad(string(round(perf["avg_var_5"] * 100, digits=4)), 11)
            v10 = lpad(string(round(perf["avg_var_10"] * 100, digits=4)), 11)
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
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM"), (:windowed, "Windowed FF")]
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

# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

function comprehensive_dcc_analysis_with_portfolio(data_path::String, n_assets::Int=30)
    println("COMPREHENSIVE DCC-GARCH ANALYSIS WITH PORTFOLIO OPTIMIZATION")
    println("Including Minimum Variance Portfolio Construction and VaR Calculation")

    train_index = 110:3450
    val_index   = 3451:3700
    test_index  = 3701:3950

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
        selected_assets = vcat(etf_names, ["rf"])  # include rf as last column
    end

    y = Float32.(returns)

    println("Selected assets: $(length(selected_assets)) total")
    println("Data dimensions: $(size(y)) (observations × assets)")

    # Prepare data
    X_train_seq, Y_train_seq = prepare_sequential_data(y, train_index)
    X_val_seq,   Y_val_seq   = prepare_sequential_data(y, val_index)
    window_size = 22
    X_train_win, Y_train_win = prepare_windowed_data(y, train_index, window_size)
    X_val_win,   Y_val_win   = prepare_windowed_data(y, val_index,   window_size)
    rolling_data_seq = prepare_rolling_window_data(y, test_index, window_size, :sequential)
    rolling_data_win = prepare_rolling_window_data(y, test_index, window_size, :windowed)

    println("Training data prepared:")
    println("  Sequential: X=$(size(X_train_seq)), Y=$(size(Y_train_seq))")
    println("  Windowed:   X=$(size(X_train_win)), Y=$(size(Y_train_win))")
    println("  Rolling forecasts: $(length(rolling_data_seq)) time points")

    hidden_size = max(16, min(64, 8 * n_assets))
    models_to_run = [
        (:rnn,     X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, rolling_data_seq, "Sequential RNN"),
        (:lstm,    X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, rolling_data_seq, "Sequential LSTM"),
        (:windowed, X_train_win, Y_train_win, X_val_win, Y_val_win, rolling_data_win, "Windowed Feedforward")
    ]

    results = Dict{Symbol, Any}()
    portfolio_results = Dict{Symbol, Any}()

    for (model_type, X_train, Y_train, X_val, Y_val, rolling_data, model_name) in models_to_run
        println("\n" * "="^60)
        println("TRAINING AND EVALUATING $model_name WITH PORTFOLIO ANALYSIS")
        println("="^60)
        bnn_train, bnn_val, chain2d, ch_raw, net, val_loglik = train_model_with_validation(
            model_type, X_train, Y_train, X_val, Y_val, n_assets; hidden_size=hidden_size, mcmc_samples=15000, chains=4)
        println("Training complete. Validation log-likelihood: $(round(val_loglik, digits=4))")

        # Portfolio-aware rolling forecast
        portfolio_forecasts = rolling_covariance_forecast_with_portfolio_analysis(
            bnn_train, chain2d, rolling_data, model_type, n_assets)
        portfolio_performance = evaluate_portfolio_performance(portfolio_forecasts, model_name)

        portfolio_results[model_type] = Dict(
            "forecasts" => portfolio_forecasts,
            "performance" => portfolio_performance,
            "model_name" => model_name
        )

        # Stochastic predictive evaluation
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
    end

    print_comprehensive_results_summary(results, n_assets)
    print_portfolio_comparison(portfolio_results, n_assets)

    return results, portfolio_results
end

# ══════════════════════════════════════════════════════════════════════════════
#                              RUN THE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

data_path = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"

results, portfolio_results = comprehensive_dcc_analysis_with_portfolio(data_path, 2)

if !isempty(portfolio_results)
    portfolio_summary = []
    for (model_type, data) in portfolio_results
        perf = data["performance"]
        push!(portfolio_summary, Dict(
            "model" => data["model_name"],
            "avg_portfolio_return" => perf["avg_portfolio_return"],
            "portfolio_volatility" => perf["portfolio_volatility"],
            "avg_var_1" => perf["avg_var_1"],
            "avg_var_5" => perf["avg_var_5"],
            "avg_var_10" => perf["avg_var_10"]
        ))
    end
    portfolio_df = DataFrame(portfolio_summary)
    CSV.write("portfolio_analysis_results.csv", portfolio_df)
    println("\nPortfolio analysis results saved to: portfolio_analysis_results.csv")
end