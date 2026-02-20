###############################################################################
#   DCC-GARCH Bayesian Neural Network - Complete Multi-Asset Analysis
#  With Train/Validation/Test Splits and Rolling Window Forecasting
#  RMSE-Only Evaluation
#Basel_Zone 0 -> Yellow (Acceptable), -1→ Red (Too many violations), 1→ Green (Good model)
###############################################################################

using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames
using Printf

Random.seed!(1212)

# ══════════════════════════════════════════════════════════════════════════════
#                              SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Mathematical helpers
sigmoid(x) = 1/(1+exp(-x))
transform_ab(a,b) = let a_=sigmoid(a); b_=sigmoid(b)*(1-a_); (a_,b_) end
nearest_pd(A) = (A + A')/2 + 1e-4I

# Data preparation functions
function prepare_sequential_data(data, indices)
    """Sequential data: X[t] -> Y[t+1] for given indices"""
    # Ensure we don't go out of bounds
    valid_indices = indices[indices .< size(data, 1)]
    X = data[valid_indices, :]'
    Y = data[valid_indices .+ 1, :]'
    return Float32.(X), Float32.(Y)
end

function prepare_windowed_data(data, indices, window_size)
    """Windowed data: X[t-w:t-1] -> Y[t] for given indices"""
    n, p = size(data)
    # Ensure indices allow for window_size lookback
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
    """Prepare data for rolling window forecasting"""
    rolling_forecasts = []
    
    for test_idx in test_indices
        if approach == :sequential
            # For sequential: use previous observation to predict current
            if test_idx > 1
                X_t = Float32.(data[test_idx-1, :]')
                Y_t = Float32.(data[test_idx, :]')
                push!(rolling_forecasts, (X_t, Y_t, test_idx))
            end
        else # windowed
            # For windowed: use window_size previous observations
            if test_idx > window_size
                X_t = Float32.(reshape(data[test_idx-window_size:test_idx-1, :]', :))
                Y_t = Float32.(data[test_idx, :]')
                push!(rolling_forecasts, (X_t, Y_t, test_idx))
            end
        end
    end
    
    return rolling_forecasts
end

# Data preprocessing
function preprocess_financial_data(df, etf_names)
    """Handle missing values and extract returns including risk-free rate"""
    # Handle missing values for ETFs
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    # Handle missing values for risk-free rate
    if any(ismissing, df[!, "rf"])
        df[!, "rf"] = coalesce.(df[!, "rf"], mean(skipmissing(df[!, "rf"])))
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)  # 30 assets total
    
    μ_returns = mean(returns, dims=1)
    return returns, vec(μ_returns)
end

# ══════════════════════════════════════════════════════════════════════════════
#                              LIKELIHOOD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Sequential RNN DCC-GARCH Likelihood
struct DCCGarchRNNSequential{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
end

DCCGarchRNNSequential(nc::NetConstructor{T,F}, prior::D, N::Int) where {T,F,D<:Distribution} =
    DCCGarchRNNSequential{T,F,D}(2, nc, prior, N)

function (ℓ::DCCGarchRNNSequential)(x::Matrix{T}, y::Matrix{T},
                                   θnet::AbstractVector, θlike::AbstractVector) where {T}
    θnet, θlike = T.(θnet), T.(θlike)
    a, b = transform_ab(θlike...)
    net = ℓ.nc(θnet)
    N, Tsteps = ℓ.N, size(x, 2)
    
    # RNN forward pass
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

# Sequential LSTM DCC-GARCH Likelihood
struct DCCGarchLSTMSequential{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
end

DCCGarchLSTMSequential(nc::NetConstructor{T,F}, prior::D, N::Int) where {T,F,D<:Distribution} =
    DCCGarchLSTMSequential{T,F,D}(2, nc, prior, N)

function (ℓ::DCCGarchLSTMSequential)(x::Matrix{T}, y::Matrix{T},
                                    θnet::AbstractVector, θlike::AbstractVector) where {T}
    θnet, θlike = T.(θnet), T.(θlike)
    a, b = transform_ab(θlike...)
    net = ℓ.nc(θnet)
    N, Tsteps = ℓ.N, size(x, 2)
    
    # LSTM forward pass
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

# Windowed Feedforward DCC-GARCH Likelihood
struct DCCGarchWindowed{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
end

DCCGarchWindowed(nc::NetConstructor{T,F}, prior::D, N::Int) where {T,F,D<:Distribution} =
    DCCGarchWindowed{T,F,D}(2, nc, prior, N)

function (ℓ::DCCGarchWindowed)(x::Matrix{T}, y::Matrix{T},
                              θnet::AbstractVector, θlike::AbstractVector) where {T}
    θnet, θlike = T.(θnet), T.(θlike)
    a, b = transform_ab(θlike...)
    net = ℓ.nc(θnet)
    N, Tsteps = ℓ.N, size(x, 2)
    
    outs = map(t -> net(view(x, :, t)), 1:Tsteps)
    return compute_dcc_likelihood(outs, y, a, b, N, Tsteps, ℓ.prior)
end

# Shared DCC likelihood computation
function compute_dcc_likelihood(outs, y, a, b, N, Tsteps, prior)
    T = eltype(y)
    
    μ = hcat(map(o -> o[1:N], outs)...)
    logσ2 = hcat(map(o -> o[N+1:2N], outs)...)
    σ = exp.(logσ2 ./ 2)
    z = (y .- μ) ./ σ

    # Initialize correlation matrix
    Q̄ = Tsteps > 1 ? (z * z') / Tsteps : Matrix{T}(I, N, N)
    d = 1 ./ sqrt.(diag(Q̄))
    Q̄ = Symmetric(diagm(d) * Q̄ * diagm(d))

    Q, logl = Q̄, zero(T)
    for t in 1:Tsteps
        d = 1 ./ sqrt.(max.(diag(Q), 1e-10))
        R = Symmetric(diagm(d) * Q * diagm(d))
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
    logl += sum(logpdf.(prior, vec(μ)))
    return logl
end

# ══════════════════════════════════════════════════════════════════════════════
#                              MODEL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

function build_rnn_model(N::Int, hidden_size::Int)
    """Build Sequential RNN DCC-GARCH model"""
    return Chain(
        RNN(N => hidden_size),
        Dense(hidden_size => 2N)
    )
end

function build_lstm_model(N::Int, hidden_size::Int)
    """Build Sequential LSTM DCC-GARCH model"""
    return Chain(
        LSTM(N => hidden_size),
        Dense(hidden_size => 2N)
    )
end

function build_windowed_model(N::Int, window_size::Int, hidden_layers::Vector{Int})
    """Build Windowed Feedforward DCC-GARCH model"""
    layers = []
    input_size = N * window_size
    
    for (i, hidden_size) in enumerate(hidden_layers)
        if i == 1
            push!(layers, Dense(input_size, hidden_size, relu))
        else
            push!(layers, Dense(hidden_layers[i-1], hidden_size, relu))
        end
    end
    
    push!(layers, Dense(hidden_layers[end], 2N))
    return Chain(layers...)
end

# ══════════════════════════════════════════════════════════════════════════════
#                              TRAINING AND VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

function train_model_with_validation(model_type::Symbol, X_train, Y_train, X_val, Y_val, N::Int; 
                                   hidden_size::Int=32, window_size::Int=22,
                                   prior_std::Float32=0.15f0, mcmc_samples::Int=25000,
                                   chains::Int=15)
    
    println("Training $model_type model with $N assets...")
    
    # Build appropriate network
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
    
    # Setup BNN components
    nc = destruct(net)
    like = likelihood_type(nc, Normal(0, prior_std), N)
    prior = GaussianPrior(nc, prior_std)
    init = InitialiseAllSame(Normal(0f0, prior_std), like, prior)
    
    # Training BNN
    bnn_train = BNN(X_train, Y_train, like, prior, init)
    
    # Validation BNN (same parameters, different data)
    bnn_val = BNN(X_val, Y_val, like, prior, init)
    
    # Find MAP estimate
    println("Finding MAP estimate...")
    θmap = find_mode(bnn_train, 50, 1000, FluxModeFinder(bnn_train, Flux.ADAM()))
    
    # MCMC sampling
    println("Starting MCMC sampling...")
    step_size = N <= 5 ? 1f-4 : 2f-5
    sampler = SGNHTS(step_size, 1f0; xi=1f0^2, μ=10f0)
    ch = mcmc(bnn_train, chains, mcmc_samples, sampler)
    
    # Keep latter half of samples
    burn_in = mcmc_samples ÷ 2
    ch = ch[:, end-burn_in+1:end]
    
    # Compute validation likelihood
    val_loglik = compute_validation_likelihood(bnn_val, ch)
    
    return bnn_train, bnn_val, ch, net, val_loglik
end

function compute_validation_likelihood(bnn_val, chain)
    """Compute average validation log-likelihood"""
    n_samples = min(size(chain, 2), 100)  # Use subset for efficiency
    sample_indices = rand(1:size(chain, 2), n_samples)
    
    val_logliks = map(sample_indices) do i
        θnet = chain[1:end-2, i]
        θlike = chain[end-1:end, i]
        try
            bnn_val.like(bnn_val.x, bnn_val.y, θnet, θlike)
        catch
            -Inf
        end
    end
    
    # Remove infinite values and compute mean
    finite_logliks = filter(isfinite, val_logliks)
    return isempty(finite_logliks) ? -Inf : mean(finite_logliks)
end

# ══════════════════════════════════════════════════════════════════════════════
#                              ROLLING WINDOW FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

function rolling_window_forecast(bnn_train, chain, rolling_data, model_type::Symbol, N::Int)
    """Perform rolling window forecasting on test data"""
    
    println("Performing rolling window forecasting...")
    n_test = length(rolling_data)
    n_samples = min(size(chain, 2), 200)  # Use subset for efficiency
    
    # Storage for forecasts
    forecasts = Dict(
        "means" => Array{Float32}(undef, N, n_test),
        "vars" => Array{Float32}(undef, N, n_test),
        "samples" => Array{Float32}(undef, N, n_test, n_samples),
        "actuals" => Array{Float32}(undef, N, n_test),
        "loglik" => Array{Float32}(undef, n_test)
    )
    
    sample_indices = rand(1:size(chain, 2), n_samples)
    
    for (t_idx, (X_t, Y_t, original_idx)) in enumerate(rolling_data)
        if t_idx % 50 == 0
            println("  Processing forecast $t_idx/$n_test")
        end
        
        # Store actual values
        forecasts["actuals"][:, t_idx] = Y_t
        
        # Generate forecasts for this time point
        forecast_samples = Array{Float32}(undef, N, n_samples)
        logliks = Array{Float32}(undef, n_samples)
        
        for (s_idx, chain_idx) in enumerate(sample_indices)
            θnet = chain[1:end-2, chain_idx]
            θlike = chain[end-1:end, chain_idx]
            
            try
                # Single point forecast
                mean_pred, var_pred, sample_pred, loglik = single_point_forecast(
                    θnet, θlike, X_t, Y_t, bnn_train.like, model_type, N
                )
                
                forecast_samples[:, s_idx] = sample_pred
                logliks[s_idx] = loglik
                
            catch e
                forecast_samples[:, s_idx] .= NaN
                logliks[s_idx] = -Inf
            end
        end
        
        # Compute summary statistics
        forecasts["means"][:, t_idx] = mean(forecast_samples, dims=2)[:, 1]
        forecasts["vars"][:, t_idx] = var(forecast_samples, dims=2)[:, 1]
        forecasts["samples"][:, t_idx, :] = forecast_samples
        forecasts["loglik"][t_idx] = mean(filter(isfinite, logliks))
    end
    
    return forecasts
end

function single_point_forecast(θnet, θlike, X_t, Y_t, likelihood, model_type::Symbol, N::Int)
    """Generate single point forecast"""
    
    # Get network prediction
    net = likelihood.nc(θnet)
    
    if model_type == :rnn
        output = rnn_single_forecast(net, X_t, N)
    elseif model_type == :lstm
        output = lstm_single_forecast(net, X_t, N)
    else  # windowed
        output = net(X_t)
    end
    
    μ_pred = output[1:N]
    logσ2_pred = output[N+1:2N]
    σ_pred = exp.(logσ2_pred ./ 2)
    
    # Transform DCC parameters
    a, b = transform_ab(θlike...)
    
    # For simplicity in single point forecast, use diagonal covariance
    # In practice, you might maintain DCC state across rolling windows
    H = Diagonal(σ_pred .^ 2) + 1e-6*I
    
    # Generate sample
    sample_pred = μ_pred + cholesky(Symmetric(H)).L * randn(Float32, N)
    
    # Compute log-likelihood
    diff = Y_t - μ_pred
    L = cholesky(Symmetric(H)).L
    quad = sum(abs2, L \ diff)
    loglik = -0.5 * (N * log(2π) + 2 * sum(log, diag(L)) + quad)
    
    return μ_pred, diag(H), sample_pred, loglik
end

function rnn_single_forecast(net, X_t, N)
    """Single RNN forecast"""
    rnn_layer, dense_layer = net[1], net[2]
    hidden_size = size(rnn_layer.cell.Wi, 1)
    h = zeros(Float32, hidden_size)
    
    # Single step forward
    h = tanh.(rnn_layer.cell.Wi * X_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
    return dense_layer(h)
end

function lstm_single_forecast(net, X_t, N)
    """Single LSTM forecast"""
    lstm_layer, dense_layer = net[1], net[2]
    hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
    h, c = zeros(Float32, hidden_size), zeros(Float32, hidden_size)
    
    # Single step forward
    gates = lstm_layer.cell.Wi * X_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
    
    i_gate = sigmoid.(gates[1:hidden_size])
    f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
    g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
    o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
    
    c = f_gate .* c .+ i_gate .* g_gate
    h = o_gate .* tanh.(c)
    
    return dense_layer(h)
end

# ══════════════════════════════════════════════════════════════════════════════
#                              EVALUATION METRICS (RMSE ONLY)
# ══════════════════════════════════════════════════════════════════════════════

function evaluate_model_comprehensive(bnn_train, bnn_val, chain, rolling_forecasts, model_name::String)
    """Comprehensive model evaluation including train/val/test performance - RMSE only"""
    N = bnn_train.like.N
    
    # Training set evaluation
    train_results = evaluate_in_sample(bnn_train, chain, "Training")
    
    # Validation set evaluation  
    val_results = evaluate_in_sample(bnn_val, chain, "Validation")
    
    # Test set evaluation (rolling forecasts)
    test_results = evaluate_out_of_sample(rolling_forecasts, "Test")
    
    # Overall convergence diagnostics
    yhats = generate_predictions(bnn_train, chain)
    chain_yhat = Chains(yhats')
    r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
    
    # DCC parameters
    dcc_params = transform_ab(mean(chain[end-1:end, :], dims=2)...)
    
    results = Dict(
        "model_name" => model_name,
        "n_assets" => N,
        "r_hat" => r_hat,
        "dcc_params" => dcc_params,
        "converged" => r_hat < 1.2,
        "train" => train_results,
        "validation" => val_results,
        "test" => test_results
    )
    
    return results
end

function evaluate_in_sample(bnn, chain, set_name::String)
    """Evaluate in-sample performance (train/validation) - RMSE only"""
    N = bnn.like.N
    
    # Generate predictions
    yhats = generate_predictions(bnn, chain)
    pred_mean = mean(yhats, dims=2)[:, 1]
    pred_var = var(yhats, dims=2)[:, 1]
    
    # Reshape for comparison
    n_obs = size(bnn.y, 2)
    y_true_flat = vec(bnn.y)
    pred_mean_reshaped = reshape(pred_mean, N, n_obs)
    
    # Compute RMSE only
    rmse = sqrt(mean((y_true_flat - pred_mean).^2))
    
    # Log-likelihood
    avg_loglik = mean([
        try
            θnet = chain[1:end-2, i]
            θlike = chain[end-1:end, i]
            bnn.like(bnn.x, bnn.y, θnet, θlike)
        catch
            -Inf
        end
        for i in 1:min(100, size(chain, 2))
    ])
    
    # QQ evaluation
    posterior_samples = generate_posterior_predictive(bnn, chain, bnn.x, bnn.y)
    t_q = 0.05:0.05:0.95
    o_q = compute_observed_quantiles(bnn.y', posterior_samples, t_q, N)
    mad_qq = mean(abs.(o_q .- t_q))
    
    return Dict(
        "set_name" => set_name,
        "rmse" => rmse,
        "loglik" => avg_loglik,
        "mad_qq" => mad_qq,
        "qq_observed" => o_q,
        "qq_target" => t_q
    )
end

function evaluate_out_of_sample(forecasts, set_name::String)
    """Evaluate out-of-sample performance (test set rolling forecasts) - RMSE only"""
    
    # Extract data
    y_true = forecasts["actuals"]
    y_pred_mean = forecasts["means"]
    y_pred_var = forecasts["vars"]
    logliks = forecasts["loglik"]
    
    # Compute RMSE only
    rmse = sqrt(mean((y_true - y_pred_mean).^2))
    avg_loglik = mean(filter(isfinite, logliks))
    
    # Coverage probabilities for prediction intervals
    coverage_90 = compute_coverage_probability(y_true, forecasts["samples"], 0.90)
    coverage_95 = compute_coverage_probability(y_true, forecasts["samples"], 0.95)
    
    # QQ evaluation using forecast samples
    t_q = 0.05:0.05:0.95
    o_q = compute_forecast_quantiles(y_true, forecasts["samples"], t_q)
    mad_qq = mean(abs.(o_q .- t_q))
    
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

function compute_coverage_probability(y_true, y_samples, confidence_level)
    """Compute empirical coverage probability for prediction intervals"""
    α = 1 - confidence_level
    lower_q = α / 2
    upper_q = 1 - α / 2
    
    N, n_test, n_samples = size(y_samples)
    total_points = N * n_test
    
    in_interval = 0
    for i in 1:N
        for t in 1:n_test
            samples_it = y_samples[i, t, :]
            lower_bound = quantile(samples_it, lower_q)
            upper_bound = quantile(samples_it, upper_q)
            
            if lower_bound <= y_true[i, t] <= upper_bound
                in_interval += 1
            end
        end
    end
    
    return in_interval / total_points
end

function compute_forecast_quantiles(y_true, y_samples, quantiles)
    """Compute observed quantiles for forecast evaluation"""
    N, n_test, n_samples = size(y_samples)
    n_quantiles = length(quantiles)
    observed_quantiles = zeros(n_quantiles)
    
    y_true_flat = vec(y_true)
    total_points = length(y_true_flat)
    
    for i in 1:n_quantiles
        q = quantiles[i]
        count_below = 0
        
        for asset in 1:N
            for t in 1:n_test
                samples_at = y_samples[asset, t, :]
                threshold = quantile(samples_at, q)
                if y_true[asset, t] < threshold
                    count_below += 1
                end
            end
        end
        
        observed_quantiles[i] = count_below / total_points
    end
    
    return observed_quantiles
end

# Helper functions for prediction generation
function generate_predictions(bnn, chain)
    """Generate naive predictions from posterior samples"""
    N = bnn.like.N
    n_samples = size(chain, 2)
    n_obs = size(bnn.x, 2)
    yhats = Array{Float32}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i in 1:n_samples
        net = bnn.like.nc(chain[:, i])
        
        if typeof(bnn.like) <: DCCGarchRNNSequential
            predictions = rnn_forward_pass(net, bnn.x, N)
        elseif typeof(bnn.like) <: DCCGarchLSTMSequential
            predictions = lstm_forward_pass(net, bnn.x, N)
        else  # Windowed
            predictions = map(t -> net(view(bnn.x, :, t))[1:N], 1:n_obs)
        end
        
        yhats[:, i] = vcat(predictions...)
    end
    
    return yhats
end

function rnn_forward_pass(net, x, N)
    """RNN forward pass for prediction"""
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
    """LSTM forward pass for prediction"""
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

function generate_posterior_predictive(bnn, chain, x, y)
    """Generate posterior predictive samples with DCC dynamics"""
    N = bnn.like.N
    n_samples = min(size(chain, 2), 200)  # Limit for efficiency
    n_obs = size(x, 2)
    pred_samples = Array{Float32}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i in 1:n_samples
        θnet = chain[1:end-2, i]
        θlike = chain[end-1:end, i]
        a, b = transform_ab(θlike...)
        
        # Get mean and variance predictions
        net = bnn.like.nc(θnet)
        μ_all, σ_all = get_moments(net, x, N, typeof(bnn.like))
        
        # Sample with DCC dynamics
        z = (y .- μ_all) ./ σ_all
        Q̄ = (z * z') / n_obs
        d = 1 ./ sqrt.(max.(diag(Q̄), 1e-8))
        Q̄ = Symmetric(diagm(d) * Q̄ * diagm(d))
        Q = copy(Q̄)
        
        samples = similar(μ_all)
        for t in 1:n_obs
            if t > 1
                zt = view(z, :, t-1)
                Q = (1-a-b) .* Q̄ .+ a .* (zt*zt') .+ b .* Q
            end
            
            d = 1 ./ sqrt.(max.(diag(Q), 1e-8))
            R = Symmetric(diagm(d) * Q * diagm(d))
            D = Diagonal(view(σ_all, :, t))
            H = D * R * D + 1e-6*I
            
            samples[:, t] = μ_all[:, t] + cholesky(Symmetric(H)).L * randn(N)
        end
        
        pred_samples[:, i] = vec(samples)
    end
    
    return pred_samples
end

function get_moments(net, x, N, likelihood_type)
    """Extract mean and variance from network outputs"""
    n_obs = size(x, 2)
    
    if likelihood_type <: DCCGarchRNNSequential
        results = rnn_moments(net, x, N)
    elseif likelihood_type <: DCCGarchLSTMSequential
        results = lstm_moments(net, x, N)
    else
        results = map(t -> net(view(x, :, t)), 1:n_obs)
    end
    
    μ = hcat([r[1:N] for r in results]...)
    σ = exp.(hcat([r[N+1:2N] for r in results]...) ./ 2)
    
    return μ, σ
end

function rnn_moments(net, x, N)
    """Get RNN moments"""
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
    """Get LSTM moments"""
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

function compute_observed_quantiles(y_true, y_samples, quantiles, N)
    """Compute observed quantiles for QQ plot"""
    n_quantiles = length(quantiles)
    observed_quantiles = zeros(n_quantiles)
    
    y_true_vector = vec(y_true)
    n_time_points = div(size(y_samples, 1), N)
    total_points = N * n_time_points
    y_samples_reshaped = reshape(y_samples[1:total_points, :], total_points, :)
    
    if length(y_true_vector) > total_points
        y_true_vector = y_true_vector[1:total_points]
    end
    
    for i in 1:n_quantiles
        q = quantiles[i]
        threshold = [quantile(y_samples_reshaped[j, :], q) for j in 1:total_points]
        observed_quantiles[i] = sum(y_true_vector .< threshold) / total_points
    end
    
    return observed_quantiles
end

function print_comprehensive_results_summary(results, n_assets)
    """Print comprehensive results summary with RMSE only"""
    
    println("\n" * "="^80)
    println("COMPREHENSIVE RESULTS SUMMARY ($n_assets Assets)")
    println("="^80)
    
    # Main results table
    println("\n┌─────────────────────┬─────────┬──────────────┬─────────────────────────────────┐")
    println("│ Model               │ R-hat   │ DCC (α, β)   │ QQ MAD (Train/Val/Test)         │")
    println("├─────────────────────┼─────────┼──────────────┼─────────────────────────────────┤")
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM"), 
                               (:windowed, "Windowed FF")]
        if results[model_type] !== nothing
            r = results[model_type]
            α, β = r["dcc_params"]
            train_qq = r["train"]["mad_qq"]
            val_qq = r["validation"]["mad_qq"]
            test_qq = r["test"]["mad_qq"]
            
            println("│ $(rpad(name, 19)) │ $(lpad(string(round(r["r_hat"], digits=3)), 7)) │ $(lpad(string(round(α, digits=3)), 4)), $(lpad(string(round(β, digits=3)), 4)) │ $(lpad(string(round(train_qq, digits=4)), 5)) / $(lpad(string(round(val_qq, digits=4)), 5)) / $(lpad(string(round(test_qq, digits=4)), 5))     │")
        end
    end
    println("└─────────────────────┴─────────┴──────────────┴─────────────────────────────────┘")
    
    # Test set performance details - RMSE only
    println("\n┌─────────────────────┬───────────┬─────────────┬─────────────┐")
    println("│ Model               │ Test RMSE │ Coverage95% │ Test LogLik │")
    println("├─────────────────────┼───────────┼─────────────┼─────────────┤")
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM"), 
                               (:windowed, "Windowed FF")]
        if results[model_type] !== nothing
            r = results[model_type]
            test_rmse = r["test"]["rmse"]
            coverage = r["test"]["coverage_95"]
            test_loglik = r["test"]["loglik"]
            
            println("│ $(rpad(name, 19)) │ $(lpad(string(round(test_rmse, digits=6)), 9)) │ $(lpad(string(round(coverage, digits=3)), 11)) │ $(lpad(string(round(test_loglik, digits=2)), 11)) │")
        end
    end
    println("└─────────────────────┴───────────┴─────────────┴─────────────┘")
    
    # Model ranking
    println("\nMODEL RANKINGS:")
    
    if all(results[k] !== nothing for k in [:rnn, :lstm, :windowed])
        # Rank by test QQ MAD (lower is better)
        test_qq_scores = [(results[k]["test"]["mad_qq"], k) for k in [:rnn, :lstm, :windowed]]
        sort!(test_qq_scores)
        
        println("\nBy Test QQ MAD (Calibration):")
        for (i, (score, model)) in enumerate(test_qq_scores)
            model_name = Dict(:rnn => "Sequential RNN", :lstm => "Sequential LSTM", :windowed => "Windowed FF")[model]
            println("  $i. $model_name: $(round(score, digits=4))")
        end
        
        # Rank by test RMSE (lower is better)
        test_rmse_scores = [(results[k]["test"]["rmse"], k) for k in [:rnn, :lstm, :windowed]]
        sort!(test_rmse_scores)
        
        println("\nBy Test RMSE (Accuracy):")
        for (i, (score, model)) in enumerate(test_rmse_scores)
            model_name = Dict(:rnn => "Sequential RNN", :lstm => "Sequential LSTM", :windowed => "Windowed FF")[model]
            println("  $i. $model_name: $(round(score, digits=6))")
        end
        
        # Rank by coverage (closest to 0.95 is better)
        coverage_scores = [(abs(results[k]["test"]["coverage_95"] - 0.95), k) for k in [:rnn, :lstm, :windowed]]
        sort!(coverage_scores)
        
        println("\nBy Coverage Accuracy (closest to 95%):")
        for (i, (deviation, model)) in enumerate(coverage_scores)
            model_name = Dict(:rnn => "Sequential RNN", :lstm => "Sequential LSTM", :windowed => "Windowed FF")[model]
            actual_coverage = results[model]["test"]["coverage_95"]
            println("  $i. $model_name: $(round(actual_coverage, digits=3)) (deviation: $(round(deviation, digits=3)))")
        end
    end
    
    # Best overall model recommendation
    if all(results[k] !== nothing for k in [:rnn, :lstm, :windowed])
        # Composite score: normalize and combine test QQ MAD, RMSE, and coverage deviation
        scores = Dict()
        
        qq_scores = [results[k]["test"]["mad_qq"] for k in [:rnn, :lstm, :windowed]]
        rmse_scores = [results[k]["test"]["rmse"] for k in [:rnn, :lstm, :windowed]]
        cov_scores = [abs(results[k]["test"]["coverage_95"] - 0.95) for k in [:rnn, :lstm, :windowed]]
        
        # Normalize scores (0-1 scale)
        qq_norm = (qq_scores .- minimum(qq_scores)) ./ (maximum(qq_scores) - minimum(qq_scores) + 1e-10)
        rmse_norm = (rmse_scores .- minimum(rmse_scores)) ./ (maximum(rmse_scores) - minimum(rmse_scores) + 1e-10)
        cov_norm = (cov_scores .- minimum(cov_scores)) ./ (maximum(cov_scores) - minimum(cov_scores) + 1e-10)
        
        # Composite score (equal weights)
        composite_scores = qq_norm .+ rmse_norm .+ cov_norm
        
        models = [:rnn, :lstm, :windowed]
        model_names = ["Sequential RNN", "Sequential LSTM", "Windowed FF"]
        
        best_idx = argmin(composite_scores)
        best_model = model_names[best_idx]
        
        println("\nRECOMMENDED MODEL: $best_model")
        println("(Based on composite score of test calibration, RMSE accuracy, and coverage)")
    end
    
    # Print individual model summaries
    println("\nINDIVIDUAL MODEL SUMMARIES:")
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM"), 
                               (:windowed, "Windowed FF")]
        if results[model_type] !== nothing
            r = results[model_type]
            println("\n$name Results Summary:")
            println("  R-hat: $(round(r["r_hat"], digits=4))")
            println("  DCC params (α, β): $(round.(r["dcc_params"], digits=4))")
            println("  Converged: $(r["converged"] ? "✓" : "✗")")
            println("  Train QQ MAD: $(round(r["train"]["mad_qq"], digits=4))")
            println("  Validation QQ MAD: $(round(r["validation"]["mad_qq"], digits=4))")
            println("  Test QQ MAD: $(round(r["test"]["mad_qq"], digits=4))")
            println("  Test RMSE: $(round(r["test"]["rmse"], digits=6))")
            println("  Test Coverage 95%: $(round(r["test"]["coverage_95"], digits=3))")
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

function comprehensive_dcc_analysis_with_splits(data_path::String, n_assets::Int=30)
    """Run comprehensive DCC-GARCH analysis with proper train/val/test splits"""
    
    println("COMPREHENSIVE DCC-GARCH BAYESIAN NEURAL NETWORK ANALYSIS")
    println("With Train/Validation/Test Splits and Rolling Window Forecasting")
    
    # Data splits (as specified)
    train_index = 110:3450
    val_index = 3451:3700
    test_index = 3701:3950
    
    println("Data splits:")
    println("  Training: $(first(train_index)) to $(last(train_index)) ($(length(train_index)) observations)")
    println("  Validation: $(first(val_index)) to $(last(val_index)) ($(length(val_index)) observations)")
    println("  Test: $(first(test_index)) to $(last(test_index)) ($(length(test_index)) observations)")
    
    # Load and preprocess data
    df = CSV.read(data_path, DataFrame)
    etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", 
                 "16418", "16421", "16423", "16424", "16426", "16433", "16437", 
                 "16452", "16460", "24697", "27635", "28272", "28273", "28274", 
                 "28275", "28276", "28277", "28278", "28279", "28280", "31372", "31466"]
    
    returns, μ_returns = preprocess_financial_data(df, etf_names)
    
    # Select subset if requested
    if n_assets < 30
        selected_indices = 1:n_assets
        returns = returns[:, selected_indices]
        μ_returns = μ_returns[selected_indices]
        selected_assets = vcat(etf_names[1:min(n_assets-1, 29)], n_assets == 30 ? ["rf"] : [])
    else
        selected_assets = vcat(etf_names, ["rf"])
    end
    
    y = Float32.(returns)
    
    println("Selected assets: $(length(selected_assets)) total")
    println("Data dimensions: $(size(y)) (observations × assets)")
    
    # Prepare data for different approaches
    # Sequential data
    X_train_seq, Y_train_seq = prepare_sequential_data(y, train_index)
    X_val_seq, Y_val_seq = prepare_sequential_data(y, val_index)
    
    # Windowed data (22-day window)
    window_size = 22
    X_train_win, Y_train_win = prepare_windowed_data(y, train_index, window_size)
    X_val_win, Y_val_win = prepare_windowed_data(y, val_index, window_size)
    
    # Rolling window test data
    rolling_data_seq = prepare_rolling_window_data(y, test_index, window_size, :sequential)
    rolling_data_win = prepare_rolling_window_data(y, test_index, window_size, :windowed)
    
    println("Training data prepared:")
    println("  Sequential: X=$(size(X_train_seq)), Y=$(size(Y_train_seq))")
    println("  Windowed: X=$(size(X_train_win)), Y=$(size(Y_train_win))")
    println("  Rolling forecasts: $(length(rolling_data_seq)) time points")
    
    # Model configurations
    hidden_size = max(16, min(64, 8 * n_assets))
    models_to_run = [
        (:rnn, X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, rolling_data_seq, "Sequential RNN"),
        (:lstm, X_train_seq, Y_train_seq, X_val_seq, Y_val_seq, rolling_data_seq, "Sequential LSTM"),
        (:windowed, X_train_win, Y_train_win, X_val_win, Y_val_win, rolling_data_win, "Windowed Feedforward")
    ]
    
    results = Dict()
    
    # Train and evaluate each model
    for (model_type, X_train, Y_train, X_val, Y_val, rolling_data, model_name) in models_to_run
        try
            println("\n" * "="^60)
            println("TRAINING AND EVALUATING $model_name")
            println("="^60)
            
            # Train model with validation
            bnn_train, bnn_val, chain, net, val_loglik = train_model_with_validation(
                model_type, X_train, Y_train, X_val, Y_val, n_assets; 
                hidden_size=hidden_size, mcmc_samples=15000, chains=10
            )
            
            println("Training complete. Validation log-likelihood: $(round(val_loglik, digits=4))")
            
            # Rolling window forecasting
            rolling_forecasts = rolling_window_forecast(bnn_train, chain, rolling_data, model_type, n_assets)
            
            # Comprehensive evaluation
            result = evaluate_model_comprehensive(bnn_train, bnn_val, chain, rolling_forecasts, model_name)
            result["val_loglik"] = val_loglik
            results[model_type] = result
            
            # Print summary
            println("\n$model_name Results Summary:")
            println("  R-hat: $(round(result["r_hat"], digits=4))")
            println("  DCC params (α, β): $(round.(result["dcc_params"], digits=4))")
            println("  Converged: $(result["converged"] ? "✓" : "✗")")
            println("  Train QQ MAD: $(round(result["train"]["mad_qq"], digits=4))")
            println("  Validation QQ MAD: $(round(result["validation"]["mad_qq"], digits=4))")
            println("  Test QQ MAD: $(round(result["test"]["mad_qq"], digits=4))")
            println("  Test RMSE: $(round(result["test"]["rmse"], digits=6))")
            println("  Test Coverage 95%: $(round(result["test"]["coverage_95"], digits=3))")
            
        catch e
            println("Error with $model_name: $e")
            results[model_type] = nothing
        end
    end
    
    # Print comprehensive summary
    print_comprehensive_results_summary(results, n_assets)
    
    return results
end

# ══════════════════════════════════════════════════════════════════════════════
#                              MULTIPLE ASSET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

function run_multiple_asset_analysis_with_splits(data_path::String)
    """Run analysis for different numbers of assets with proper splits - RMSE only"""
    
    println("="^80)
    println("MULTI-ASSET DCC-GARCH ANALYSIS WITH TRAIN/VAL/TEST SPLITS")
    println("Running analysis for 2, 5, 10, and 30 assets")
    println("="^80)
    
    asset_counts = [2, 5, 10, 30]
    all_results = Dict()
    summary_results = []
    
    for n_assets in asset_counts
        println("\n" * "="^60)
        println("ANALYZING $n_assets ASSETS")
        println("="^60)
        
        try
            # Run analysis for current asset count
            results = comprehensive_dcc_analysis_with_splits(data_path, n_assets)
            all_results[n_assets] = results
            
            # Extract summary statistics - RMSE only
            for (model_type, name) in [(:rnn, "RNN"), (:lstm, "LSTM"), (:windowed, "Windowed")]
                if results[model_type] !== nothing
                    r = results[model_type]
                    α, β = r["dcc_params"]
                    push!(summary_results, Dict(
                        "n_assets" => n_assets,
                        "model" => name,
                        "r_hat" => r["r_hat"],
                        "alpha" => α,
                        "beta" => β,
                        "train_qq_mad" => r["train"]["mad_qq"],
                        "val_qq_mad" => r["validation"]["mad_qq"],
                        "test_qq_mad" => r["test"]["mad_qq"],
                        "test_rmse" => r["test"]["rmse"],  # Changed from test_mse to test_rmse
                        "test_coverage_95" => r["test"]["coverage_95"],
                        "test_loglik" => r["test"]["loglik"],
                        "val_loglik" => r["val_loglik"],
                        "converged" => r["converged"]
                    ))
                end
            end
            
        catch e
            println("ERROR with $n_assets assets: $e")
            all_results[n_assets] = nothing
        end
        
        # Small delay to let system breathe
        sleep(2)
    end
    
    # Print comprehensive summary
    print_final_comprehensive_summary(summary_results)
    
    # Save results
    if !isempty(summary_results)
        summary_df = DataFrame(summary_results)
        CSV.write("dcc_garch_comprehensive_results_rmse.csv", summary_df)
        println("\nResults saved to: dcc_garch_comprehensive_results_rmse.csv")
    end
    
    return all_results, summary_results
end

function print_final_comprehensive_summary(summary_results)
    """Print the final comprehensive summary of all results - RMSE only"""
    
    println("\n" * "="^80)
    println("FINAL COMPREHENSIVE SUMMARY - ALL MODELS AND ASSET COUNTS")
    println("="^80)
    
    if isempty(summary_results)
        println("No results to summarize.")
        return
    end
    
    # Sort results by asset count then model
    sorted_results = sort(summary_results, by = x -> (x["n_assets"], x["model"]))
    
    # Main performance table - RMSE only
    println("\n" * "="^100)
    println("MAIN PERFORMANCE METRICS")
    println("="^100)
    println("┌─────────┬───────────┬─────────┬─────────┬───────────┬──────────┬─────────────┐")
    println("│ Assets  │ Model     │ R-hat   │ TestQQMAD│ Test RMSE │ Coverage │ TestLogLik  │")
    println("├─────────┼───────────┼─────────┼─────────┼───────────┼──────────┼─────────────┤")
    
    for result in sorted_results
        assets_str = lpad(string(result["n_assets"]), 7)
        model_str = rpad(result["model"], 9)
        r_hat_str = lpad(string(round(result["r_hat"], digits=3)), 7)
        qq_mad_str = lpad(string(round(result["test_qq_mad"], digits=4)), 8)
        rmse_str = lpad(string(round(result["test_rmse"], digits=6)), 9)
        coverage_str = lpad(string(round(result["test_coverage_95"], digits=3)), 8)
        loglik_str = lpad(string(round(result["test_loglik"], digits=2)), 11)
        
        println("│ $assets_str │ $model_str │ $r_hat_str │ $qq_mad_str │ $rmse_str │ $coverage_str │ $loglik_str │")
    end
    println("└─────────┴───────────┴─────────┴─────────┴───────────┴──────────┴─────────────┘")
    
    # Best performers by metric
    println("\nBEST PERFORMERS BY METRIC:")
    
    # Best test QQ MAD by asset count
    println("\nBest Test Calibration (QQ MAD) by Asset Count:")
    for n_assets in [2, 5, 10, 30]
        asset_results = filter(x -> x["n_assets"] == n_assets, sorted_results)
        if !isempty(asset_results)
            best = minimum(asset_results, by = x -> x["test_qq_mad"])
            println("  $n_assets assets: $(best["model"]) ($(round(best["test_qq_mad"], digits=4)))")
        end
    end
    
    # Best test RMSE by asset count
    println("\nBest Test Accuracy (RMSE) by Asset Count:")
    for n_assets in [2, 5, 10, 30]
        asset_results = filter(x -> x["n_assets"] == n_assets, sorted_results)
        if !isempty(asset_results)
            best = minimum(asset_results, by = x -> x["test_rmse"])
            println("  $n_assets assets: $(best["model"]) ($(round(best["test_rmse"], digits=6)))")
        end
    end
    
    # Best coverage by asset count
    println("\nBest Test Coverage (closest to 95%) by Asset Count:")
    for n_assets in [2, 5, 10, 30]
        asset_results = filter(x -> x["n_assets"] == n_assets, sorted_results)
        if !isempty(asset_results)
            best = minimum(asset_results, by = x -> abs(x["test_coverage_95"] - 0.95))
            println("  $n_assets assets: $(best["model"]) ($(round(best["test_coverage_95"], digits=3)))")
        end
    end
    
    # Overall best model
    println("\nOVERALL PERFORMANCE ANALYSIS:")
    
    # Count wins by model across all metrics and asset counts
    model_wins = Dict("RNN" => 0, "LSTM" => 0, "Windowed" => 0)
    
    for n_assets in [2, 5, 10, 30]
        asset_results = filter(x -> x["n_assets"] == n_assets, sorted_results)
        if length(asset_results) >= 3
            # QQ MAD winner
            qq_winner = minimum(asset_results, by = x -> x["test_qq_mad"])["model"]
            model_wins[qq_winner] += 1
            
            # RMSE winner
            rmse_winner = minimum(asset_results, by = x -> x["test_rmse"])["model"]
            model_wins[rmse_winner] += 1
            
            # Coverage winner
            cov_winner = minimum(asset_results, by = x -> abs(x["test_coverage_95"] - 0.95))["model"]
            model_wins[cov_winner] += 1
        end
    end
    
    println("\nModel Performance Wins (out of 12 possible: 4 asset counts × 3 metrics):")
    for (model, wins) in sort(collect(model_wins), by = x -> x[2], rev = true)
        println("  $model: $wins wins")
    end
    
    # Convergence summary
    println("\nCONVERGENCE ANALYSIS:")
    for model in ["RNN", "LSTM", "Windowed"]
        model_results = filter(x -> x["model"] == model, sorted_results)
        converged_count = sum(r["converged"] for r in model_results)
        total_count = length(model_results)
        if total_count > 0
            percentage = round(100 * converged_count / total_count, digits=1)
            println("  $model: $converged_count/$total_count converged ($percentage%)")
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                              RUN THE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# Set the data path
data_path = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"

println("Starting comprehensive multi-asset DCC-GARCH analysis with train/val/test splits...")

# For testing, you can run with a single asset count first:
results = comprehensive_dcc_analysis_with_splits(data_path, 2)

# For full analysis:
all_results, summary_results = run_multiple_asset_analysis_with_splits(data_path)

println("\nAnalysis complete!")