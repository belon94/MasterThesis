###############################################################################
#   DCC-GARCH Bayesian Neural Network - Complete Multi-Asset Analysis
#  With Train/Validation/Test Splits and Rolling Window Forecasting
#  RMSE Evaluation with Robust + Portfolio Optimization & VaR
###############################################################################

using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames
using Printf

Random.seed!(1212)

# Ensure all required constants are defined
const f0 = 0f0  # Define f0 as Float32 zero if it's expected somewhere

# ══════════════════════════════════════════════════════════════════════════════
#                              SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Mathematical helpers - FIXED
sigmoid(x) = 1.0f0/(1.0f0+exp(-x))

# FIXED: More explicit transform_ab function
function transform_ab(a, b)
    a_transformed = 1.0f0 / (1.0f0 + exp(-a))  # sigmoid transformation
    b_transformed = a_transformed * (1.0f0 / (1.0f0 + exp(-b)))  # constrained sigmoid
    return (a_transformed, b_transformed)
end

nearest_pd(A) = (A + A')/2 + 1e-4f0*I

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
    """Prepare data for rolling window forecasting - FIXED DIMENSIONS"""
    rolling_forecasts = []
    
    for test_idx in test_indices
        if approach == :sequential
            # For sequential: use previous observation to predict current
            if test_idx > 1
                X_t = Float32.(vec(data[test_idx-1, :])) 
                Y_t = Float32.(vec(data[test_idx, :]))    
            end
        else # windowed
            # For windowed: use window_size previous observations
            if test_idx > window_size
                X_t = Float32.(reshape(data[test_idx-window_size:test_idx-1, :]', :))
                Y_t = Float32.(vec(data[test_idx, :]))   
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
#                    COVARIANCE MATRIX AND PORTFOLIO FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

function extract_covariance_matrices_from_chain(bnn, chain, X_forecast)
    """Extract covariance matrices from MCMC samples for forecasting - FIXED DIMENSIONS"""
    N = bnn.like.N
    n_samples = size(chain, 2)
    covariance_matrices = Array{Float32}(undef, N, N, n_samples)
    forecast_means = Array{Float32}(undef, N, n_samples)
    
    for i in 1:n_samples
        θnet = chain[1:end-2, i]
        θlike = chain[end-1:end, i]
        
        # Get network prediction
        net = bnn.like.nc(θnet)
        
        # Generate forecast for single time point - FIXED: Ensure X_forecast is correct shape
        if typeof(bnn.like) <: DCCGarchRNNSequential
            output = rnn_single_forecast(net, X_forecast, N)
        elseif typeof(bnn.like) <: DCCGarchLSTMSequential
            output = lstm_single_forecast(net, X_forecast, N)
        else  # windowed
            output = net(X_forecast)
        end
        
        μ_pred = output[1:N]
        logσ2_pred = output[N+1:2N]
        
        # Clamp log-variance to reasonable range
        logσ2_pred = clamp.(logσ2_pred, -10.0f0, 5.0f0)
        σ_pred = exp.(logσ2_pred ./ 2.0f0)
        
        # Transform DCC parameters
        a, b = transform_ab(θlike...)
        
        # Construct correlation matrix using DCC dynamics
        # For forecasting, we use a simplified approach with identity initial correlation
        Q̄ = Matrix{Float32}(I, N, N)
        R = Q̄  # Simplified for one-step forecast
        
        # Construct covariance matrix
        D = Diagonal(σ_pred)
        H = D * R * D
        H = nearest_pd(H)  # Ensure positive definiteness
        
        covariance_matrices[:, :, i] = H
        forecast_means[:, i] = μ_pred
    end
    
    return covariance_matrices, forecast_means
end

function compute_average_covariance_matrix(covariance_matrices)
    """Compute average covariance matrix across MCMC samples"""
    N, _, n_samples = size(covariance_matrices)
    avg_cov = zeros(Float32, N, N)
    
    valid_samples = 0
    for i in 1:n_samples
        cov_matrix = covariance_matrices[:, :, i]
        if all(isfinite, cov_matrix) && isposdef(cov_matrix)
            avg_cov += cov_matrix
            valid_samples += 1
        end
    end
    
    avg_cov /= valid_samples
    return nearest_pd(avg_cov)  # Ensure positive definiteness
end

function construct_minimum_variance_portfolio(cov_matrix)
    """Construct minimum variance portfolio weights"""
    N = size(cov_matrix, 1)
    ones_vec = ones(Float32, N)
    
    # Minimum variance portfolio: w = (Σ^(-1) * 1) / (1' * Σ^(-1) * 1)
    cov_inv = inv(cov_matrix)
    numerator = cov_inv * ones_vec
    denominator = ones_vec' * cov_inv * ones_vec
    
    weights = numerator / denominator[1]
    
    return weights
end

function calculate_portfolio_var(portfolio_weights, forecast_mean, cov_matrix, confidence_levels=[0.01, 0.05, 0.10])
    """Calculate portfolio VaR using analytical formula"""
    
    # Portfolio expected return
    μ_portfolio = portfolio_weights' * forecast_mean
    
    # Portfolio variance
    σ²_portfolio = portfolio_weights' * cov_matrix * portfolio_weights
    σ_portfolio = sqrt(σ²_portfolio[1])
    
    # Z-scores for different confidence levels for the VaR calculation https://www.investopedia.com/articles/04/092904.asp
    z_scores = Dict(
        0.01 => 2.326f0,  # 99% confidence (1% tail)  
        0.05 => 1.645f0,  # 95% confidence (5% tail)
        0.10 => 1.282f0   # 90% confidence (10% tail)
    )
    
    # Calculate VaR using formula: VaR = -μ_p + z_α * σ_p
    var_results = Dict()
    for α in confidence_levels
        if haskey(z_scores, α)
            var_value = -μ_portfolio[1] + z_scores[α] * σ_portfolio
            var_results[α] = var_value
        end
    end
    
    return var_results, μ_portfolio[1], σ_portfolio
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
#                              TRAINING AND VALIDATION - FIXED INITIALIZATION
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
    
    # Setup BNN components - FIXED INITIALIZATION
    nc = destruct(net)
    like = likelihood_type(nc, Normal(0f0, prior_std), N)  # FIXED: Use 0f0 instead of 0
    prior = GaussianPrior(nc, prior_std)
    
    # FIXED: Use more explicit initialization
    init_dist = Normal(0f0, prior_std)
    init = InitialiseAllSame(init_dist, like, prior)
    
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
        bnn_val.like(bnn_val.x, bnn_val.y, θnet, θlike)
    end
    
    # Remove infinite values and compute mean
    finite_logliks = filter(isfinite, val_logliks)
    return mean(finite_logliks)
end

# ══════════════════════════════════════════════════════════════════════════════
#                              ENHANCED ROLLING WINDOW FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

function rolling_covariance_forecast_with_portfolio_analysis(bnn_train, chain, rolling_data, model_type::Symbol, N::Int)
    """Enhanced rolling window forecasting with covariance matrix extraction and portfolio analysis"""
    
    println("Performing rolling window forecasting with portfolio analysis...")
    n_test = length(rolling_data)
    n_samples = min(size(chain, 2), 100)  # Use subset for efficiency
    
    # Storage for forecasts and portfolio analysis
    forecasts = Dict(
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
    
    for (t_idx, (X_t, Y_t, original_idx)) in enumerate(rolling_data)
        if t_idx % 25 == 0
            println("  Processing forecast $t_idx/$n_test")
        end
        
        # Store actual values
        forecasts["actuals"][:, t_idx] = Y_t
        
        # Extract covariance matrices from MCMC samples
        cov_matrices, forecast_means = extract_covariance_matrices_from_chain(bnn_train, chain, X_t)
        
        # Compute average covariance matrix and forecast mean
        avg_cov_matrix = compute_average_covariance_matrix(cov_matrices)
        avg_forecast_mean = mean(forecast_means, dims=2)[:, 1]
        
        # Store results
        forecasts["means"][:, t_idx] = avg_forecast_mean
        forecasts["covariance_matrices"][:, :, t_idx] = avg_cov_matrix
        
        # Construct minimum variance portfolio
        mv_weights = construct_minimum_variance_portfolio(avg_cov_matrix)
        forecasts["mv_weights"][:, t_idx] = mv_weights
        
        # Calculate portfolio metrics
        var_results, portfolio_return, portfolio_vol = calculate_portfolio_var(
            mv_weights, avg_forecast_mean, avg_cov_matrix, [0.01, 0.05, 0.10]
        )
        
        forecasts["portfolio_returns"][t_idx] = portfolio_return
        forecasts["portfolio_volatility"][t_idx] = portfolio_vol
        forecasts["portfolio_var_1"][t_idx] = var_results[0.01]
        forecasts["portfolio_var_5"][t_idx] = var_results[0.05]
        forecasts["portfolio_var_10"][t_idx] = var_results[0.10]
    end
    
    return forecasts
end

# Original rolling window forecast function (kept for compatibility)
function rolling_window_forecast(bnn_train, chain, rolling_data, model_type::Symbol, N::Int)
    """Perform rolling window forecasting on test data with robust NaN handling"""
    
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
        
        valid_samples = 0
        for (s_idx, chain_idx) in enumerate(sample_indices)
            θnet = chain[1:end-2, chain_idx]
            θlike = chain[end-1:end, chain_idx]
            
            # Single point forecast
            mean_pred, var_pred, sample_pred, loglik = single_point_forecast(
                θnet, θlike, X_t, Y_t, bnn_train.like, model_type, N
            )
            
            # Check for valid predictions
            if all(isfinite, mean_pred) && all(isfinite, var_pred) && all(isfinite, sample_pred) && isfinite(loglik)
                forecast_samples[:, s_idx] = sample_pred
                logliks[s_idx] = loglik
                valid_samples += 1
            end
        end
        
        # Compute summary statistics with robust handling
        # Filter out invalid samples before computing statistics
        valid_mask = [all(isfinite, forecast_samples[:, i]) for i in 1:n_samples]
        valid_samples_filtered = forecast_samples[:, valid_mask]
        
        forecasts["means"][:, t_idx] = mean(valid_samples_filtered, dims=2)[:, 1]
        forecasts["vars"][:, t_idx] = var(valid_samples_filtered, dims=2)[:, 1]
        
        # Store all samples
        forecasts["samples"][:, t_idx, :] = forecast_samples
        
        # Average log-likelihood from valid samples
        valid_logliks = logliks[isfinite.(logliks)]
        forecasts["loglik"][t_idx] = mean(valid_logliks)
    end
    
    return forecasts
end

function single_point_forecast(θnet, θlike, X_t, Y_t, likelihood, model_type::Symbol, N::Int)
    """Generate single point forecast with robust error handling - FIXED SCOPE"""
    
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
    
    # Clamp log-variance to reasonable range
    logσ2_pred = clamp.(logσ2_pred, -10.0f0, 5.0f0)
    σ_pred = exp.(logσ2_pred ./ 2.0f0)
    
    # Transform DCC parameters
    a, b = transform_ab(θlike...)
    
    # Ensure σ_pred is positive and finite
    σ_pred = max.(σ_pred, 1e-6f0)
    
    # For simplicity in single point forecast, use diagonal covariance
    H = Diagonal(σ_pred .^ 2) + 1e-6f0*I
    
    # FIXED: Declare sample_pred and loglik before try-catch blocks
    sample_pred = zeros(Float32, N)  # Initialize with default value
    loglik = 0f0  # Initialize with default value
    
    # Generate sample with error handling
    try
        L = cholesky(Symmetric(H)).L
        sample_pred = μ_pred + L * randn(Float32, N)
    catch
        # Fallback: use diagonal covariance only
        sample_pred = μ_pred + σ_pred .* randn(Float32, N)
    end
    
    # Compute log-likelihood with error handling
    try
        diff = Y_t - μ_pred
        L = cholesky(Symmetric(H)).L
        quad = sum(abs2, L \ diff)
        loglik = -0.5f0 * (N * log(2π) + 2 * sum(log, diag(L)) + quad)
    catch
        # Fallback likelihood computation
        diff = Y_t - μ_pred
        quad = sum(abs2, diff ./ σ_pred)
        loglik = -0.5f0 * (N * log(2π) + 2 * sum(log, σ_pred) + quad)
    end
    
    return μ_pred, σ_pred .^ 2, sample_pred, loglik
end

function rnn_single_forecast(net, X_t, N)
    """Single RNN forecast - FIXED: X_t should be a vector"""
    rnn_layer, dense_layer = net[1], net[2]
    hidden_size = size(rnn_layer.cell.Wi, 1)
    h = zeros(Float32, hidden_size)
    
    # Single step forward - X_t is now correctly a vector
    h = tanh.(rnn_layer.cell.Wi * X_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
    return dense_layer(h)
end

function lstm_single_forecast(net, X_t, N)
    """Single LSTM forecast - FIXED: X_t should be a vector"""
    lstm_layer, dense_layer = net[1], net[2]
    hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
    h, c = zeros(Float32, hidden_size), zeros(Float32, hidden_size)
    
    # Single step forward - X_t is now correctly a vector
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
#                              PORTFOLIO PERFORMANCE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

function evaluate_portfolio_performance(forecasts, model_name::String)
    """Evaluate portfolio performance including VaR backtesting"""
    
    println("\n" * "="^60)
    println("PORTFOLIO PERFORMANCE ANALYSIS: $model_name")
    println("="^60)
    
    n_test = length(forecasts["portfolio_returns"])
    
    # Portfolio return statistics
    avg_portfolio_return = mean(forecasts["portfolio_returns"])
    portfolio_volatility = mean(forecasts["portfolio_volatility"])
    
    # Average VaR values
    avg_var_1 = mean(forecasts["portfolio_var_1"])
    avg_var_5 = mean(forecasts["portfolio_var_5"])
    avg_var_10 = mean(forecasts["portfolio_var_10"])
    
    println("Portfolio Performance Summary:")
    println("  Average Portfolio Return: $(round(avg_portfolio_return * 100, digits=4))%")
    println("  Average Portfolio Volatility: $(round(portfolio_volatility * 100, digits=4))%")
    println("\nValue at Risk Summary:")
    println("  Average 1% VaR: $(round(avg_var_1 * 100, digits=4))%")
    println("  Average 5% VaR: $(round(avg_var_5 * 100, digits=4))%")
    println("  Average 10% VaR: $(round(avg_var_10 * 100, digits=4))%")
    
    return Dict(
        "avg_portfolio_return" => avg_portfolio_return,
        "portfolio_volatility" => portfolio_volatility,
        "avg_var_1" => avg_var_1,
        "avg_var_5" => avg_var_5,
        "avg_var_10" => avg_var_10
    )
end

# ══════════════════════════════════════════════════════════════════════════════
#                              EVALUATION METRICS (FIXED NaN HANDLING)
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
        bnn.like(bnn.x, bnn.y, chain[1:end-2, i], chain[end-1:end, i])
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
    """Evaluate out-of-sample performance with robust NaN handling"""
    
    # Extract data
    y_true = forecasts["actuals"]
    y_pred_mean = forecasts["means"]
    y_pred_var = forecasts["vars"]
    logliks = forecasts["loglik"]
    
    # Filter out non-finite predictions for RMSE computation
    valid_mask = isfinite.(y_true) .& isfinite.(y_pred_mean)
    rmse = sqrt(mean((y_true[valid_mask] - y_pred_mean[valid_mask]).^2))
    
    # Average log-likelihood from finite values
    finite_logliks = logliks[isfinite.(logliks)]
    avg_loglik = mean(finite_logliks)
    
    # Coverage probabilities with robust handling
    coverage_90 = compute_coverage_probability(y_true, forecasts["samples"], 0.90)
    coverage_95 = compute_coverage_probability(y_true, forecasts["samples"], 0.95)
    
    # QQ evaluation with robust handling
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
    """Compute empirical coverage probability with robust NaN handling"""
    α = 1 - confidence_level
    lower_q = α / 2
    upper_q = 1 - α / 2
    
    N, n_test, n_samples = size(y_samples)
    total_points = 0
    in_interval = 0
    
    for i in 1:N
        for t in 1:n_test
            samples_it = y_samples[i, t, :]
            
            # Filter out non-finite values
            valid_samples = samples_it[isfinite.(samples_it)]
            
            lower_bound = quantile(valid_samples, lower_q)
            upper_bound = quantile(valid_samples, upper_q)
            
            if isfinite(y_true[i, t]) && lower_bound <= y_true[i, t] <= upper_bound
                in_interval += 1
            end
            total_points += 1
        end
    end
    
    return in_interval / total_points
end

function compute_forecast_quantiles(y_true, y_samples, quantiles)
    """Compute observed quantiles with robust NaN handling"""
    N, n_test, n_samples = size(y_samples)
    n_quantiles = length(quantiles)
    observed_quantiles = zeros(n_quantiles)
    
    total_valid_points = 0
    
    for i in 1:n_quantiles
        q = quantiles[i]
        count_below = 0
        points_processed = 0
        
        for asset in 1:N
            for t in 1:n_test
                samples_at = y_samples[asset, t, :]
                
                # Filter out non-finite values
                valid_samples = samples_at[isfinite.(samples_at)]
                
                if isfinite(y_true[asset, t])
                    threshold = quantile(valid_samples, q)
                    if isfinite(threshold) && y_true[asset, t] < threshold
                        count_below += 1
                    end
                    points_processed += 1
                end
            end
        end
        
        if i == 1
            total_valid_points = points_processed
        end
        
        observed_quantiles[i] = count_below / points_processed
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
    """Generate posterior predictive samples with robust error handling"""
    N = bnn.like.N
    n_samples = min(size(chain, 2), 200)
    n_obs = size(x, 2)
    pred_samples = Array{Float32}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i in 1:n_samples
        θnet = chain[1:end-2, i]
        θlike = chain[end-1:end, i]
        a, b = transform_ab(θlike...)
        
        # Get mean and variance predictions
        net = bnn.like.nc(θnet)
        μ_all, σ_all = get_moments(net, x, N, typeof(bnn.like))
        
        # Clamp extreme values
        σ_all = clamp.(σ_all, 1e-6f0, 10f0)
        
        # Sample with simplified covariance for robustness
        samples = similar(μ_all)
        for t in 1:n_obs
            # Use diagonal covariance for robustness
            samples[:, t] = μ_all[:, t] + σ_all[:, t] .* randn(Float32, N)
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
    """Compute observed quantiles with robust NaN handling"""
    n_quantiles = length(quantiles)
    observed_quantiles = zeros(n_quantiles)
    
    y_true_vector = vec(y_true)
    n_time_points = div(size(y_samples, 1), N)
    total_points = N * n_time_points
    
    # Ensure consistent dimensions
    if size(y_samples, 1) < total_points
        total_points = size(y_samples, 1)
    end
    
    y_samples_reshaped = y_samples[1:total_points, :]
    
    if length(y_true_vector) > total_points
        y_true_vector = y_true_vector[1:total_points]
    end
    
    for i in 1:n_quantiles
        q = quantiles[i]
        points_below = 0
        valid_points = 0
        
        for j in 1:total_points
            if isfinite(y_true_vector[j])
                # Get valid samples for this point
                point_samples = y_samples_reshaped[j, :]
                valid_samples = point_samples[isfinite.(point_samples)]
                
                threshold = quantile(valid_samples, q)
                if isfinite(threshold) && y_true_vector[j] < threshold
                    points_below += 1
                end
                valid_points += 1
            end
        end
        
        observed_quantiles[i] = points_below / valid_points
    end
    
    return observed_quantiles
end

# ══════════════════════════════════════════════════════════════════════════════
#                              ENHANCED REPORTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

function print_portfolio_comparison(portfolio_results, n_assets)
    """Print comparison of portfolio performance across models"""
    
    println("\n" * "="^80)
    println("PORTFOLIO PERFORMANCE COMPARISON ($n_assets Assets)")
    println("="^80)
    
    println("\n┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    println("│ Model               │ Avg Return% │ Volatility% │ VaR 1%      │ VaR 5%      │ VaR 10%     │")
    println("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM"), 
                               (:windowed, "Windowed FF")]
        perf = portfolio_results[model_type]["performance"]
        
        avg_ret_str = lpad(string(round(perf["avg_portfolio_return"] * 100, digits=4)), 11)
        vol_str = lpad(string(round(perf["portfolio_volatility"] * 100, digits=4)), 11)
        var1_str = lpad(string(round(perf["avg_var_1"] * 100, digits=4)), 11)
        var5_str = lpad(string(round(perf["avg_var_5"] * 100, digits=4)), 11)
        var10_str = lpad(string(round(perf["avg_var_10"] * 100, digits=4)), 11)
        
        println("│ $(rpad(name, 19)) │ $avg_ret_str │ $vol_str │ $var1_str │ $var5_str │ $var10_str │")
    end
    println("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")
end

function print_comprehensive_results_summary(results, n_assets)
    """Print comprehensive results summary with Train/Val/Test RMSE"""
    
    println("\n" * "="^80)
    println("COMPREHENSIVE RESULTS SUMMARY ($n_assets Assets)")
    println("="^80)
    
    # Enhanced results table with all RMSE values
    println("\n┌─────────────────────┬─────────┬──────────────┬─────────────────────────────────────────────────┐")
    println("│ Model               │ R-hat   │ DCC (α, β)   │ RMSE (Train/Val/Test)                           │")
    println("├─────────────────────┼─────────┼──────────────┼─────────────────────────────────────────────────┤")
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM"), 
                               (:windowed, "Windowed FF")]
        r = results[model_type]
        α, β = r["dcc_params"]
        train_rmse = r["train"]["rmse"]
        val_rmse = r["validation"]["rmse"] 
        test_rmse = r["test"]["rmse"]
        
        println("│ $(rpad(name, 19)) │ $(lpad(string(round(r["r_hat"], digits=3)), 7)) │ $(lpad(string(round(α, digits=3)), 4)), $(lpad(string(round(β, digits=3)), 4)) │ $(lpad(string(round(train_rmse, digits=6)), 7)) / $(lpad(string(round(val_rmse, digits=6)), 7)) / $(lpad(string(round(test_rmse, digits=6)), 7)) │")
    end
    println("└─────────────────────┴─────────┴──────────────┴─────────────────────────────────────────────────┘")
end

# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN ANALYSIS FUNCTION WITH PORTFOLIO OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

function comprehensive_dcc_analysis_with_portfolio(data_path::String, n_assets::Int=30)
    """Run comprehensive DCC-GARCH analysis with portfolio optimization and VaR calculation"""
    
    println("COMPREHENSIVE DCC-GARCH ANALYSIS WITH PORTFOLIO OPTIMIZATION")
    println("Including Minimum Variance Portfolio Construction and VaR Calculation")
    
    # Data splits
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
    portfolio_results = Dict()
    
    # Train and evaluate each model with portfolio analysis
    for (model_type, X_train, Y_train, X_val, Y_val, rolling_data, model_name) in models_to_run
        println("\n" * "="^60)
        println("TRAINING AND EVALUATING $model_name WITH PORTFOLIO ANALYSIS")
        println("="^60)
        
        # Train model with validation
        bnn_train, bnn_val, chain, net, val_loglik = train_model_with_validation(
            model_type, X_train, Y_train, X_val, Y_val, n_assets; 
            hidden_size=hidden_size, mcmc_samples=15000, chains=10
        )
        
        println("Training complete. Validation log-likelihood: $(round(val_loglik, digits=4))")
        
        # Enhanced rolling window forecasting with portfolio analysis
        portfolio_forecasts = rolling_covariance_forecast_with_portfolio_analysis(
            bnn_train, chain, rolling_data, model_type, n_assets
        )
        
        # Evaluate portfolio performance
        portfolio_performance = evaluate_portfolio_performance(portfolio_forecasts, model_name)
        
        # Store portfolio results
        portfolio_results[model_type] = Dict(
            "forecasts" => portfolio_forecasts,
            "performance" => portfolio_performance,
            "model_name" => model_name
        )
        
        # Also compute regular model evaluation for comparison
        regular_forecasts = rolling_window_forecast(bnn_train, chain, rolling_data, model_type, n_assets)
        result = evaluate_model_comprehensive(bnn_train, bnn_val, chain, regular_forecasts, model_name)
        result["val_loglik"] = val_loglik
        results[model_type] = result
        
        # Print model summary
        println("\n$model_name Results Summary:")
        println("  R-hat: $(round(result["r_hat"], digits=4))")
        println("  DCC params (α, β): $(round.(result["dcc_params"], digits=4))")
        println("  Converged: $(result["converged"] ? "✓" : "✗")")
        println("  Training RMSE: $(round(result["train"]["rmse"], digits=6))")
        println("  Validation RMSE: $(round(result["validation"]["rmse"], digits=6))")
        println("  Test RMSE: $(round(result["test"]["rmse"], digits=6))")
    end
    
    # Print comprehensive summaries
    print_comprehensive_results_summary(results, n_assets)
    print_portfolio_comparison(portfolio_results, n_assets)
    
    return results, portfolio_results
end

# ══════════════════════════════════════════════════════════════════════════════
#                              RUN THE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# Set the data path
data_path = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"

# Run analysis with portfolio optimization for a specific number of assets
results, portfolio_results = comprehensive_dcc_analysis_with_portfolio(data_path, 2)

# Save portfolio results
if !isempty(portfolio_results)
    # Extract portfolio performance data for CSV
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