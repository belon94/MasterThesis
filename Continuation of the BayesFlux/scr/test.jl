###############################################################################
#  Comprehensive DCC-GARCH Bayesian Neural Network Suite
#  Comparing RNN, LSTM, and Feedforward Approaches
#  WITH OUT-OF-SAMPLE FORECASTING 
###############################################################################

using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames

# ══════════════════════════════════════════════════════════════════════════════
#                              SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Mathematical helpers
sigmoid(x) = 1/(1+exp(-x))
transform_ab(a,b) = let a_=sigmoid(a); b_=sigmoid(b)*(1-a_); (a_,b_) end
nearest_pd(A) = (A + A')/2 + 1e-4I

# Data preparation functions
function prepare_sequential_data(data)
    """Sequential data: X[t] -> Y[t+1]"""
    X = data[1:end-1, :]'
    Y = data[2:end, :]'
    return Float32.(X), Float32.(Y)
end

function prepare_windowed_data(data, window_size)
    """Windowed data: X[t-w:t-1] -> Y[t]"""
    n, p = size(data)
    n_samples = n - window_size
    X = Array{Float32}(undef, p * window_size, n_samples)
    Y = Array{Float32}(undef, p, n_samples)
    
    for i in 1:n_samples
        X[:, i] = reshape(data[i:i+window_size-1, :]', :)
        Y[:, i] = data[i+window_size, :]
    end
    return X, Y
end

# Data preprocessing
function preprocess_financial_data(df, asset_names)
    """Handle missing values and extract returns"""
    for col in asset_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    returns = Matrix{Float64}(df[!, asset_names])
    μ_returns = mean(returns, dims=1)
    return returns, vec(μ_returns)
end

function create_chronological_splits(data, train_ratio=0.6, val_ratio=0.2)
    """Create chronological train/validation/test splits"""
    n_obs = size(data, 1)
    
    train_end = Int(floor(train_ratio * n_obs))
    val_end = Int(floor((train_ratio + val_ratio) * n_obs))
    
    train_idx = 1:train_end
    val_idx = (train_end + 1):val_end
    test_idx = (val_end + 1):n_obs
    
    return train_idx, val_idx, test_idx
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
#                              TRAINING AND EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

function train_model(model_type::Symbol, X, Y, N::Int; 
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
    bnn = BNN(X, Y, like, prior, init)
    
    # Find MAP estimate
    println("Finding MAP estimate...")
    θmap = find_mode(bnn, 50, 1000, FluxModeFinder(bnn, Flux.ADAM(0.005)))
    
    # MCMC sampling
    println("Starting MCMC sampling...")
    step_size = N <= 5 ? 1f-4 : 2f-5
    sampler = SGNHTS(step_size, 1f0; xi=1f0^2, μ=10f0)
    ch = mcmc(bnn, chains, mcmc_samples, sampler)
    
    # Keep latter half of samples
    burn_in = mcmc_samples ÷ 2
    ch = ch[:, end-burn_in+1:end]
    
    return bnn, ch, net
end

function evaluate_model(bnn, chain, X, Y, model_name::String)
    """Comprehensive model evaluation"""
    N = bnn.like.N
    
    # Generate predictions
    yhats = generate_predictions(bnn, chain)
    chain_yhat = Chains(yhats')
    r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
    
    # Posterior predictive samples
    posterior_samples = generate_posterior_predictive(bnn, chain, X, Y)
    
    # QQ evaluation
    t_q = 0.05:0.05:0.95
    o_q = compute_observed_quantiles(Y', posterior_samples, t_q, N)
    
    # DCC parameters
    dcc_params = transform_ab(mean(chain[end-1:end, :], dims=2)...)
    
    # Create QQ plot
    qq_plot = plot(t_q, o_q, label=model_name, legend=:topleft,
                   xlabel="Target Quantile", ylabel="Observed Quantile",
                   linewidth=2, title="Q-Q Plot: $model_name")
    plot!(qq_plot, x->x, t_q, label="Perfect Calibration", 
          linestyle=:dash, color=:black)
    
    results = Dict(
        "model_name" => model_name,
        "n_assets" => N,
        "r_hat" => r_hat,
        "dcc_params" => dcc_params,
        "mad_qq" => mean(abs.(o_q .- t_q)),
        "qq_plot" => qq_plot,
        "converged" => r_hat < 1.2
    )
    
    return results
end

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

# ══════════════════════════════════════════════════════════════════════════════
#                          OUT-OF-SAMPLE FORECASTING MODULE
# ══════════════════════════════════════════════════════════════════════════════

function forecast_one_step_ahead(bnn, chain, x_history, model_type; n_samples=200)
    """Generate one-step-ahead forecasts for returns and covariance"""
    N = bnn.like.N
    n_posterior_samples = min(size(chain, 2), n_samples)
    
    # Storage for forecasts
    return_forecasts = Array{Float32}(undef, N, n_posterior_samples)
    covariance_forecasts = Array{Float32}(undef, N, N, n_posterior_samples)
    
    Threads.@threads for i in 1:n_posterior_samples
        θnet = chain[1:end-2, i]
        θlike = chain[end-1:end, i]
        a, b = transform_ab(θlike...)
        
        net = bnn.like.nc(θnet)
        
        # Get point forecast for mean and log-variance
        if model_type == :rnn
            μ_forecast, logσ2_forecast = forecast_rnn_step(net, x_history, N)
        elseif model_type == :lstm
            μ_forecast, logσ2_forecast = forecast_lstm_step(net, x_history, N)
        else  # windowed
            last_window = view(x_history, :, size(x_history, 2))
            output = net(last_window)
            μ_forecast = output[1:N]
            logσ2_forecast = output[N+1:2N]
        end
        
        # Forecast covariance matrix using DCC dynamics
        σ_forecast = exp.(logσ2_forecast ./ 2)
        R_forecast = forecast_correlation_matrix(x_history, a, b, N)
        D_forecast = Diagonal(σ_forecast)
        H_forecast = D_forecast * R_forecast * D_forecast
        
        return_forecasts[:, i] = μ_forecast
        covariance_forecasts[:, :, i] = H_forecast
    end
    
    return return_forecasts, covariance_forecasts
end

function forecast_rnn_step(net, x_history, N)
    """One-step RNN forecast"""
    rnn_layer, dense_layer = net[1], net[2]
    hidden_size = size(rnn_layer.cell.Wi, 1)
    h = zeros(Float32, hidden_size)
    
    # Forward pass through historical data
    for t in 1:size(x_history, 2)
        input_t = view(x_history, :, t)
        h = tanh.(rnn_layer.cell.Wi * input_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
    end
    
    # Generate forecast (using last hidden state)
    output = dense_layer(h)
    return output[1:N], output[N+1:2N]
end

function forecast_lstm_step(net, x_history, N)
    """One-step LSTM forecast"""
    lstm_layer, dense_layer = net[1], net[2]
    hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
    h, c = zeros(Float32, hidden_size), zeros(Float32, hidden_size)
    
    # Forward pass through historical data
    for t in 1:size(x_history, 2)
        input_t = view(x_history, :, t)
        gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
        
        i_gate = sigmoid.(gates[1:hidden_size])
        f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
        g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
        o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
        
        c = f_gate .* c .+ i_gate .* g_gate
        h = o_gate .* tanh.(c)
    end
    
    output = dense_layer(h)
    return output[1:N], output[N+1:2N]
end

function forecast_correlation_matrix(x_history, a, b, N)
    """Forecast correlation matrix using DCC dynamics"""
    n_hist = size(x_history, 2)
    
    if n_hist < 2
        return Matrix{Float32}(I, N, N)
    end
    
    # Compute historical standardized residuals
    returns_hist = x_history
    μ_hist = mean(returns_hist, dims=2)
    σ_hist = std(returns_hist, dims=2)
    z_hist = (returns_hist .- μ_hist) ./ max.(σ_hist, 1e-8)
    
    # Initialize correlation matrix
    Q̄ = (z_hist * z_hist') / n_hist
    d = 1 ./ sqrt.(max.(diag(Q̄), 1e-8))
    Q̄ = Symmetric(diagm(d) * Q̄ * diagm(d))
    
    # Evolve Q using DCC dynamics
    Q = copy(Q̄)
    for t in 2:n_hist
        zt_prev = view(z_hist, :, t-1)
        Q = (1-a-b) .* Q̄ .+ a .* (zt_prev * zt_prev') .+ b .* Q
    end
    
    # Final correlation matrix
    d = 1 ./ sqrt.(max.(diag(Q), 1e-8))
    R = Symmetric(diagm(d) * Q * diagm(d))
    
    return Matrix(R)
end

function rolling_forecast_evaluation(data, model_configs; 
                                   window_size=252, # 1 year rolling window
                                   forecast_horizon=22) # ~1 month ahead
    """Perform rolling out-of-sample forecast evaluation"""
    
    n_obs, N = size(data)
    n_forecasts = n_obs - window_size - forecast_horizon + 1
    
    if n_forecasts <= 0
        error("Not enough data for rolling forecast evaluation. Need at least $(window_size + forecast_horizon) observations, got $n_obs")
    end
    
    println("Performing $n_forecasts rolling forecast evaluations...")
    
    # Storage for results
    forecast_results = Dict{Symbol, Any}()
    
    for (model_type, model_name) in model_configs
        println("Starting rolling forecasts for $model_name...")
        
        return_forecasts = Array{Float32}(undef, N, forecast_horizon, n_forecasts)
        covariance_forecasts = Array{Float32}(undef, N, N, forecast_horizon, n_forecasts)
        actual_returns = Array{Float32}(undef, N, forecast_horizon, n_forecasts)
        
        for i in 1:n_forecasts
            if i % 20 == 0 || i == 1
                println("  Forecast window $i/$n_forecasts")
            end
            
            # Define training window
            train_start = i
            train_end = i + window_size - 1
            forecast_start = train_end + 1
            forecast_end = forecast_start + forecast_horizon - 1
            
            train_data = data[train_start:train_end, :]
            actual_data = data[forecast_start:forecast_end, :]
            
            try
                # Train model on rolling window
                if model_type in [:rnn, :lstm]
                    X_train, Y_train = prepare_sequential_data(train_data)
                else
                    X_train, Y_train = prepare_windowed_data(train_data, 22)
                end
                
                # Quick training with fewer samples for rolling forecast
                bnn, chain, _ = train_model(model_type, X_train, Y_train, N; 
                                          mcmc_samples=3000, chains=3)
                
                # Generate forecasts for each day in forecast horizon
                for h in 1:forecast_horizon
                    if h == 1
                        # Use training data as history
                        if model_type == :windowed
                            x_hist = X_train[:, end:end]  # Last window
                        else
                            x_hist = train_data'  # Full sequence
                        end
                    else
                        # Use actual data up to h-1 for forecasting step h
                        if model_type == :windowed
                            recent_data = vcat(train_data[end-21:end, :], actual_data[1:h-1, :])
                            x_hist = reshape(recent_data[end-21:end, :]', :, 1)
                        else
                            x_hist = vcat(train_data, actual_data[1:h-1, :])'
                        end
                    end
                    
                    returns_h, cov_h = forecast_one_step_ahead(bnn, chain, x_hist, model_type; n_samples=50)
                    
                    return_forecasts[:, h, i] = mean(returns_h, dims=2)
                    covariance_forecasts[:, :, h, i] = mean(cov_h, dims=3)
                    actual_returns[:, h, i] = actual_data[h, :]
                end
                
            catch e
                println("    Error in forecast window $i: $e")
                # Fill with NaN for failed forecasts
                return_forecasts[:, :, i] .= NaN32
                covariance_forecasts[:, :, :, i] .= NaN32
                actual_returns[:, :, i] .= actual_data'
            end
        end
        
        forecast_results[model_type] = Dict(
            "model_name" => model_name,
            "return_forecasts" => return_forecasts,
            "covariance_forecasts" => covariance_forecasts,
            "actual_returns" => actual_returns
        )
    end
    
    return forecast_results
end

function compute_forecast_metrics(forecast_results)
    """Compute comprehensive forecast accuracy metrics"""
    metrics = Dict{Symbol, Any}()
    
    for (model_type, results) in forecast_results
        if results === nothing
            continue
        end
        
        return_pred = results["return_forecasts"]
        return_actual = results["actual_returns"]
        cov_pred = results["covariance_forecasts"]
        
        # Remove NaN forecasts
        valid_mask = .!isnan.(return_pred[1, 1, :])
        if sum(valid_mask) == 0
            println("Warning: No valid forecasts for $model_type")
            continue
        end
        
        return_pred = return_pred[:, :, valid_mask]
        return_actual = return_actual[:, :, valid_mask]
        cov_pred = cov_pred[:, :, :, valid_mask]
        
        # Return forecast metrics
        return_errors = return_pred - return_actual
        
        rmse_returns = sqrt(mean(return_errors.^2))
        mae_returns = mean(abs.(return_errors))
        
        # Individual asset metrics
        asset_rmse = [sqrt(mean(return_errors[i, :, :].^2)) for i in 1:size(return_errors, 1)]
        asset_mae = [mean(abs.(return_errors[i, :, :])) for i in 1:size(return_errors, 1)]
        
        # Horizon-specific metrics
        horizon_rmse = [sqrt(mean(return_errors[:, h, :].^2)) for h in 1:size(return_errors, 2)]
        horizon_mae = [mean(abs.(return_errors[:, h, :])) for h in 1:size(return_errors, 2)]
        
        # Covariance forecast metrics (Frobenius norm)
        cov_errors = Array{Float32}(undef, size(cov_pred, 3), size(cov_pred, 4))
        for h in 1:size(cov_pred, 3), i in 1:size(cov_pred, 4)
            # Realized covariance (using outer product of returns)
            realized_cov = return_actual[:, h, i] * return_actual[:, h, i]'
            pred_cov = cov_pred[:, :, h, i]
            cov_errors[h, i] = norm(pred_cov - realized_cov, 2)  # Frobenius norm
        end
        
        rmse_covariance = sqrt(mean(filter(!isnan, cov_errors).^2))
        mae_covariance = mean(abs.(filter(!isnan, cov_errors)))
        
        # Direction accuracy (% of correct directional predictions)
        direction_pred = sign.(return_pred)
        direction_actual = sign.(return_actual)
        direction_accuracy = mean(direction_pred .== direction_actual)
        
        metrics[model_type] = Dict(
            "model_name" => results["model_name"],
            "rmse_returns" => rmse_returns,
            "mae_returns" => mae_returns,
            "asset_rmse" => asset_rmse,
            "asset_mae" => asset_mae,
            "horizon_rmse" => horizon_rmse,
            "horizon_mae" => horizon_mae,
            "rmse_covariance" => rmse_covariance,
            "mae_covariance" => mae_covariance,
            "direction_accuracy" => direction_accuracy,
            "n_valid_forecasts" => sum(valid_mask)
        )
    end
    
    return metrics
end

function print_forecast_results(metrics)
    """Print comprehensive forecast evaluation results"""
    
    println("OUT-OF-SAMPLE FORECAST EVALUATION RESULTS")
    
    
    # Overall performance table
    println("\n📊 OVERALL FORECAST PERFORMANCE")
    println("┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    println("│ Model               │ RMSE Returns│ MAE Returns │ RMSE Cov    │ Dir. Acc.   │")
    println("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")
    
    for (model_type, m) in metrics
        if m !== nothing
            println("│ $(rpad(m["model_name"], 19)) │ $(lpad(round(m["rmse_returns"], digits=6), 11)) │ $(lpad(round(m["mae_returns"], digits=6), 11)) │ $(lpad(round(m["rmse_covariance"], digits=6), 11)) │ $(lpad(round(m["direction_accuracy"]*100, digits=1), 9))% │")
        end
    end
    println("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")
    
    # Horizon analysis
    println("FORECAST HORIZON ANALYSIS (RMSE)")
    for (model_type, m) in metrics
        if m !== nothing && haskey(m, "horizon_rmse")
            println("\n$(m["model_name"]):")
            for (h, rmse) in enumerate(m["horizon_rmse"][1:min(10, end)])  # Show first 10 days
                println("  Day $h: $(round(rmse, digits=6))")
            end
        end
    end
    
    # Asset-specific performance
    println("\n🎯 ASSET-SPECIFIC PERFORMANCE (RMSE)")
    for (model_type, m) in metrics
        if m !== nothing && haskey(m, "asset_rmse")
            println("\n$(m["model_name"]):")
            for (i, rmse) in enumerate(m["asset_rmse"])
                println("  Asset $i: $(round(rmse, digits=6))")
            end
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#                          UPDATED MAIN ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

function comprehensive_dcc_analysis(data_path::String, n_assets::Int=5)
    """Run comprehensive IN-SAMPLE DCC-GARCH analysis comparing all three approaches"""
    
    
    println("COMPREHENSIVE DCC-GARCH BAYESIAN NEURAL NETWORK ANALYSIS")
    println("Comparing RNN, LSTM, and Windowed Feedforward Approaches")
    
    
    # Load and preprocess data
    df = CSV.read(data_path, DataFrame)
    etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", 
                 "16418", "16421", "16423", "16424", "16426", "16433", "16437", 
                 "16452", "16460", "24697", "27635", "28272", "28273", "28274", 
                 "28275", "28276", "28277", "28278", "28279", "28280", "31372", "31466"]
    
    selected_assets = etf_names[1:n_assets]
    returns, μ_returns = preprocess_financial_data(df, selected_assets)
    y = Float32.(returns)
    
    println("Selected assets: $selected_assets")
    println("Data dimensions: $(size(y)) (observations × assets)")
    println("Mean returns: $(round.(μ_returns, digits=6))")
    
    # Train-test split
    train_size = Int(floor(0.8 * size(y, 1)))
    train_idx = 1:train_size
    
    # Prepare data for different approaches
    X_seq, Y_seq = prepare_sequential_data(y[train_idx, :])  # Sequential data
    X_win, Y_win = prepare_windowed_data(y[train_idx, :], 22)  # Windowed data
    
    println("Sequential data: X=$(size(X_seq)), Y=$(size(Y_seq))")
    println("Windowed data: X=$(size(X_win)), Y=$(size(Y_win))")
    
    # Model configurations
    hidden_size = max(16, min(64, 8 * n_assets))
    models_to_run = [
        (:rnn, X_seq, Y_seq, "Sequential RNN"),
        (:lstm, X_seq, Y_seq, "Sequential LSTM"),
        (:windowed, X_win, Y_win, "Windowed Feedforward")
    ]
    
    results = Dict()
    all_plots = []
    
    # Train and evaluate each model
    for (model_type, X, Y, model_name) in models_to_run
        try
            
            println("TRAINING $model_name")
            
            
            # Train model
            bnn, chain, net = train_model(model_type, X, Y, n_assets; 
                                        hidden_size=hidden_size,
                                        mcmc_samples=20000, chains=12)
            
            # Evaluate model
            result = evaluate_model(bnn, chain, X, Y, model_name)
            results[model_type] = result
            push!(all_plots, result["qq_plot"])
            
            # Print summary
            println("\n$model_name Results:")
            println("  R-hat: $(round(result["r_hat"], digits=4))")
            println("  DCC params (α, β): $(round.(result["dcc_params"], digits=4))")
            println("  QQ MAD: $(round(result["mad_qq"], digits=4))")
            println("  Converged: $(result["converged"] ? "✓" : "✗")")
            
        catch e
            println("Error with $model_name: $e")
            results[model_type] = nothing
        end
    end
    
    # Comparative Analysis
    println("\n" * "="^60)
    println("COMPARATIVE ANALYSIS SUMMARY")
    println("="^60)
    
    # Results table
    println("\n┌─────────────────────┬─────────┬──────────────┬─────────┬────────────┐")
    println("│ Model               │ R-hat   │ DCC (α, β)   │ QQ MAD  │ Converged  │")
    println("├─────────────────────┼─────────┼──────────────┼─────────┼────────────┤")
    
    for (model_type, name) in [(:rnn, "Sequential RNN"), (:lstm, "Sequential LSTM"), 
                               (:windowed, "Windowed FF")]
        if results[model_type] !== nothing
            r = results[model_type]
            α, β = r["dcc_params"]
            println("│ $(rpad(name, 19)) │ $(lpad(round(r["r_hat"], digits=3), 7)) │ $(lpad(round(α, digits=3), 4)), $(lpad(round(β, digits=3), 4)) │ $(lpad(round(r["mad_qq"], digits=4), 7)) │ $(lpad(r["converged"] ? "✓" : "✗", 9))  │")
        end
    end
    println("└─────────────────────┴─────────┴──────────────┴─────────┴────────────┘")
    
    # Create comparison plot
    if length(all_plots) > 1
        comparison_plot = plot(all_plots..., layout=(1, length(all_plots)), 
                             size=(400*length(all_plots), 400))
        plot!(comparison_plot, plot_title="DCC-GARCH Model Comparison ($n_assets Assets)")
        display(comparison_plot)
    end
    
    return results
end

function comprehensive_oos_forecast_analysis(data_path::String, n_assets::Int=3)
    """Run comprehensive out-of-sample forecast analysis"""
    
    
    println("OUT-OF-SAMPLE DCC-GARCH FORECAST ANALYSIS")
    println("Comparing RNN, LSTM, and Windowed Feedforward Approaches")
    
    
    # Load and preprocess data
    df = CSV.read(data_path, DataFrame)
    etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", 
                 "16418", "16421", "16423", "16424", "16426", "16433", "16437", 
                 "16452", "16460", "24697", "27635", "28272", "28273", "28274", 
                 "28275", "28276", "28277", "28278", "28279", "28280", "31372", "31466"]
    
    selected_assets = etf_names[1:n_assets]
    returns, μ_returns = preprocess_financial_data(df, selected_assets)
    
    println("Selected assets: $selected_assets")
    println("Data dimensions: $(size(returns)) (observations × assets)")
    
    # Model configurations
    model_configs = [
        (:rnn, "Sequential RNN"),
        (:lstm, "Sequential LSTM"),
        (:windowed, "Windowed Feedforward")
    ]
    
    # Perform rolling forecast evaluation (reduced parameters for efficiency)
    println("\n🔄 Starting rolling forecast evaluation...")
    forecast_results = rolling_forecast_evaluation(returns, model_configs; 
                                                   window_size=200, forecast_horizon=10)
    
    # Compute forecast metrics
    println("\n📊 Computing forecast accuracy metrics...")
    metrics = compute_forecast_metrics(forecast_results)
    
    # Print results
    print_forecast_results(metrics)
    
    return forecast_results, metrics
end

function run_complete_analysis(data_path::String, n_assets::Int=3)
    """Run both in-sample and out-of-sample analysis"""
    
    println("STARTING COMPLETE DCC-GARCH ANALYSIS")

    
    # Run in-sample analysis
    println("PHASE 1: IN-SAMPLE ANALYSIS")
    insample_results = comprehensive_dcc_analysis(data_path, n_assets)
    
    
    
    # Run out-of-sample analysis  
    println(" PHASE 2: OUT-OF-SAMPLE FORECASTING")
    oos_results, oos_metrics = comprehensive_oos_forecast_analysis(data_path, n_assets)
    
    println("COMPLETE ANALYSIS FINISHED!")
    
    return insample_results, oos_results, oos_metrics
end


# Set your data path
data_path = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"

println("DCC-GARCH Bayesian Neural Network Suite with Out-of-Sample Forecasting Ready!")
println("1. In-sample analysis only:")
println("   results = comprehensive_dcc_analysis(data_path, 5)")
println("\n2. Out-of-sample forecasting only:")
println("   forecast_results, metrics = comprehensive_oos_forecast_analysis(data_path, 3)")
println("\n3. Complete analysis (both in-sample and out-of-sample):")
println("   insample, oos_results, oos_metrics = run_complete_analysis(data_path, 3)")

insample_results, oos_results, oos_metrics = run_complete_analysis(data_path, 3)