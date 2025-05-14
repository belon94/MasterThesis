# ------------------------------------------------------------------
# DCC-GARCH Normal likelihood for Bayesian neural networks
# ------------------------------------------------------------------
using LinearAlgebra, Statistics, Distributions, Random
using Flux
using BayesFlux

"""
    struct DCCGarchNormal

Likelihood for a Bayesian neural network that outputs
    – conditional means μₜ (length N)
    – log-variances log σ²ₜ (length N)

and wraps them in a Dynamic Conditional Correlation-GARCH model with
normal errors. Two extra parameters (a,b) control the DCC recursion.
"""
struct DCCGarchNormal{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int          # here: raw_a, raw_b
    nc::NetConstructor{T,F}
    prior_μ::D                    # prior on the network's mean head
    N::Int                        # dimension of yₜ
end

DCCGarchNormal(nc::NetConstructor{T,F}, prior_μ::D, N::Int) where {T,F,D<:Distribution} =
    DCCGarchNormal{T,F,D}(2, nc, prior_μ, N)

# ------------------- helpers --------------------------------------------------
sigmoid(x) = 1 / (1 + exp(-x))

@inline function transform_ab(raw_a, raw_b)
    a = sigmoid(raw_a)
    b = sigmoid(raw_b) * (1 - a)      # guarantees a+b < 1
    return a, b
end

"""
Return the nearest symmetric positive-definite matrix (Higham 1988).
Safe for AD — no branching on factorisation success.
"""
function nearest_pd(A)
    B  = (A + A') / 2
    F  = eigen(B)
    Dp = Diagonal(max.(F.values, 1e-10))
    return F.vectors * Dp * F.vectors'
end
# ------------------------------------------------------------------------------

# More specific method signature using Matrix{T} instead of AbstractMatrix{T}
function (l::DCCGarchNormal{T,F,D})(x::Matrix{T},
                                   y::Matrix{T},
                                   θnet::AbstractVector,
                                   θlike::AbstractVector) where {T,F,D}
    θnet, θlike = T.(θnet), T.(θlike)
    
    # Validate input dimensions - batch dimension is the second dimension
    N, Tsteps = l.N, size(y, 2)
    @assert size(y, 1) == N "Output dimension mismatch: expected $N, got $(size(y, 1))"
    @assert size(x, 2) == Tsteps "Time steps mismatch: expected $Tsteps, got $(size(x, 2))"

    # DCC parameters
    raw_a, raw_b = θlike
    a, b         = transform_ab(raw_a, raw_b)

    # build network from flat parameter vector
    net = l.nc(θnet)

    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)            # std. devs.
    z = similar(μ)            # standardised residuals

    @inbounds for t in 1:Tsteps
        out = net(x[:, t])        # Pass the entire column as input
        @assert length(out) == 2N "Network must output 2N scalars, got $(length(out))"

        μ[:, t] .= out[1:N]
        σ[:, t] .= exp.(out[N+1:2N] ./ 2)  # σ = exp(½ log σ²)
        z[:, t] .= (y[:, t] .- μ[:, t]) ./ σ[:, t]
    end

    # unconditional correlation Q̄
    Qbar = Tsteps > 1 ? (z * z') / Tsteps : I(N)
    if Tsteps > 1
        d_inv = 1 ./ sqrt.(diag(Qbar))
        Qbar  = Symmetric(Diagonal(d_inv) * Qbar * Diagonal(d_inv))
    end
    Q = copy(Qbar)

    logl = zero(T)

    for t in 1:Tsteps
        # correlation Rₜ from Qₜ
        d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
        R     = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))

        # conditional covariance Hₜ
        Dt = Diagonal(σ[:, t])
        Ht = nearest_pd(Dt * R * Dt)
        Hchol = cholesky(Ht)

        diff  = y[:, t] .- μ[:, t]
        quad  = sum(abs2, Hchol.L \ diff)

        # Log-likelihood contribution
        logl += -0.5T * (N*log(2π) + 2*sum(log, diag(Hchol.L)) + quad)

        # recursive update of Q (skip after last obs)
        if t < Tsteps
            Q = (1 - a - b) .* Qbar .+ a .* (z[:, t] * z[:, t]') .+ b .* Q
        end
    end

    # priors: on means AND on (a,b)
    logl += sum(logpdf.(l.prior_μ, μ)) +
            logpdf(Beta(20, 1.5), a) +
            logpdf(Beta(20, 1.5), b)

    return logl         # log-posterior kernel
end

# ------------------------------------------------------------------
# Posterior-predictive draw
# ------------------------------------------------------------------
# More specific method signature using Matrix{T}
function posterior_predict(l::DCCGarchNormal{T,F,D},
                           x::Matrix{T},
                           θnet::AbstractVector,
                           θlike::AbstractVector,
                           y_hist::Matrix{T}) where {T,F,D}

    θnet, θlike = T.(θnet), T.(θlike)
    a, b        = transform_ab(θlike...)

    net      = l.nc(θnet)
    N, Tsteps = l.N, size(x, 2)

    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)
    z = similar(μ)

    @inbounds for t in 1:Tsteps
        out = net(x[:, t])
        μ[:, t] .= out[1:N]
        σ[:, t] .= exp.(out[N+1:2N] ./ 2)
        z[:, t] .= (y_hist[:, t] .- μ[:, t]) ./ σ[:, t]
    end

    # unconditional correlation Q̄
    Qbar = Tsteps > 1 ? (z * z') / Tsteps : I(N)
    if Tsteps > 1
        d_inv = 1 ./ sqrt.(diag(Qbar))
        Qbar  = Symmetric(Diagonal(d_inv) * Qbar * Diagonal(d_inv))
    end
    Q = copy(Qbar)

    # run DCC recursion up to the end of the sample
    for t in 1:(Tsteps-1)
        Q = (1 - a - b) .* Qbar .+ a .* (z[:, t] * z[:, t]') .+ b .* Q
    end
    Q = (1 - a - b) .* Qbar .+ a .* (z[:, Tsteps] * z[:, Tsteps]') .+ b .* Q

    # correlation R_T
    d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
    R     = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))

    μp = μ[:, end]
    Hp = nearest_pd(Diagonal(σ[:, end]) * R * Diagonal(σ[:, end]))

    return rand(MvNormal(μp, Hp))
end

# More specific helper for BNN posterior prediction
function posterior_predict(bnn::BNN{L}, x_new::Matrix{T}, params) where {T<:AbstractFloat, L<:DCCGarchNormal}
    y_hist = bnn.y  # Use historical data from the BNN
    θnet = params.θnet
    θlike = params.θlike
    return posterior_predict(bnn.like, x_new, θnet, θlike, y_hist)
end

# --------------------------------------------------------------
# Test script for the DCC-GARCH BNN
# --------------------------------------------------------------

# Set random seed for reproducibility
Random.seed!(123)

# Define test parameters
N = 2       # Dimension of time series (number of assets)
Tsteps = 100 # Number of time steps
lookback = 10 # Number of lookback periods for the network

# Create synthetic data
# Simulate a simple DCC-GARCH process
function simulate_dcc_garch(N, Tsteps, a=0.1, b=0.8)
    # Initialize
    y = zeros(N, Tsteps)
    μ = zeros(N, Tsteps)
    σ = ones(N, Tsteps)
    z = randn(N, Tsteps)
    
    # Simple correlation structure
    R_base = fill(0.5, N, N)
    for i in 1:N
        R_base[i, i] = 1.0
    end
    
    # Initialize Q with unconditional value
    Q = copy(R_base)
    
    # Generate the series
    for t in 2:Tsteps
        # Simple constant mean
        μ[:, t] = [0.001, 0.002]  # Different means for each series
        
        # GARCH(1,1) volatility (simplified)
        for i in 1:N
            σ[i, t] = sqrt(0.01 + 0.1*(y[i, t-1] - μ[i, t-1])^2 + 0.8*σ[i, t-1]^2)
        end
        
        # Update Q (DCC recursion)
        if t > 2
            Q = (1 - a - b) * R_base + a * (z[:, t-1] * z[:, t-1]') + b * Q
        end
        
        # Get correlation matrix from Q
        d_inv = 1 ./ sqrt.(diag(Q))
        R = Diagonal(d_inv) * Q * Diagonal(d_inv)
        
        # Generate standardized residuals
        ε = randn(N)
        
        # Convert R to Cholesky for multiplication
        R_chol = cholesky(Symmetric(R)).L
        
        # Generate correlated residuals
        z[:, t] = R_chol * ε
        
        # Generate returns
        y[:, t] = μ[:, t] + σ[:, t] .* z[:, t]
    end
    
    return y, μ, σ
end

# Prepare input tensors for BNN - fixed dimension order
function prepare_input_tensor(y, lookback)
    N, Tsteps = size(y)
    num_samples = Tsteps - lookback
    
    # Create inputs with batch dimension as the second dim
    x = zeros(Float32, lookback * N, num_samples)
    y_out = zeros(Float32, N, num_samples)
    
    for t in 1:num_samples
        # Flatten the window into a column vector
        x[:, t] = vec(y[:, t:(t+lookback-1)])
        y_out[:, t] = y[:, t+lookback]
    end
    
    return x, y_out
end

# Main execution script
function run_dcc_garch_bnn()
    println("Simulating DCC-GARCH data...")
    y, true_μ, true_σ = simulate_dcc_garch(N, Tsteps)
    println("Data simulation complete.")

    println("Preparing data for BNN...")
    x, y_train = prepare_input_tensor(y, lookback)
    println("Data preparation complete. x shape: ", size(x), ", y shape: ", size(y_train))

    # Define a neural network (now with inputs that match our format)
    function create_network(n_in, n_hidden, n_out)
        return Chain(
            Dense(n_in, n_hidden, tanh),
            Dense(n_hidden, n_hidden, tanh),
            Dense(n_hidden, n_out)
        )
    end

    # Network parameters
    n_in = lookback * N  # Flattened input size
    n_hidden = 16
    n_out = 2 * N        # Means and log-variances

    # Create the network
    net = create_network(n_in, n_hidden, n_out)
    nc = destruct(net)
    println("Network creation complete.")

    # Set prior for means (Normal with zero mean and unit variance)
    prior_μ = Normal(0.0f0, 1.0f0)

    # Create the DCC-GARCH likelihood
    println("Creating DCC-GARCH likelihood...")
    likelihood = DCCGarchNormal(nc, prior_μ, N)
    println("Likelihood creation complete.")

    # Create a prior for network parameters
    println("Setting up priors and initializers...")
    prior_net = BayesFlux.GaussianPrior(nc, 0.5f0)

    # Create an initializer
    init = InitialiseAllSame(Normal(0.0f0, 0.5f0), likelihood, prior_net)

    # Create the BNN
    bnn = BNN(x, y_train, likelihood, prior_net, init)
    println("BNN created successfully!")

    # First find the MAP estimate
    println("\nFinding MAP estimate...")
    opt = FluxModeFinder(bnn, Flux.ADAM(0.01))
    θmap = find_mode(bnn, 10, 500, opt)
    
    # Test the model with MAP estimate
    println("\nTesting MAP estimate...")
    netmap = nc(θmap.θnet)
    mapout = [netmap(x[:, t]) for t in 1:size(x, 2)]
    μmap = hcat([out[1:N] for out in mapout]...)
    σmap = hcat([exp.(out[N+1:2N] ./ 2) for out in mapout]...)
    
    # Calculate RMSE for means
    rmse_μ = sqrt(mean(abs2, y_train .- μmap))
    println("RMSE for means: ", rmse_μ)

    # MCMC sampling
    println("\nRunning MCMC sampling...")
    
    # Configure SGLD sampler
    sampler = SGLD(stepsize_a=0.01f0, stepsize_b=0.0f0, stepsize_γ=0.55f0)
    
    # Run MCMC chain (burnin + samples)
    burnin = 1000
    samples = 5000
    chain = mcmc(bnn, 10, burnin + samples, sampler)
    
    # Discard burnin
    chain = chain[:, burnin+1:end]
    println("MCMC complete: collected $(size(chain, 2)) samples")

    # Make predictions on test data (one-step ahead forecast)
    println("\nMaking predictions with MCMC samples...")
    
    # Use the last time window for prediction
    test_x = x[:, end:end]  # Keep column structure
    test_y_hist = y_train[:, end:end]
    
    # Make predictions using multiple MCMC samples
    n_pred_samples = 100
    pred_indices = rand(1:size(chain, 2), n_pred_samples)
    
    # Convert chain structure to match what posterior_predict expects
    predictions = []
    for i in pred_indices
        sample = (θnet = chain.θnet[:, i], θlike = chain.θlike[:, i])
        push!(predictions, posterior_predict(bnn, test_x, sample))
    end
    
    # Convert predictions to array
    pred_array = hcat(predictions...)
    
    # Calculate prediction stats
    pred_mean = mean(pred_array, dims=2)[:]
    pred_std = std(pred_array, dims=2)[:]
    pred_lower = [quantile(pred_array[i,:], 0.025) for i in 1:N]
    pred_upper = [quantile(pred_array[i,:], 0.975) for i in 1:N]
    
    println("Prediction means: ", pred_mean)
    println("Prediction 95% CI: [", pred_lower, ", ", pred_upper, "]")
    println("Actual values: ", y[:, end])
    
    # Return results
    return Dict(
        "bnn" => bnn,
        "chain" => chain,
        "map_estimate" => θmap,
        "predictions" => pred_array,
        "pred_stats" => (mean=pred_mean, std=pred_std, lower=pred_lower, upper=pred_upper),
        "actual" => y[:, end]
    )
end

# Run the model
results = run_dcc_garch_bnn()
println("\nExecution complete!")