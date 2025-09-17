using Flux
using BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors
Random.seed!(1212)

# Function to generate multivariate DCC-GARCH process
function dccGarchNDraws(n::Int, N::Int, α::Float64, β::Float64, vol::Vector{Float64}, ρ_unconditional::Float64)
    # Initialize parameters
    ω = vol .* (1 .- α .- β)  # GARCH constant terms
    
    # Create unconditional correlation matrix
    Rbar = Matrix{Float64}(I, N, N)
    for i in 1:N
        for j in (i+1):N
            Rbar[i,j] = ρ_unconditional
            Rbar[j,i] = ρ_unconditional
        end
    end
    
    # Ensure Rbar is positive definite
    eig_vals = eigvals(Rbar)
    if minimum(eig_vals) <= 0
        # Adjust the correlation value to make it positive definite
        println("Adjusting initial correlation matrix to ensure positive definiteness")
        ρ_adjusted = max(0.0, min(0.99, ρ_unconditional * 0.9))
        Rbar = Matrix{Float64}(I, N, N)
        for i in 1:N
            for j in (i+1):N
                Rbar[i,j] = ρ_adjusted
                Rbar[j,i] = ρ_adjusted
            end
        end
    end
    
    # Initialize series
    L = zeros(n, N)  # Returns
    σ_2 = zeros(n+1, N)  # Conditional variances
    σ_2[1,:] .= vol
    Qt = copy(Rbar)  # Initial Q matrix
    Rt = copy(Rbar)  # Initial correlation matrix (ensuring it's PD)
    z = zeros(n, N)  # Standardized residuals
    
    # Generate data
    for t in 1:n
        # Calculate correlation matrix R_t
        if t > 1
            # Update Q matrix using DCC recursion
            outer_z = z[t-1,:] * z[t-1,:]'
            # Force symmetry
            outer_z = (outer_z + outer_z')/2
            
            Qt = (1-α-β) * Rbar + α * outer_z + β * Qt
            
            # Force symmetry after update
            Qt = (Qt + Qt')/2
        end
        
        # Calculate correlation matrix from Q
        d_inv = 1.0 ./ sqrt.(max.(diag(Qt), 1e-12))
        Rt_raw = Diagonal(d_inv) * Qt * Diagonal(d_inv)
        
        # Force symmetry and ensure positive definiteness
        Rt_raw = (Rt_raw + Rt_raw')/2
        
        # Use nearest positive definite method if needed - avoiding try/catch
        eigen_decomp = eigen(Symmetric(Rt_raw))
        pos_values = max.(eigen_decomp.values, 1e-5)
        Rt_fix = eigen_decomp.vectors * Diagonal(pos_values) * eigen_decomp.vectors'
        
        # Rescale to ensure diagonal is 1
        d_diag = sqrt.(diag(Rt_fix))
        Rt_fix = Diagonal(1.0 ./ d_diag) * Rt_fix * Diagonal(1.0 ./ d_diag)
        
        # Final symmetry enforcement
        Rt_fix = (Rt_fix + Rt_fix')/2
        
        # Use fixed correlation matrix
        std_noise = rand(MvNormal(zeros(N), Symmetric(Rt_fix)))
        Rt = Rt_fix
        
        # Compute returns
        for i in 1:N
            L[t,i] = sqrt(σ_2[t,i]) * std_noise[i]
            z[t,i] = std_noise[i]  # Save standardized residuals for next iteration
        end
        
        # Update conditional variances for next period
        for i in 1:N
            σ_2[t+1,i] = ω[i] + α * L[t,i]^2 + β * σ_2[t,i]
        end
    end
    
    # Return output
    output = Dict(
        "L" => L,  # Returns
        "sigma_squared" => σ_2[1:n,:],  # Conditional variances
        "next_sigma" => σ_2[n+1,:],  # Next period forecast
        "R" => Rt  # Final correlation matrix
    )
    
    return output
end

# Define the DCCGarchNormal likelihood function
struct DCCGarchNormal{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int  # For DCC parameters (a,b)
    nc::NetConstructor{T,F}
    prior_μ::D
    N::Int  # Dimension of multivariate output
end

function DCCGarchNormal(nc::NetConstructor{T,F}, prior_μ::D, N::Int) where {T,F,D<:Distributions.Distribution}
    return DCCGarchNormal(2, nc, prior_μ, N)  # 2 params: a and b
end

# Helper function for nearest positive definite matrix
function nearest_pd(A)
    # Compute symmetric part
    B = (A + A') / 2 
    
    # Eigendecomposition
    F = eigen(B)
    
    # Replace negative eigenvalues with small positive values
    D = 1e-4 * I
    
    # Reconstruct matrix
    return B+D
end

# Helper function to transform DCC parameters to valid range
function transform_ab(raw_a, raw_b)
    a = sigmoid(raw_a)
    b = sigmoid(raw_b) * (1 - a)
    return a, b
end

function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

# COMPLETELY REVISED: Data preparation function for BayesFlux
function prepare_time_series_data(data, window_size)
    n_samples = size(data, 1) - window_size
    n_features = size(data, 2)
    
    # BayesFlux expects:
    # x with shape [features, samples]
    # y with shape [output_dims, samples]
    
    # For time series with window, we'll flatten the window into features
    x = Array{Float32}(undef, n_features * window_size, n_samples)
    y = Array{Float32}(undef, n_features, n_samples)
    
    for i in 1:n_samples
        # Get the window and flatten it
        window = data[i:(i+window_size-1), :]
        # Flatten window row by row
        x_flattened = reshape(transpose(window), :)
        # Store flattened window
        x[:, i] = x_flattened
        # Store target
        y[:, i] = data[i+window_size, :]
    end
    
    return x, y
end

# Modified implementation of the likelihood function for flattened input
function (l::DCCGarchNormal{T,F,D})(x::Matrix{T}, y::Matrix{T}, 
                                   θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)
    
    # Transform DCC parameters to valid range
    a, b = transform_ab(θlike[1], θlike[2])
    
    # Initialize network
    net = l.nc(θnet)
    
    # Dimensions
    N, Tsteps = l.N, size(x, 2)
    
    # Initialize arrays for means, standard deviations and standardized residuals
    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)  # standard deviation instead of variance
    z = similar(μ)
    
    # Extract parameters and compute standardized residuals for each time step
    @inbounds for t in 1:Tsteps
        # Get the flattened window for this sample
        xt_flat = x[:, t]
        
        # Get network output
        out = net(xt_flat)     
        length(out) == 2N || error("Network output must be 2N scalars.")
        
        # Process output
        μ[:, t] .= out[1:N]       # Means
        σ[:, t] .= exp.(out[N+1:2N] ./ 2)  # Standard deviations (exp of half-log-variance)
        z[:, t] .= (y[:, t] .- μ[:, t]) ./ σ[:, t]  # Standardized residuals
    end
    
    # Compute unconditional correlation matrix
    Qbar = Tsteps > 1 ? (z * z') / Tsteps : I(N)
    
    # Ensure Qbar is correlation matrix
    if Tsteps > 1
        d_inv = 1 ./ sqrt.(diag(Qbar))
        Qbar = Symmetric(Diagonal(d_inv) * Qbar * Diagonal(d_inv))
    end
    
    # Initialize Q1 as unconditional correlation
    Q = copy(Qbar)
    
    # Initialize log-likelihood
    logl = zero(T)
    
    # Process first time step separately
    t = 1
    
    # Compute correlation matrix R_t from Q
    d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
    R = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
    
    # Construct covariance matrix H_t
    Dt = Diagonal(σ[:, t])  # Using standard deviation directly
    H_t = Dt * R * Dt
    
    # Always ensure H_t is positive definite 
    H_t = nearest_pd(H_t)
    Hchol = cholesky(H_t)
    
    # Compute log-likelihood contribution
    diff = y[:, t] .- μ[:, t]
    quad = sum(abs2, Hchol.L \ diff)
    
    logl -= 0.5 * (N * log(2π) + 2*sum(log, diag(Hchol.L)) + quad)
    
    # Update Q for second time step using first observation
    if Tsteps > 1
        Q_next = (1-a-b).*Qbar .+ a.*(z[:,1]*z[:,1]') .+ b.*Q
        Q = Q_next
    end
    
    # DCC recursion for remaining time steps
    for t in 2:Tsteps
        # Compute correlation matrix R_t from Q_t (which uses information up to t-1)
        d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
        R = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
        
        # Construct covariance matrix H_t
        Dt = Diagonal(σ[:, t])  # Using standard deviation directly
        H_t = Dt * R * Dt
        
        # Always ensure H_t is positive definite - no try/catch for Zygote compatibility
        H_t = nearest_pd(H_t)
        Hchol = cholesky(H_t)
        
        # Compute log-likelihood contribution
        diff = y[:, t] .- μ[:, t]
        quad = sum(abs2, Hchol.L \ diff)
        
        logl -= 0.5 * (N * log(2π) + 2*sum(log, diag(Hchol.L)) + quad)
        
        # Update Q for next time step using current observation
        if t < Tsteps
            Q_next = (1-a-b).*Qbar .+ a.*(z[:,t]*z[:,t]') .+ b.*Q
            Q = Q_next
        end
    end
    
    # Add prior contribution
    logl += sum(logpdf.(l.prior_μ, μ))
    
    return logl
end

# Modified posterior prediction implementation
function posterior_predict(l::DCCGarchNormal{T,F,D}, 
                          x::Matrix{T}, 
                          θnet::AbstractVector, 
                          θlike::AbstractVector,
                          y_hist::Matrix{T}) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)
    
    # Transform parameters to valid range
    a, b = transform_ab(θlike[1], θlike[2])
    
    # Initialize network
    net = l.nc(θnet)
    
    # Dimensions
    N, Tsteps = l.N, size(x, 2)
    
    # Process historical data for DCC recursion
    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)  # standard deviation instead of variance
    z = similar(μ)
    
    @inbounds for t in 1:Tsteps
        # Get the flattened window for this sample
        xt_flat = x[:, t]
        
        # Get network output
        out = net(xt_flat)
        length(out) == 2N || error("Network output must be 2N scalars.")
        
        # Process output
        μ[:, t] .= out[1:N]
        σ[:, t] .= exp.(out[N+1:2N] ./ 2)  # Standard deviations (exp of half-log-variance)
        z[:, t] .= (y_hist[:, t] .- μ[:, t]) ./ σ[:, t]  # Standardized residuals
    end
    
    # Compute unconditional correlation matrix
    Qbar = Tsteps > 1 ? (z * z') / Tsteps : I(N)
    
    # Ensure Qbar is correlation matrix
    if Tsteps > 1
        d_inv = 1 ./ sqrt.(diag(Qbar))
        Qbar = Symmetric(Diagonal(d_inv) * Qbar * Diagonal(d_inv))
    end
    
    # Initialize Q1 as unconditional correlation
    Q = copy(Qbar)
    
    
    # Proper DCC recursion with temporal ordering
    if Tsteps > 1
        # Process observations sequentially
        for t in 1:(Tsteps-1)
            Q_next = (1-a-b).*Qbar .+ a.*(z[:,t]*z[:,t]') .+ b.*Q
            Q = Q_next
        end
        
        # Final update with last observation
        Q = (1-a-b).*Qbar .+ a.*(z[:,Tsteps]*z[:,Tsteps]') .+ b.*Q
    end
    
    # Compute correlation matrix R_t
    d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
    R = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
    
    # Means and variance for prediction (from last time step)
    μp = μ[:, end]
    Hp = Diagonal(σ[:, end]) * R * Diagonal(σ[:, end])  # Using standard deviation directly
    
    # Make sure Hp is positive definite - no try/catch for Zygote compatibility
    Hp = nearest_pd(Hp)
    
    # Generate prediction
    return rand(MvNormal(μp, Hp))
end

# Now set up the model parameters and run the simulation
# Sample size
n = 500
N = 2  # Number of series

# More conservative DCC parameters
α = 0.05  # DCC parameter
β = 0.90  # DCC parameter
vol = [(20^2)/252, (15^2)/252]  # Different volatilities for each series
ρ_unconditional = 0.5  # Unconditional correlation

# Simulate DCC-GARCH process
simulated_DccGarch = dccGarchNDraws(n, N, α, β, vol, ρ_unconditional)

# Extract data
y = Float32.(simulated_DccGarch["L"])
simulated_σ = sqrt.(Float32.(simulated_DccGarch["sigma_squared"]))

# Split into training and testing sets
train_index = 1:400
test_index = 401:500
full_index = 1:500

y_train = y[train_index, :]
y_test = y[test_index, :]
y_full = y[full_index, :]

# Use window size of 5
window_size = 5

# COMPLETELY REVISED: Create data for BNN in the format it expects
x_train, y_train_target = prepare_time_series_data(y_train, window_size)

# Changed network architecture to work with flattened input
# Input size is now N * window_size (features * time steps)
net = Chain(
    Dense(N * window_size, 20, relu),  # First layer handles flattened input
    Dense(20, 20, relu),               # Hidden layer
    Dense(20, 2*N)                     # Output layer: mean and log-variance for each series
)

nc = destruct(net)

# Use DCCGarchNormal likelihood
like = DCCGarchNormal(nc, Normal(0, 0.5), N)
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)

# Create BNN
bnn = BNN(x_train, y_train_target, like, prior, init)

# Find MAP estimate
opt = FluxModeFinder(bnn, Flux.ADAM())
θmap = find_mode(bnn, 2, 500, opt)

# Get predictions from MAP estimate
nethat = nc(θmap)

# Function to extract predictions from the model
function extract_predictions(net, x)
    n_features = size(net.layers[end].weight, 1) ÷ 2
    n_samples = size(x, 2)
    
    means = zeros(Float32, n_samples, n_features)
    sigmas = zeros(Float32, n_samples, n_features)
    
    for i in 1:n_samples
        out = net(x[:, i])
        means[i, :] = out[1:n_features]
        sigmas[i, :] = exp.(out[n_features+1:2*n_features] ./ 2)
    end
    
    return means, sigmas
end

# Get predictions for training set
means_train, sigmas_train = extract_predictions(nethat, x_train)

# Create test sequences
x_test, y_test_target = prepare_time_series_data(y_test, window_size)

# Get predictions for test set
means_test, sigmas_test = extract_predictions(nethat, x_test)

# Prepare indices for plotting (accounting for window size)
train_plot_indices = (window_size+1):length(y_train)

# Plot results
p1 = plot(train_plot_indices, y_train[train_plot_indices, 1], label="Actual Returns (Train)", color=:blue, 
          title="Series 1", xlabel="Time", ylabel="Returns")
plot!(p1, train_plot_indices, means_train[:, 1], label="Predicted Mean (Train)", color=:red)
plot!(p1, train_plot_indices, [means_train[:, 1] + 2*sigmas_train[:, 1], means_train[:, 1] - 2*sigmas_train[:, 1]], 
      label=["95% CI" nothing], color=:red, alpha=0.3, linestyle=:dash)

p2 = plot(train_plot_indices, y_train[train_plot_indices, 2], label="Actual Returns (Train)", color=:blue, 
          title="Series 2", xlabel="Time", ylabel="Returns")
plot!(p2, train_plot_indices, means_train[:, 2], label="Predicted Mean (Train)", color=:red)
plot!(p2, train_plot_indices, [means_train[:, 2] + 2*sigmas_train[:, 2], means_train[:, 2] - 2*sigmas_train[:, 2]], 
      label=["95% CI" nothing], color=:red, alpha=0.3, linestyle=:dash)

plot(p1, p2, layout=(2,1), size=(800, 600))


############ BNN-RNN ##########################


###############################################################################
#  DCC–GARCH Bayesian RNN example (BayesFlux.jl) - CORRECTED                 #
###############################################################################
using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics

Random.seed!(1212)

# ───────────────────────── 1.  DCC-GARCH simulator ──────────────────────────
function dccGarchNDraws(n::Int, N::Int, α::Float64, β::Float64,
                        vol::Vector{Float64}, ρ::Float64)

    ω = vol .* (1 .- α .- β)

    R̄ = Matrix{Float64}(I,N,N)
    for i in 1:N, j in i+1:N
        R̄[i,j] = R̄[j,i] = ρ
    end
    if minimum(eigvals(R̄)) ≤ 0
        ρ_adj = max(0.0, min(0.99, ρ*0.9))
        for i in 1:N, j in i+1:N
            R̄[i,j] = R̄[j,i] = ρ_adj
        end
    end

    L   = zeros(n,N)
    σ²  = vcat(vol', zeros(n,N))
    Q   = copy(R̄)
    z   = zeros(n,N)

    for t in 1:n
        if t>1
            Q = (1-α-β)*R̄ + α*(z[t-1,:]*z[t-1,:]') + β*Q |> Symmetric
        end
        d  = 1 ./ sqrt.(diag(Q))
        R  = Symmetric(diagm(d)*Q*diagm(d))

        ε          = rand(MvNormal(zeros(N),R))
        L[t,:]     = sqrt.(σ²[t,:]) .* ε
        z[t,:]     = ε
        σ²[t+1,:]  = ω .+ α .* L[t,:].^2 .+ β .* σ²[t,:]
    end
    return Dict("L"=>L, "sigma_squared"=>σ²[1:end-1,:])
end

# ───────────────────────── 2.  helpers ──────────────────────────────────────
sigmoid(x) = 1/(1+exp(-x))
transform_ab(a,b) = let a_=sigmoid(a); b_=sigmoid(b)*(1-a_); (a_,b_) end
nearest_pd(A) = (A + A')/2 + 1e-4I             # quick PD repair

# Data preparation for RNN - sequences format
function prepare_time_series_data_rnn(data, w)
    n = size(data,1) - w
    p = size(data,2)
    
    # Create sequences: each column is a sequence of length w
    X = Array{Float32}(undef, p * w, n)  # Flattened sequences
    Y = Array{Float32}(undef, p, n)
    
    for i in 1:n
        # Flatten the sequence for this sample
        sequence = data[i:i+w-1, :]'  # (features, time)
        X[:, i] = vec(sequence)       # Flatten to vector
        Y[:, i] = data[i+w, :]
    end
    return X, Y
end

# ─────────────────── 3.  custom DCC-GARCH RNN likelihood (FIXED) ─────────────
struct DCCGarchRNN{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
    window::Int
end
DCCGarchRNN(nc::NetConstructor{T,F}, prior::D, N::Int, window::Int) where {T,F,D<:Distribution} =
    DCCGarchRNN{T,F,D}(2, nc, prior, N, window)

function (ℓ::DCCGarchRNN)(x::Matrix{T}, y::Matrix{T},
                         θnet::AbstractVector, θlike::AbstractVector) where {T}
    
    θnet, θlike = T.(θnet), T.(θlike)
    a, b = transform_ab(θlike...)
    net = ℓ.nc(θnet)
    N, window, Tsteps = ℓ.N, ℓ.window, size(x,2)

    # Process each sequence independently using map (no mutations)
    outs = map(1:Tsteps) do t
        # Reshape sequence for RNN processing
        seq_input = reshape(view(x, :, t), N, window)
        
        # Manual RNN forward pass to avoid state mutations
        rnn_layer = net[1]
        dense_layer = net[2]
        
        # Initialize hidden state
        h = zeros(T, size(rnn_layer.cell.Wi, 1))
        
        # Process sequence step by step
        for step in 1:window
            input_step = view(seq_input, :, step)
            h = tanh.(rnn_layer.cell.Wi * input_step + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
        end
        
        # Use final hidden state for prediction
        final_output = dense_layer(h)
        return final_output
    end
    
    μ = hcat(map(o -> o[1:N], outs)...)
    logσ2 = hcat(map(o -> o[N+1:2N], outs)...)
    σ = exp.(logσ2 ./ 2)
    z = (y .- μ) ./ σ

    Q̄ = Tsteps>1 ? (z*z')/Tsteps : Matrix{T}(I,N,N)
    d = 1 ./ sqrt.(diag(Q̄))
    Q̄ = Symmetric(diagm(d)*Q̄*diagm(d))

    Q, logl = Q̄, zero(T)
    for t in 1:Tsteps
        d = 1 ./ sqrt.(max.(diag(Q),1e-10))
        R = Symmetric(diagm(d)*Q*diagm(d))
        D = Diagonal(view(σ,:,t))
        H = nearest_pd(D*R*D)
        L = cholesky(Symmetric(H)).L

        diff = view(y,:,t) .- view(μ,:,t)
        quad = sum(abs2, L \ diff)
        logl -= 0.5*(N*log(2π) + 2*sum(log,diag(L)) + quad)

        if t < Tsteps
            zt = view(z,:,t)
            Q = (1-a-b).*Q̄ .+ a.*(zt*zt') .+ b.*Q
        end
    end
    logl += sum(logpdf.(ℓ.prior, vec(μ)))
    return logl
end

# ───────────────────────── 4.  simulate data ────────────────────────────────
n, N = 500, 2  # Number of observations and variables
α, β = 0.05, 0.90  # DCC-GARCH parameters
vol = [(20^2)/252, (15^2)/252]  # Initial volatilities (annualized)
ρ = 0.5  # Correlation coefficient

# Generate simulated data
sim = dccGarchNDraws(n, N, α, β, vol, ρ)
y = Float32.(sim["L"])  # Returns

# Define training set
train_idx = 1:400  # Use first 400 observations for training
window = 22     # Use 22 lags as requested
Xtr, Ytr = prepare_time_series_data_rnn(y[train_idx,:], window)

# ─────────────────────── 5.  build & train RNN BNN ────────────────────────────
# Define RNN architecture
hidden_size = 20
net = Chain(
    RNN(N => hidden_size),       # RNN layer: N inputs to hidden_size outputs
    Dense(hidden_size => 2N)     # Output layer: mean and log variance for each series
)

# Setup BNN components
nc = destruct(net)  # Network constructor
like = DCCGarchRNN(nc, Normal(0, 0.5), N, window)  # Custom RNN likelihood
prior = GaussianPrior(nc, 0.5f0)  # Prior distribution
init = InitialiseAllSame(Normal(0f0, 0.5f0), like, prior)  # Parameter initialization
bnn = BNN(Xtr, Ytr, like, prior, init)  # BNN model

# Find mode of posterior for initialization
θmap = find_mode(bnn, 50, 500, FluxModeFinder(bnn, Flux.ADAM()))

# ─────────────────────── 6.  posterior sampling (SGNHTS) ─────────────────────
# Smaller step size for better convergence
sampler = SGNHTS(1f-4, 1f0; xi = 1f0^2, μ = 10f0)
ch = mcmc(bnn, 10, 50_000, sampler)
# Keep only the last 20,000 samples (discard 30,000 as burn-in)
ch = ch[:, end-20_000+1:end]

# Convert to MCMCChains format
chain = Chains(ch')

# ─────────────────────── 7. Naive prediction for RNN ─────────────────────────
function naive_prediction_rnn(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    window = bnn.like.window
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    
    yhats = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        net = bnn.like.nc(draws[:, i])
        
        # Process each observation
        predictions = map(1:n_obs) do t
            # Reshape sequence and process through RNN
            seq_input = reshape(view(x, :, t), N, window)
            
            # Manual RNN forward pass
            rnn_layer = net[1]
            dense_layer = net[2]
            h = zeros(T, size(rnn_layer.cell.Wi, 1))
            
            for step in 1:window
                input_step = view(seq_input, :, step)
                h = tanh.(rnn_layer.cell.Wi * input_step + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
            end
            
            final_output = dense_layer(h)
            return final_output[1:N]  # Extract means only
        end
        
        yhats[:, i] = vcat(predictions...)
    end
    
    return yhats
end

# Generate naive predictions
yhats = naive_prediction_rnn(bnn, ch)
chain_yhat = Chains(yhats')

# Check convergence
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
println("Maximum R-hat: ", r_hat)

# ─────────────────────── 8. Posterior predictive checks ─────────────────────
function posterior_predict_rnn(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    window = bnn.like.window
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    
    pred_samples = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        θnet = draws[1:end-2, i]
        θlike = draws[end-1:end, i]
        a, b = transform_ab(θlike...)
        
        net = bnn.like.nc(θnet)
        
        # Get predictions for all observations
        results = map(1:n_obs) do t
            seq_input = reshape(view(x, :, t), N, window)
            
            # Manual RNN forward pass
            rnn_layer = net[1]
            dense_layer = net[2]
            h = zeros(T, size(rnn_layer.cell.Wi, 1))
            
            for step in 1:window
                input_step = view(seq_input, :, step)
                h = tanh.(rnn_layer.cell.Wi * input_step + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
            end
            
            final_output = dense_layer(h)
            μ_t = final_output[1:N]
            σ_t = exp.(final_output[N+1:2N] ./ 2)
            return (μ_t, σ_t)
        end
        
        μ_all = hcat([r[1] for r in results]...)
        σ_all = hcat([r[2] for r in results]...)
        
        # DCC dynamics and sampling
        z = (y .- μ_all) ./ σ_all
        Q̄ = (z * z') / n_obs
        d = 1 ./ sqrt.(diag(Q̄))
        Q̄ = Symmetric(diagm(d) * Q̄ * diagm(d))
        Q = copy(Q̄)
        
        samples = similar(μ_all)
        for t = 1:n_obs
            if t > 1
                zt = view(z, :, t-1)
                Q = (1-a-b) .* Q̄ .+ a .* (zt*zt') .+ b .* Q
            end
            
            d = 1 ./ sqrt.(max.(diag(Q), 1e-8))
            R = Symmetric(diagm(d) * Q * diagm(d))
            D = Diagonal(view(σ_all, :, t))
            H = D * R * D
            
            samples[:, t] = μ_all[:, t] + cholesky(Symmetric(H + 1e-6*I)).L * randn(N)
        end
        
        pred_samples[:, i] = vec(samples)
    end
    
    return pred_samples
end

posterior_yhat = posterior_predict_rnn(bnn, ch)

# Define quantiles and evaluation
t_q = 0.05:0.05:0.95

function get_observed_quantiles(y_true, y_samples, quantiles)
    N = 2
    n_quantiles = length(quantiles)
    observed_quantiles = zeros(n_quantiles)
    
    if ndims(y_true) == 2 && size(y_true, 2) == N
        y_true_vector = vec(y_true)
    else
        y_true_vector = y_true
    end
    
    n_time_points = div(size(y_samples, 1), N)
    total_points = N * n_time_points
    
    y_samples_reshaped = reshape(y_samples[1:total_points, :], total_points, :)
    
    if length(y_true_vector) != total_points
        y_true_vector = y_true_vector[1:total_points]
    end
    
    for i = 1:n_quantiles
        q = quantiles[i]
        threshold = [quantile(y_samples_reshaped[j, :], q) for j = 1:total_points]
        observed_quantiles[i] = sum(y_true_vector .< threshold) / total_points
    end
    
    return observed_quantiles
end

o_q = get_observed_quantiles(y, posterior_yhat, t_q)

# Create quantile-quantile plot
p1 = plot(t_q, o_q, label = "RNN Posterior Predictive", legend=:topleft,
     xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(p1, x->x, t_q, label = "Target")
plot!(p1, x->x, t_q, label = "45-degree line", linestyle=:dash) 
plot!(p1, title = "RNN DCC-GARCH Quantile-Quantile Plot", size = (800, 600))

# Display the plot
display(p1)



