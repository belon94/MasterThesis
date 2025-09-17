include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/BayesFlux.jl")
using Flux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors
using BayesFlux
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

# Implementation of the likelihood function - Zygote-compatible
function (l::DCCGarchNormal{T,F,D})(x::Array{T,3}, y::Matrix{T}, 
                                   θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)
    
    # Transform DCC parameters to valid range
    a, b = transform_ab(θlike[1], θlike[2])
    
    # Initialize network
    net = l.nc(θnet)
    
    # Dimensions
    N, Tsteps = l.N, size(x, 1)
    
    # Initialize arrays for means, standard deviations and standardized residuals
    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)  # standard deviation instead of variance
    z = similar(μ)
    
    # Extract parameters and compute standardized residuals for each time step
    @inbounds for t in 1:Tsteps
        # Use view for efficient slicing of x
        xt_view = @view x[t, :, :]
        # Get network output
        out = net(xt_view)     
        length(out) == 2N || error("Network output must be 2N scalars.")
        
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
    
    # Always ensure H_t is positive definite - no try/catch for Zygote compatibility
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

# Implementation of posterior prediction - Zygote-compatible
function posterior_predict(l::DCCGarchNormal{T,F,D}, 
                          x::Array{T,3}, 
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
    N, Tsteps = l.N, size(x, 1)
    
    # Process historical data for DCC recursion
    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)  # standard deviation instead of variance
    z = similar(μ)
    
    @inbounds for t in 1:Tsteps
        # Use view for efficient slicing of x
        xt_view = @view x[t, :, :]
        # Get network output
        out = net(xt_view)
        length(out) == 2N || error("Network output must be 2N scalars.")
        
        # Fixed: don't use @view on output vector, use indexing instead
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

# Prepare data for RNN - make sequences
function make_sequences(data, window_size)
    n_samples = size(data, 1) - window_size
    n_features = size(data, 2)
    
    x = Array{Float32, 3}(undef, n_samples, window_size, n_features)
    y = Array{Float32, 2}(undef, n_samples, n_features)
    
    for i in 1:n_samples
        x[i, :, :] = data[i:(i+window_size-1), :]
        y[i, :] = data[i+window_size, :]
    end
    
    return x, y
end

# Create sequences with a window size of 5
window_size = 5
x_train, y_train_target = make_sequences(y_train, window_size)

# Setup neural network for DCC-GARCH
# We need to output 2*N parameters (mean and log-variance for each series)
net = Chain(
    LSTM(N, 10),  # LSTM layer: N inputs, 10 hidden units
    Dense(10, 20, relu),  # Hidden layer
    Dense(20, 2*N)  # Output layer: mean and log-variance for each series
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
θmap = find_mode(bnn, 2, 5000, opt)

# Get predictions from MAP estimate
nethat = nc(θmap)

# Function to extract predictions
function extract_predictions(net, x)
    n_samples = size(x, 1)
    n_features = size(x, 3)
    
    means = zeros(Float32, n_samples, n_features)
    sigmas = zeros(Float32, n_samples, n_features)
    
    for i in 1:n_samples
        out = net(x[i, :, :])
        means[i, :] = out[1:n_features]
        sigmas[i, :] = exp.(out[n_features+1:2*n_features] ./ 2)
    end
    
    return means, sigmas
end

# Get predictions
means_train, sigmas_train = extract_predictions(nethat, x_train)

##----------------------------------------------------------------

using Flux
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
    D = Diagonal(max.(F.values, 1e-10))
    
    # Reconstruct matrix
    return F.vectors * D * F.vectors'
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

# Modified data preparation function for sequences
function make_sequences(data, window_size)
    n_samples = size(data, 1) - window_size
    n_features = size(data, 2)
    
    # Create arrays with correct dimensions for features-first format
    x = Array{Float32, 3}(undef, n_samples, n_features, window_size)
    y = Array{Float32, 2}(undef, n_samples, n_features)
    
    for i in 1:n_samples
        # Transpose the window so features come first (n_features × window_size)
        x[i, :, :] = permutedims(data[i:(i+window_size-1), :], (2, 1))
        y[i, :] = data[i+window_size, :]
    end
    
    return x, y
end

# Implementation of the likelihood function - Zygote-compatible - with corrected input handling
function (l::DCCGarchNormal{T,F,D})(x::Array{T,3}, y::Matrix{T}, 
                                   θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)
    
    # Transform DCC parameters to valid range
    a, b = transform_ab(θlike[1], θlike[2])
    
    # Initialize network
    net = l.nc(θnet)
    
    # Dimensions
    N, Tsteps = l.N, size(x, 1)
    
    # Initialize arrays for means, standard deviations and standardized residuals
    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)  # standard deviation instead of variance
    z = similar(μ)
    
    # Extract parameters and compute standardized residuals for each time step
    @inbounds for t in 1:Tsteps
        # CORRECTED: Properly reshape x for the LSTM - no need for view
        xt_reshaped = x[t, :, :]  # Already in format [n_features, window_size]
        
        # Get network output
        out = net(xt_reshaped)     
        length(out) == 2N || error("Network output must be 2N scalars.")
        
        # Fixed: don't use @view on output vector, use indexing instead
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
    
    # Always ensure H_t is positive definite - no try/catch for Zygote compatibility
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

# Implementation of posterior prediction - Zygote-compatible - with corrected input handling
function posterior_predict(l::DCCGarchNormal{T,F,D}, 
                          x::Array{T,3}, 
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
    N, Tsteps = l.N, size(x, 1)
    
    # Process historical data for DCC recursion
    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)  # standard deviation instead of variance
    z = similar(μ)
    
    @inbounds for t in 1:Tsteps
        # CORRECTED: Properly reshape x for the LSTM
        xt_reshaped = x[t, :, :]  # Already in format [n_features, window_size]
        
        # Get network output
        out = net(xt_reshaped)
        length(out) == 2N || error("Network output must be 2N scalars.")
        
        # Fixed: don't use @view on output vector, use indexing instead
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

# Create sequences with a window size of 5
window_size = 5
x_train, y_train_target = make_sequences(y_train, window_size)

# Setup neural network for DCC-GARCH
# We need to output 2*N parameters (mean and log-variance for each series)
net = Chain(
    LSTM(N, 10),  # LSTM layer: N inputs, 10 hidden units
    Dense(10, 20, relu),  # Hidden layer
    Dense(20, 2*N)  # Output layer: mean and log-variance for each series
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
θmap = find_mode(bnn, 2, 5000, opt)

# Get predictions from MAP estimate
nethat = nc(θmap)

# Function to extract predictions
function extract_predictions(net, x)
    n_samples = size(x, 1)
    n_features = size(x, 3)
    
    means = zeros(Float32, n_samples, n_features)
    sigmas = zeros(Float32, n_samples, n_features)
    
    for i in 1:n_samples
        out = net(x[i, :, :])
        means[i, :] = out[1:n_features]
        sigmas[i, :] = exp.(out[n_features+1:2*n_features] ./ 2)
    end
    
    return means, sigmas
end

# Get predictions
means_train, sigmas_train = extract_predictions(nethat, x_train)

# Create test sequences
x_test, y_test_target = make_sequences(y_test, window_size)

# Get predictions for test set
means_test, sigmas_test = extract_predictions(nethat, x_test)

# Plot results
p1 = plot(y_train[window_size+1:end, 1], label="Actual Returns (Train)", color=:blue, 
          title="Series 1", xlabel="Time", ylabel="Returns")
plot!(p1, means_train[:, 1], label="Predicted Mean (Train)", color=:red)
plot!(p1, [means_train[:, 1] + 2*sigmas_train[:, 1], means_train[:, 1] - 2*sigmas_train[:, 1]], 
      label=["95% CI" nothing], color=:red, alpha=0.3, linestyle=:dash)

p2 = plot(y_train[window_size+1:end, 2], label="Actual Returns (Train)", color=:blue, 
          title="Series 2", xlabel="Time", ylabel="Returns")
plot!(p2, means_train[:, 2], label="Predicted Mean (Train)", color=:red)
plot!(p2, [means_train[:, 2] + 2*sigmas_train[:, 2], means_train[:, 2] - 2*sigmas_train[:, 2]], 
      label=["95% CI" nothing], color=:red, alpha=0.3, linestyle=:dash)

plot(p1, p2, layout=(2,1), size=(800, 600))


###-----------------------------------------------------------------
###############################################################################
#  DCC–GARCH Bayesian neural-network example (BayesFlux.jl)                   #
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

function prepare_time_series_data(data, w)
    n = size(data,1) - w
    p = size(data,2)
    X = Array{Float32}(undef, p*w, n)
    Y = Array{Float32}(undef, p,   n)
    for i in 1:n
        X[:,i] = reshape(data[i:i+w-1,:]', :)
        Y[:,i] = data[i+w,:]
    end
    return X, Y
end

# ─────────────────── 3.  custom DCC-GARCH likelihood ────────────────────────
struct DCCGarchNormal{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
end
DCCGarchNormal(nc::NetConstructor{T,F}, prior::D, N::Int) where {T,F,D<:Distribution} =
    DCCGarchNormal{T,F,D}(2, nc, prior, N)

function (ℓ::DCCGarchNormal)(x::Matrix{T}, y::Matrix{T},
                             θnet::AbstractVector, θlike::AbstractVector) where {T}

    θnet, θlike = T.(θnet), T.(θlike)
    a,b         = transform_ab(θlike...)
    net         = ℓ.nc(θnet)
    N, Tsteps   = ℓ.N, size(x,2)

    outs  = map(t -> net(view(x,:,t)), 1:Tsteps)
    μ     = hcat(map(o -> o[1:N]    , outs)...)
    logσ2 = hcat(map(o -> o[N+1:2N] , outs)...)
    σ     = exp.(logσ2 ./ 2)
    z     = (y .- μ) ./ σ

    Q̄ = Tsteps>1 ? (z*z')/Tsteps : Matrix{T}(I,N,N)
    d  = 1 ./ sqrt.(diag(Q̄))
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
            Q  = (1-a-b).*Q̄ .+ a.*(zt*zt') .+ b.*Q
        end
    end
    logl += sum(logpdf.(ℓ.prior, vec(μ)))
    return logl
end

# ───────────────────────── 4.  simulate data ────────────────────────────────
n, N = 500, 2
α, β  = 0.05, 0.90
vol   = [(20^2)/252, (15^2)/252]
ρ     = 0.5

sim = dccGarchNDraws(n,N,α,β,vol,ρ)
y   = Float32.(sim["L"])

train_idx = 1:400
window    = 20
Xtr, Ytr  = prepare_time_series_data(y[train_idx,:], window)

# ─────────────────────── 5.  build & train BNN ──────────────────────────────
net = Chain(
    Dense(N*window,20,relu),
    Dense(20,20,relu),
    Dense(20,2N)
)
nc    = destruct(net)
like  = DCCGarchNormal(nc, Normal(0,0.5), N)
prior = GaussianPrior(nc, 0.5f0)
init  = InitialiseAllSame(Normal(0f0,0.5f0), like, prior)
bnn   = BNN(Xtr, Ytr, like, prior, init)

θmap = find_mode(bnn, 50, 500, FluxModeFinder(bnn, Flux.ADAM()))

# ─────────────────────── 6.  posterior sampling (SGLD) ──────────────────────
sgld = SGLD(Float32;               # \gamma<Tab> for the Unicode letter
            stepsize_a = 1f-3,     # a bit smaller → fewer NaNs
            stepsize_b = 0f0,
            stepsize_γ = 0.55f0)

draws = mcmc(bnn, 100, 5_000, sgld)          # (#par × #draws)
draws = draws[:, 2001:end]                  # discard first 2 000 burn-in

# ── NEW: drop any column that contains NaN or Inf ───────────────────────────
good   = map(i -> all(isfinite, view(draws,:,i)), axes(draws,2))
draws  = draws[:, good]                      # keep only “good” draws

samples = [draws[:,i] for i in axes(draws,2)]    # vector-of-vectors

# ─────────────── 7.  diagnostics (MCMCChains) ───────────────────────────────
param_names = ["θ_$i" for i in 1:size(draws,1)]

########There is an error in the code below, it is not working. I will fix it later########
chn = Chains(permutedims(draws,(2,1)), param_names)



###############

###############################################################################
#  DCC–GARCH Bayesian neural-network example (BayesFlux.jl)                   #
###############################################################################
using Flux,BayesFlux
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

function prepare_time_series_data(data, w)
    n = size(data,1) - w
    p = size(data,2)
    X = Array{Float32}(undef, p*w, n)
    Y = Array{Float32}(undef, p,   n)
    for i in 1:n
        X[:,i] = reshape(data[i:i+w-1,:]', :)
        Y[:,i] = data[i+w,:]
    end
    return X, Y
end

# ─────────────────── 3.  custom DCC-GARCH likelihood ────────────────────────
struct DCCGarchNormal{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior::D
    N::Int
end
DCCGarchNormal(nc::NetConstructor{T,F}, prior::D, N::Int) where {T,F,D<:Distribution} =
    DCCGarchNormal{T,F,D}(2, nc, prior, N)

function (ℓ::DCCGarchNormal)(x::Matrix{T}, y::Matrix{T},
                             θnet::AbstractVector, θlike::AbstractVector) where {T}

    θnet, θlike = T.(θnet), T.(θlike)
    a,b         = transform_ab(θlike...)
    net         = ℓ.nc(θnet)
    N, Tsteps   = ℓ.N, size(x,2)

    outs  = map(t -> net(view(x,:,t)), 1:Tsteps)
    μ     = hcat(map(o -> o[1:N]    , outs)...)
    logσ2 = hcat(map(o -> o[N+1:2N] , outs)...)
    σ     = exp.(logσ2 ./ 2)
    z     = (y .- μ) ./ σ

    Q̄ = Tsteps>1 ? (z*z')/Tsteps : Matrix{T}(I,N,N)
    d  = 1 ./ sqrt.(diag(Q̄))
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
            Q  = (1-a-b).*Q̄ .+ a.*(zt*zt') .+ b.*Q
        end
    end
    logl += sum(logpdf.(ℓ.prior, vec(μ)))
    return logl
end

# ───────────────────────── 4.  simulate data ────────────────────────────────
n, N = 500, 2  # Number of observations and variables
α, β  = 0.05, 0.90  # DCC-GARCH parameters
vol   = [(20^2)/252, (15^2)/252]  # Initial volatilities (annualized)
ρ     = 0.5  # Correlation coefficient

# Generate simulated data
sim = dccGarchNDraws(n, N, α, β, vol, ρ)
y   = Float32.(sim["L"])  # Returns

# Define training set
train_idx = 1:400  # Use first 400 observations for training
window    = 22     # Use 22 lags as requested
Xtr, Ytr  = prepare_time_series_data(y[train_idx,:], window)

# ─────────────────────── 5.  build & train BNN ──────────────────────────────
# Define neural network architecture
net = Chain(
    Dense(N*window, 20, relu),  # Input layer (2*22=44 inputs)
    Dense(20, 20, relu),        # Hidden layer
    Dense(20, 2N)               # Output layer (4 outputs: mean and log variance for each series)
)

# Setup BNN components
nc    = destruct(net)  # Network constructor
like  = DCCGarchNormal(nc, Normal(0, 0.5), N)  # Custom likelihood
prior = GaussianPrior(nc, 0.5f0)  # Prior distribution
init  = InitialiseAllSame(Normal(0f0, 0.5f0), like, prior)  # Parameter initialization
bnn   = BNN(Xtr, Ytr, like, prior, init)  # BNN model

# Find mode of posterior for initialization
θmap = find_mode(bnn, 50, 500, FluxModeFinder(bnn, Flux.ADAM()))

# ─────────────────────── 6.  posterior sampling (SGNHTS) ─────────────────────
 # Try a much smaller step size (10x smaller)
sampler = SGNHTS(1f-4, 1f0; xi = 1f0^2, μ = 10f0)  # Changed from 1f-2 to 1f-4
ch = mcmc(bnn, 10, 50_000, sampler) # 10 parameters, 50,000 samples
# Keep only the last 20,000 samples (discard 30,000 as burn-in)
ch = ch[:, end-20_000+1:end]

# Convert to MCMCChains format
chain = Chains(ch')

# ─────────────────────── 7. Naive prediction ─────────────────────────────────
# Define the naive_prediction function for your DCC-GARCH model
function naive_prediction(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N  # Number of time series (2 in your case)
    n_samples = size(draws, 2)  # Number of posterior samples
    n_obs = size(x, 2)  # Number of observations
    
    # Initialize array to store predictions (means only, not variances)
    yhats = Array{T, 2}(undef, N * n_obs, n_samples)
    
    # Loop through each posterior sample
    Threads.@threads for i = 1:n_samples
        # Reconstruct network from parameters
        net = bnn.like.nc(draws[:, i])
        
        # Get predictions for all observations
        outs = map(t -> net(view(x, :, t)), 1:n_obs)
        
        # Extract means (first N elements of each output)
        μ = hcat(map(o -> o[1:N], outs)...)
        
        # Store means in the output array
        yhats[:, i] = vec(μ)
    end
    
    return yhats
end

# Generate naive predictions from the BNN using the MCMC samples
yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')

# Check convergence using R-hat statistic
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
println("Maximum R-hat: ", r_hat)

# ─────────────────────── 8. Posterior predictive checks ─────────────────────


# Sample from the posterior predictive distribution
function my_posterior_predict(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N  # Number of time series
    n_samples = size(draws, 2)  # Number of posterior samples
    n_obs = size(x, 2)  # Number of observations
    
    # Initialize array to store sampled predictions
    pred_samples = Array{T, 2}(undef, N * n_obs, n_samples)
    
    # Loop through each posterior sample
    Threads.@threads for i = 1:n_samples
        # Get likelihood parameters (a,b for DCC)
        θnet = draws[1:end-2, i]  # Network parameters
        θlike = draws[end-1:end, i]  # Likelihood parameters (a,b)
        a, b = transform_ab(θlike...)
        
        # Reconstruct network from parameters
        net = bnn.like.nc(θnet)
        
        # Get means and log variances for all observations
        outs = map(t -> net(view(x, :, t)), 1:n_obs)
        μ = hcat(map(o -> o[1:N], outs)...)
        logσ2 = hcat(map(o -> o[N+1:2N], outs)...)
        σ = exp.(logσ2 ./ 2)
        
        # Initialize Q matrix (unconditional correlation)
        z = (y .- μ) ./ σ  # Standardized residuals
        Q̄ = (z * z') / n_obs  # Unconditional correlation
        d = 1 ./ sqrt.(diag(Q̄))
        Q̄ = Symmetric(diagm(d) * Q̄ * diagm(d))
        Q = copy(Q̄)
        
        # Sample from predictive distribution with correlation
        samples = similar(μ)
        for t = 1:n_obs
            # Update Q based on DCC recursion
            if t > 1
                zt = view(z, :, t-1)
                Q = (1-a-b) .* Q̄ .+ a .* (zt*zt') .+ b .* Q
            end
            
            # Compute correlation matrix R
            d = 1 ./ sqrt.(max.(diag(Q), 1e-8))
            R = Symmetric(diagm(d) * Q * diagm(d))
            
            # Compute covariance matrix H = DRD
            D = Diagonal(view(σ, :, t))
            H = D * R * D
            
            # Generate correlated sample
            samples[:, t] = μ[:, t] + cholesky(Symmetric(H + 1e-6*I)).L * randn(N)
        end
        
        # Store samples in the output array
        pred_samples[:, i] = vec(samples)
    end
    
    return pred_samples
end

posterior_yhat = my_posterior_predict(bnn, ch)

# Define quantiles for evaluation
t_q = 0.05:0.05:0.95

# Get observed quantiles from posterior predictions
function get_observed_quantiles(y_true, y_samples, quantiles)
    N = 2  # Number of time series
    n_quantiles = length(quantiles)
    observed_quantiles = zeros(n_quantiles)
    
    # Reshape true values if needed
    if ndims(y_true) == 2 && size(y_true, 2) == N
        y_true_vector = vec(y_true)  # Flatten the matrix
    else
        y_true_vector = y_true  # Already in the right shape
    end
    
    # Get number of time points from the posterior samples
    n_time_points = div(length(y_samples), N * size(y_samples, 2))
    total_points = N * n_time_points
    
    # Reshape samples for quantile calculation
    # This creates a matrix with rows = data points, cols = posterior samples
    y_samples_reshaped = reshape(y_samples, total_points, :)
    
    # Ensure y_true matches the dimensions of reshaped samples
    if length(y_true_vector) != total_points
        # Use only the relevant portion of y_true that matches the predictions
        y_true_vector = y_true_vector[1:total_points]
    end
    
    # Calculate observed quantiles
    for i = 1:n_quantiles
        q = quantiles[i]
        # For each data point, calculate the predicted quantile
        threshold = [quantile(y_samples_reshaped[j, :], q) for j = 1:total_points]
        observed_quantiles[i] = sum(y_true_vector .< threshold) / total_points
    end
    
    return observed_quantiles
end

o_q = get_observed_quantiles(y, posterior_yhat, t_q)

# Create quantile-quantile plot to assess model calibration
p1 = plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
     xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(p1, x->x, t_q, label = "Target")
plot!(p1, x->x, t_q, label = "45-degree line", linestyle=:dash) 
plot!(p1, title = "Quantile-Quantile Plot", size = (800, 600))

#### This is the LSTM model for DCC-GARCH

###############################################################################
#  DCC–GARCH Bayesian Sequential LSTM  (BayesFlux.jl) - COMPLETE      #
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

# Sequential data preparation (no windowing)
function prepare_sequential_data(data)
    # Each column is a time step, each row is a feature
    X = data[1:end-1, :]'  # Input: all but last observation, transposed
    Y = data[2:end, :]'    # Target: all but first observation, transposed
    return Float32.(X), Float32.(Y)
end

# ─────────────────── 3.  Sequential LSTM DCC-GARCH likelihood ─────────────────
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
    
    # Process entire sequence through LSTM
    lstm_layer = net[1]
    dense_layer = net[2]
    
    # Initialize LSTM states
    hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
    h = zeros(T, hidden_size)
    c = zeros(T, hidden_size)
    
    # Process each time step sequentially and collect outputs
    outs = map(1:Tsteps) do t
        input_t = view(x, :, t)
        
        # LSTM forward pass with state persistence
        gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
        
        # Split into 4 gates: input, forget, cell, output
        i_gate = sigmoid.(gates[1:hidden_size])
        f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
        g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
        o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
        
        # Update cell state and hidden state (state carries forward!)
        c = f_gate .* c .+ i_gate .* g_gate
        h = o_gate .* tanh.(c)
        
        # Generate prediction for this time step
        output = dense_layer(h)
        return output
    end
    
    # Extract means and log variances
    μ = hcat(map(o -> o[1:N], outs)...)
    logσ2 = hcat(map(o -> o[N+1:2N], outs)...)
    σ = exp.(logσ2 ./ 2)
    z = (y .- μ) ./ σ

    # DCC-GARCH likelihood computation
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

# Define training set (no windowing needed!)
train_idx = 1:400  # Use first 400 observations for training
Xtr, Ytr = prepare_sequential_data(y[train_idx,:])

# ─────────────────────── 5.  build & train Sequential LSTM BNN ────────────────
# Define Sequential LSTM architecture
hidden_size = 20
net = Chain(
    LSTM(N => hidden_size),      # LSTM layer: N inputs to hidden_size outputs
    Dense(hidden_size => 2N)     # Output layer: mean and log variance for each series
)

# Setup BNN components
nc = destruct(net)  # Network constructor
like = DCCGarchLSTMSequential(nc, Normal(0, 0.5), N)  # Sequential LSTM likelihood
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

# ─────────────────────── 7. Sequential LSTM Prediction Functions ─────────────
function naive_prediction_sequential_lstm(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    
    yhats = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        net = bnn.like.nc(draws[:, i])
        lstm_layer = net[1]
        dense_layer = net[2]
        
        # Initialize LSTM states
        hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
        h = zeros(T, hidden_size)
        c = zeros(T, hidden_size)
        
        # Process sequence and collect predictions
        predictions = map(1:n_obs) do t
            input_t = view(x, :, t)
            
            # LSTM forward pass
            gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
            
            i_gate = sigmoid.(gates[1:hidden_size])
            f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
            g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
            o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
            
            # Update states
            c = f_gate .* c .+ i_gate .* g_gate
            h = o_gate .* tanh.(c)
            
            final_output = dense_layer(h)
            return final_output[1:N]  # Extract means only
        end
        
        yhats[:, i] = vcat(predictions...)
    end
    
    return yhats
end

# Generate naive predictions
yhats = naive_prediction_sequential_lstm(bnn, ch)
chain_yhat = Chains(yhats')

# Check convergence
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
println("Maximum R-hat: ", r_hat)

# ─────────────────────── 8. Sequential LSTM Posterior Predictive Checks ─────────────
function posterior_predict_sequential_lstm(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    
    pred_samples = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        θnet = draws[1:end-2, i]
        θlike = draws[end-1:end, i]
        a, b = transform_ab(θlike...)
        
        net = bnn.like.nc(θnet)
        lstm_layer = net[1]
        dense_layer = net[2]
        
        # Initialize LSTM states
        hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
        h = zeros(T, hidden_size)
        c = zeros(T, hidden_size)
        
        # Get predictions for all observations
        results = map(1:n_obs) do t
            input_t = view(x, :, t)
            
            # LSTM forward pass
            gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
            
            i_gate = sigmoid.(gates[1:hidden_size])
            f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
            g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
            o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
            
            # Update states
            c = f_gate .* c .+ i_gate .* g_gate
            h = o_gate .* tanh.(c)
            
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

posterior_yhat = posterior_predict_sequential_lstm(bnn, ch)

# ─────────────────────── 9. Model Evaluation ─────────────────────────────────
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
p1 = plot(t_q, o_q, label = "Sequential LSTM Posterior Predictive", legend=:topleft,
     xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(p1, x->x, t_q, label = "45-degree line", linestyle=:dash) 
plot!(p1, title = "Sequential LSTM DCC-GARCH Quantile-Quantile Plot", size = (800, 600))

# Display the plot
display(p1)

# ─────────────────────── 10. Additional Diagnostics ─────────────────────────
println("=== Model Summary ===")
println("Data points (training): ", size(Xtr, 2))
println("Features: ", N)
println("LSTM hidden size: ", hidden_size)
println("Total parameters: ", length(θmap))
println("DCC parameters (a, b): ", transform_ab(mean(ch[end-1:end, :], dims=2)...))
println("Maximum R-hat: ", r_hat)

# Plot some sample paths
if r_hat < 1.1  # Only if converged
    n_plot_samples = min(100, size(posterior_yhat, 2))
    sample_indices = rand(1:size(posterior_yhat, 2), n_plot_samples)
    
    p2 = plot(title="Posterior Predictive Samples vs Actual", size=(800, 400))
    
    # Plot actual data
    plot!(p2, vec(y[train_idx[2:end], 1]), label="Actual Series 1", color=:black, linewidth=2)
    
    # Plot sample predictions
    for i in 1:min(20, n_plot_samples)  # Show max 20 samples
        y_sample = reshape(posterior_yhat[:, sample_indices[i]], N, :)
        plot!(p2, vec(y_sample[1, :]), alpha=0.1, color=:blue, label=i==1 ? "Predicted Samples" : "")
    end
    
    display(p2)
end

println("=== Sequential LSTM DCC-GARCH Model Complete ===")


###############################################################################
#  DCC–GARCH Bayesian Sequential RNN  (BayesFlux.jl) - COMPLETE       #
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
nearest_pd(A) = (A + A')/2 + 1e-4I

# Sequential data preparation (no windowing)
function prepare_sequential_data(data)
    X = data[1:end-1, :]'  # Input: all but last observation, transposed
    Y = data[2:end, :]'    # Target: all but first observation, transposed
    return Float32.(X), Float32.(Y)
end

# ─────────────────── 3.  Sequential RNN DCC-GARCH likelihood ─────────────────
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
    
    # Process entire sequence through RNN
    rnn_layer = net[1]
    dense_layer = net[2]
    
    # Initialize RNN hidden state (much simpler than LSTM!)
    hidden_size = size(rnn_layer.cell.Wi, 1)
    h = zeros(T, hidden_size)
    
    # Process each time step sequentially and collect outputs
    outs = map(1:Tsteps) do t
        input_t = view(x, :, t)
        
        # Simple RNN forward pass: h_t = tanh(W_i * x_t + W_h * h_{t-1} + b)
        h = tanh.(rnn_layer.cell.Wi * input_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
        
        # Generate prediction for this time step
        output = dense_layer(h)
        return output
    end
    
    # Extract means and log variances
    μ = hcat(map(o -> o[1:N], outs)...)
    logσ2 = hcat(map(o -> o[N+1:2N], outs)...)
    σ = exp.(logσ2 ./ 2)
    z = (y .- μ) ./ σ

    # DCC-GARCH likelihood computation
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
n, N = 500, 2
α, β = 0.05, 0.90
vol = [(20^2)/252, (15^2)/252]
ρ = 0.5

sim = dccGarchNDraws(n, N, α, β, vol, ρ)
y = Float32.(sim["L"])

train_idx = 1:400
Xtr, Ytr = prepare_sequential_data(y[train_idx,:])

# ─────────────────────── 5.  build & train Sequential RNN BNN ────────────────
# Define Sequential RNN architecture (simpler than LSTM!)
hidden_size = 20
net = Chain(
    RNN(N => hidden_size),       # RNN layer: N inputs to hidden_size outputs
    Dense(hidden_size => 2N)     # Output layer: mean and log variance for each series
)

# Setup BNN components
nc = destruct(net)
like = DCCGarchRNNSequential(nc, Normal(0, 0.5), N)  # Sequential RNN likelihood
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0f0, 0.5f0), like, prior)
bnn = BNN(Xtr, Ytr, like, prior, init)

# Find mode of posterior for initialization
θmap = find_mode(bnn, 50, 500, FluxModeFinder(bnn, Flux.ADAM()))

# ─────────────────────── 6.  posterior sampling (SGNHTS) ─────────────────────
sampler = SGNHTS(1f-4, 1f0; xi = 1f0^2, μ = 10f0)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]

chain = Chains(ch')

# ─────────────────────── 7. Sequential RNN Prediction Functions ─────────────
function naive_prediction_sequential_rnn(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    
    yhats = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        net = bnn.like.nc(draws[:, i])
        rnn_layer = net[1]
        dense_layer = net[2]
        
        # Initialize RNN hidden state (only one state, not two like LSTM!)
        hidden_size = size(rnn_layer.cell.Wi, 1)
        h = zeros(T, hidden_size)
        
        # Process sequence and collect predictions
        predictions = map(1:n_obs) do t
            input_t = view(x, :, t)
            
            # Simple RNN forward pass
            h = tanh.(rnn_layer.cell.Wi * input_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
            
            final_output = dense_layer(h)
            return final_output[1:N]  # Extract means only
        end
        
        yhats[:, i] = vcat(predictions...)
    end
    
    return yhats
end

# Generate naive predictions
yhats = naive_prediction_sequential_rnn(bnn, ch)
chain_yhat = Chains(yhats')

# Check convergence
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
println("Maximum R-hat: ", r_hat)

# ─────────────────────── 8. Sequential RNN Posterior Predictive Checks ─────────────
function posterior_predict_sequential_rnn(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    
    pred_samples = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        θnet = draws[1:end-2, i]
        θlike = draws[end-1:end, i]
        a, b = transform_ab(θlike...)
        
        net = bnn.like.nc(θnet)
        rnn_layer = net[1]
        dense_layer = net[2]
        
        # Initialize RNN hidden state
        hidden_size = size(rnn_layer.cell.Wi, 1)
        h = zeros(T, hidden_size)
        
        # Get predictions for all observations
        results = map(1:n_obs) do t
            input_t = view(x, :, t)
            
            # RNN forward pass
            h = tanh.(rnn_layer.cell.Wi * input_t + rnn_layer.cell.Wh * h .+ rnn_layer.cell.b)
            
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

posterior_yhat = posterior_predict_sequential_rnn(bnn, ch)

# ─────────────────────── 9. Model Evaluation ─────────────────────────────────
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
p1 = plot(t_q, o_q, label = "Sequential RNN Posterior Predictive", legend=:topleft,
     xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(p1, x->x, t_q, label = "45-degree line", linestyle=:dash) 
plot!(p1, title = "Sequential RNN DCC-GARCH Quantile-Quantile Plot", size = (800, 600))

display(p1)

# ─────────────────────── 10. Additional Diagnostics ─────────────────────────
println("=== Sequential RNN Model Summary ===")
println("Data points (training): ", size(Xtr, 2))
println("Features: ", N)
println("RNN hidden size: ", hidden_size)
println("Total parameters: ", length(θmap))
println("DCC parameters (a, b): ", transform_ab(mean(ch[end-1:end, :], dims=2)...))
println("Maximum R-hat: ", r_hat)

# Plot some sample paths
if r_hat < 1.1
    n_plot_samples = min(100, size(posterior_yhat, 2))
    sample_indices = rand(1:size(posterior_yhat, 2), n_plot_samples)
    
    p2 = plot(title="Sequential RNN: Posterior Predictive Samples vs Actual", size=(800, 400))
    
    plot!(p2, vec(y[train_idx[2:end], 1]), label="Actual Series 1", color=:black, linewidth=2)
    
    for i in 1:min(20, n_plot_samples)
        y_sample = reshape(posterior_yhat[:, sample_indices[i]], N, :)
        plot!(p2, vec(y_sample[1, :]), alpha=0.1, color=:red, label=i==1 ? "RNN Predicted Samples" : "")
    end
    
    display(p2)
end

println("=== Sequential RNN DCC-GARCH Model Complete ===")