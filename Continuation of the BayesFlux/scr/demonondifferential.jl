using Flux
using BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors
using Zygote
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

# Helper functions
function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

function transform_ab(raw_a, raw_b)
    a = sigmoid(raw_a)
    b = sigmoid(raw_b) * (1 - a)
    return a, b
end

# Mark data preparation as completely non-differentiable
Zygote.@nograd function prepare_time_series_data(data, window_size)
    n_samples = size(data, 1) - window_size
    n_features = size(data, 2)
    
    x = Array{Float32}(undef, n_features * window_size, n_samples)
    y = Array{Float32}(undef, n_features, n_samples)
    
    for i in 1:n_samples
        window = data[i:(i+window_size-1), :]
        x_flattened = reshape(transpose(window), :)
        x[:, i] = x_flattened
        y[:, i] = data[i+window_size, :]
    end
    
    return x, y
end

# Complete rewrite of likelihood function with Zygote.ignore for non-differentiable parts
function (l::DCCGarchNormal{T,F,D})(x::Matrix{T}, y::Matrix{T}, 
                                   θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)
    
    # Transform parameters
    a, b = transform_ab(θlike[1], θlike[2])
    
    # Initialize network
    net = l.nc(θnet)
    
    # Dimensions
    N, Tsteps = l.N, size(x, 2)
    
    # Compute all network outputs and predictions first - these need gradients
    network_outputs = [net(x[:, t]) for t in 1:Tsteps]
    means = [network_outputs[t][1:N] for t in 1:Tsteps]
    stds = [exp.(network_outputs[t][N+1:2N] ./ 2) for t in 1:Tsteps]
    
    # Compute standardized residuals - these need gradients too
    zs = [(y[:, t] .- means[t]) ./ stds[t] for t in 1:Tsteps]
    
    # Wrap the DCC calculation in a Zygote.ignore block
    # This part doesn't need gradients and can use mutations
    logl = Zygote.ignore() do
        # Compute Qbar (unconditional correlation matrix)
        z_matrix = reduce(hcat, zs)
        Qbar = Tsteps > 1 ? (z_matrix * z_matrix') / Tsteps : Matrix{T}(I, N, N)
        
        # Ensure Qbar is correlation matrix
        if Tsteps > 1
            d_inv = 1 ./ sqrt.(diag(Qbar))
            Qbar = Symmetric(Diagonal(d_inv) * Qbar * Diagonal(d_inv))
        end
        
        # Initialize log-likelihood and Q
        logl_value = T(0)
        Q = copy(Qbar)
        
        # Process all time steps
        for t in 1:Tsteps
            # Compute correlation matrix R_t from Q
            d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
            R = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
            
            # Construct covariance matrix H_t
            Dt = Diagonal(stds[t])
            H_t = Dt * R * Dt
            
            # Ensure H_t is positive definite
            λ = max(1e-4, abs(minimum(eigvals(Symmetric(H_t)))))
            H_t = Symmetric(H_t) + λ * I
            
            # Compute log-likelihood contribution using Cholesky
            H_chol = cholesky(H_t)
            diff = y[:, t] .- means[t]
            quad = sum(abs2, H_chol.L \ diff)
            logl_value -= T(0.5) * (N * log(T(2π)) + 2*sum(log, diag(H_chol.L)) + quad)
            
            # Update Q for next time step if not the last step
            if t < Tsteps
                Q = (1-a-b) .* Qbar .+ a .* (zs[t] * zs[t]') .+ b .* Q
            end
        end
        
        return logl_value
    end
    
    # Add prior contribution - needs gradients
    prior_logl = sum([sum(logpdf.(l.prior_μ, means[t])) for t in 1:Tsteps])
    
    return logl + prior_logl
end

# Mark posterior prediction as completely non-differentiable
Zygote.@nograd function posterior_predict(l::DCCGarchNormal{T,F,D}, 
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
    σ = Matrix{T}(undef, N, Tsteps)
    z = Matrix{T}(undef, N, Tsteps)
    
    for t in 1:Tsteps
        # Get the flattened window for this sample
        xt_flat = x[:, t]
        
        # Get network output
        out = net(xt_flat)
        length(out) == 2N || error("Network output must be 2N scalars.")
        
        # Process output
        μ[:, t] = out[1:N]
        σ[:, t] = exp.(out[N+1:2N] ./ 2)  # Standard deviations
        z[:, t] = (y_hist[:, t] .- μ[:, t]) ./ σ[:, t]  # Standardized residuals
    end
    
    # Compute unconditional correlation matrix
    Qbar = Tsteps > 1 ? (z * z') / Tsteps : Matrix{T}(I, N, N)
    
    # Ensure Qbar is correlation matrix
    if Tsteps > 1
        d_inv = 1 ./ sqrt.(diag(Qbar))
        Qbar = Symmetric(Diagonal(d_inv) * Qbar * Diagonal(d_inv))
    end
    
    # Initialize Q as unconditional correlation
    Q = copy(Qbar)
    
    # Proper DCC recursion with temporal ordering
    if Tsteps > 1
        # Process observations sequentially
        for t in 1:(Tsteps-1)
            Q = (1-a-b) .* Qbar .+ a .* (z[:,t] * z[:,t]') .+ b .* Q
        end
        
        # Final update with last observation
        Q = (1-a-b) .* Qbar .+ a .* (z[:,Tsteps] * z[:,Tsteps]') .+ b .* Q
    end
    
    # Compute correlation matrix R_t
    d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
    R = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
    
    # Means and variance for prediction (from last time step)
    μp = μ[:, end]
    Hp = Diagonal(σ[:, end]) * R * Diagonal(σ[:, end])
    
    # Make sure Hp is positive definite
    λ = max(1e-4, abs(minimum(eigvals(Symmetric(Hp)))))
    Hp = Symmetric(Hp) + λ * I
    
    # Generate prediction
    return rand(MvNormal(μp, Hp))
end

# Mark prediction extraction as completely non-differentiable
Zygote.@nograd function extract_predictions(net, x)
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

# Create data for BNN in the format it expects
x_train, y_train_target = prepare_time_series_data(y_train, window_size)

# Network architecture
net = Chain(
    Dense(N * window_size, 20, relu),
    Dense(20, 20, relu),
    Dense(20, 2*N)
)

nc = destruct(net)

# Use DCCGarchNormal likelihood
like = DCCGarchNormal(nc, Normal(0, 0.5f0), N)
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)

# Create BNN
bnn = BNN(x_train, y_train_target, like, prior, init)

# Additional Zygote configuration to ignore BayesFlux internals
Zygote.@nograd BayesFlux.FluxModeFinder

# Find MAP estimate - use batch size from warnings
batch_size = 395  # Match the size mentioned in the warnings
opt = FluxModeFinder(bnn, Flux.ADAM())

# Wrap the find_mode call in Zygote.ignore to prevent any gradient tracking
θmap = Zygote.ignore() do
    find_mode(bnn, 1000, batch_size, opt)
end

# Get predictions from MAP estimate
nethat = nc(θmap)

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