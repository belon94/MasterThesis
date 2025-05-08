include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/BayesFlux.jl")
using Flux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors
using .BayesFlux
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
        
        try
            # Try to directly use Rt_raw for MvNormal
            std_noise = rand(MvNormal(zeros(N), Symmetric(Rt_raw)))
            Rt = Rt_raw  # Only update Rt if successful
        catch e
            if isa(e, PosDefException)
                # If Rt is not positive definite, fix it
                println("Fixing non-positive definite correlation matrix at time $t")
                
                # Method 1: Add small value to diagonal
                Rt_fix1 = Rt_raw + 1e-6 * I
                
                try
                    # Try with first fix
                    std_noise = rand(MvNormal(zeros(N), Symmetric(Rt_fix1)))
                    Rt = Rt_fix1
                catch e2
                    if isa(e2, PosDefException)
                        # Method 2: Use nearest correlation matrix via eigendecomposition
                        eigen_decomp = eigen(Symmetric(Rt_raw))
                        pos_values = max.(eigen_decomp.values, 1e-5)
                        Rt_fix2 = eigen_decomp.vectors * Diagonal(pos_values) * eigen_decomp.vectors'
                        
                        # Rescale to ensure diagonal is 1
                        d_diag = sqrt.(diag(Rt_fix2))
                        Rt_fix2 = Diagonal(1.0 ./ d_diag) * Rt_fix2 * Diagonal(1.0 ./ d_diag)
                        
                        # Final symmetry enforcement
                        Rt_fix2 = (Rt_fix2 + Rt_fix2')/2
                        
                        # Try with second fix
                        std_noise = rand(MvNormal(zeros(N), Symmetric(Rt_fix2)))
                        Rt = Rt_fix2
                    else
                        rethrow(e2)
                    end
                end
            else
                rethrow(e)
            end
        end
        
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

# Implementation of the likelihood function
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
    
    # Use Cholesky decomposition with checking enabled
    try
        Hchol = cholesky(H_t; check=true)
        
        # Compute log-likelihood contribution
        diff = y[:, t] .- μ[:, t]
        quad = sum(abs2, Hchol.L \ diff)
        
        logl -= 0.5 * (N * log(2π) + 2*sum(log, diag(Hchol.L)) + quad)
    catch e
        if isa(e, LinearAlgebra.PosDefException)
            # Fix the non-positive definite matrix
            H_t = nearest_pd(H_t)
            Hchol = cholesky(H_t)
            
            # Compute log-likelihood contribution
            diff = y[:, t] .- μ[:, t]
            quad = sum(abs2, Hchol.L \ diff)
            
            logl -= 0.5 * (N * log(2π) + 2*sum(log, diag(Hchol.L)) + quad)
        else
            rethrow(e)
        end
    end
    
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
        
        # Use Cholesky decomposition with checking enabled
        try
            Hchol = cholesky(H_t; check=true)
            
            # Compute log-likelihood contribution
            diff = y[:, t] .- μ[:, t]
            quad = sum(abs2, Hchol.L \ diff)
            
            logl -= 0.5 * (N * log(2π) + 2*sum(log, diag(Hchol.L)) + quad)
        catch e
            if isa(e, LinearAlgebra.PosDefException)
                # Fix the non-positive definite matrix
                H_t = nearest_pd(H_t)
                Hchol = cholesky(H_t)
                
                # Compute log-likelihood contribution
                diff = y[:, t] .- μ[:, t]
                quad = sum(abs2, Hchol.L \ diff)
                
                logl -= 0.5 * (N * log(2π) + 2*sum(log, diag(Hchol.L)) + quad)
            else
                rethrow(e)
            end
        end
        
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

# Implementation of posterior prediction
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
    
    # Make sure Hp is positive definite
    try
        cholesky(Hp; check=true)
    catch e
        if isa(e, LinearAlgebra.PosDefException)
            Hp = nearest_pd(Hp)
        else
            rethrow(e)
        end
    end
    
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
println("Finding MAP estimate...")
opt = FluxModeFinder(bnn, Flux.ADAM())
θmap = find_mode(bnn, 32, 1000, opt) ## ~Error based on the likelihood function
println("MAP estimation complete.")

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

# Calculate RMSE of volatility estimates
rmse_vol = zeros(N)
for i in 1:N
    true_sigmas = simulated_σ[window_size+1:train_index[end], i]
    rmse_vol[i] = sqrt(mean(abs2, true_sigmas .- sigmas_train[:, i]))
end

println("Volatility RMSE: ", rmse_vol)

# Plot actual vs. estimated volatility for each series
for i in 1:N
    p = plot(1:size(sigmas_train, 1), simulated_σ[window_size+1:train_index[end], i], 
        label="Actual σ series $i", legend=:topleft)
    plot!(p, 1:size(sigmas_train, 1), sigmas_train[:, i], 
        label="Estimated σ series $i")
    display(p)
end

# MCMC sampling
println("Starting MCMC sampling...")
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)
ch = mcmc(bnn, 32, 10_000, sampler, θstart=θmap)
ch = ch[:, end-5_000+1:end]  # Discard burn-in
chain = Chains(ch')
println("MCMC sampling complete.")

# Function to get volatility predictions from MCMC samples
function dcc_vol_prediction(bnn, draws::Array{T, 2}; x=bnn.x) where {T}
    n_draws = size(draws, 2)
    n_samples = size(x, 1)
    n_features = size(x, 3)
    
    vols = Array{T, 3}(undef, n_samples, n_features, n_draws)
    
    Threads.@threads for i in 1:n_draws
        net = bnn.like.nc(draws[:, i])
        for j in 1:n_samples
            out = net(x[j, :, :])
            vols[j, :, i] = exp.(out[n_features+1:2*n_features] ./ 2)
        end
    end
    
    return vols
end

# Get volatility predictions from MCMC
println("Computing posterior predictive...")
vol_predictions = dcc_vol_prediction(bnn, ch)

# Calculate mean and quantiles of volatility predictions
vol_mean = mean(vol_predictions, dims=3)[:, :, 1]
vol_lower = mapslices(x -> quantile(x, 0.025), vol_predictions, dims=3)[:, :, 1]
vol_upper = mapslices(x -> quantile(x, 0.975), vol_predictions, dims=3)[:, :, 1]

# Plot volatility with uncertainty bands for each series
for i in 1:N
    p = plot(1:size(vol_mean, 1), simulated_σ[window_size+1:train_index[end], i], 
        label="Actual σ series $i", legend=:topleft)
    plot!(p, 1:size(vol_mean, 1), vol_mean[:, i], 
        label="Mean estimated σ")
    plot!(p, 1:size(vol_mean, 1), vol_lower[:, i], 
        label="Lower 95% CI", linestyle=:dash)
    plot!(p, 1:size(vol_mean, 1), vol_upper[:, i], 
        label="Upper 95% CI", linestyle=:dash)
    display(p)
end

# Make sequences for the test set
x_test = Array{Float32, 3}(undef, size(y_test, 1) - window_size, window_size, N)
y_test_target = Array{Float32, 2}(undef, size(y_test, 1) - window_size, N)

for i in 1:(size(y_test, 1) - window_size)
    x_test[i, :, :] = y_test[i:(i+window_size-1), :]
    y_test_target[i, :] = y_test[i+window_size, :]
end

# Function to calculate one-step ahead forecasts on the test set
function forecast_test_set(bnn, ch, x_test)
    n_draws = size(ch, 2)
    n_samples = size(x_test, 1)
    n_features = size(x_test, 3)
    
    means_all = zeros(Float32, n_samples, n_features, n_draws)
    sigmas_all = zeros(Float32, n_samples, n_features, n_draws)
    
    Threads.@threads for i in 1:n_draws
        net = bnn.like.nc(ch[:, i])
        for j in 1:n_samples
            out = net(x_test[j, :, :])
            means_all[j, :, i] = out[1:n_features]
            sigmas_all[j, :, i] = exp.(out[n_features+1:2*n_features] ./ 2)
        end
    end
    
    # Calculate summary statistics
    means_mean = mean(means_all, dims=3)[:, :, 1]
    means_lower = mapslices(x -> quantile(x, 0.025), means_all, dims=3)[:, :, 1]
    means_upper = mapslices(x -> quantile(x, 0.975), means_all, dims=3)[:, :, 1]
    
    sigmas_mean = mean(sigmas_all, dims=3)[:, :, 1]
    sigmas_lower = mapslices(x -> quantile(x, 0.025), sigmas_all, dims=3)[:, :, 1]
    sigmas_upper = mapslices(x -> quantile(x, 0.975), sigmas_all, dims=3)[:, :, 1]
    
    return Dict(
        "means" => Dict("mean" => means_mean, "lower" => means_lower, "upper" => means_upper),
        "sigmas" => Dict("mean" => sigmas_mean, "lower" => sigmas_lower, "upper" => sigmas_upper)
    )
end

# Forecast test set
println("Forecasting test set...")
test_forecasts = forecast_test_set(bnn, ch, x_test)

# Calculate test set RMSE
test_rmse_returns = zeros(N)
test_rmse_vol = zeros(N)

for i in 1:N
    test_rmse_returns[i] = sqrt(mean(abs2, y_test_target[:, i] .- test_forecasts["means"]["mean"][:, i]))
    test_rmse_vol[i] = sqrt(mean(abs2, simulated_σ[train_index[end]+window_size+1:end, i] .- test_forecasts["sigmas"]["mean"][:, i]))
end

println("Test set returns RMSE: ", test_rmse_returns)
println("Test set volatility RMSE: ", test_rmse_vol)

# Plot test set forecasts
for i in 1:N
    # Returns
    p1 = plot(1:size(y_test_target, 1), y_test_target[:, i], 
        label="Actual returns", legend=:topleft)
    plot!(p1, 1:size(test_forecasts["means"]["mean"], 1), test_forecasts["means"]["mean"][:, i], 
        label="Predicted returns")
    plot!(p1, 1:size(test_forecasts["means"]["mean"], 1), test_forecasts["means"]["lower"][:, i], 
        label="Lower 95% CI", linestyle=:dash)
    plot!(p1, 1:size(test_forecasts["means"]["mean"], 1), test_forecasts["means"]["upper"][:, i], 
        label="Upper 95% CI", linestyle=:dash)
    title!(p1, "Series $i Returns Forecasts")
    display(p1)
    
    # Volatility
    p2 = plot(1:size(y_test_target, 1), simulated_σ[train_index[end]+window_size+1:end, i], 
        label="Actual volatility", legend=:topleft)
    plot!(p2, 1:size(test_forecasts["sigmas"]["mean"], 1), test_forecasts["sigmas"]["mean"][:, i], 
        label="Predicted volatility")
    plot!(p2, 1:size(test_forecasts["sigmas"]["mean"], 1), test_forecasts["sigmas"]["lower"][:, i], 
        label="Lower 95% CI", linestyle=:dash)
    plot!(p2, 1:size(test_forecasts["sigmas"]["mean"], 1), test_forecasts["sigmas"]["upper"][:, i], 
        label="Upper 95% CI", linestyle=:dash)
    title!(p2, "Series $i Volatility Forecasts")
    display(p2)
end

# Function to calculate Value-at-Risk and Expected Shortfall
function calculate_risk_metrics(forecasts, confidence_level=0.95)
    n_samples = size(forecasts["means"]["mean"], 1)
    n_features = size(forecasts["means"]["mean"], 2)
    n_draws = size(ch, 2)
    
    # Generate samples from the predictive distribution
    samples = zeros(Float32, n_samples, n_features, 1000)
    
    for i in 1:n_samples
        for j in 1:n_features
            # For each prediction point, generate samples using the mean and stdev
            for k in 1:1000
                # Randomly select one MCMC draw
                draw_idx = rand(1:n_draws)
                net = bnn.like.nc(ch[:, draw_idx])
                out = net(x_test[i, :, :])
                
                # Extract mean and sigma
                mean_val = out[j]
                sigma_val = exp(out[j+n_features] / 2)
                
                # Sample from normal distribution
                samples[i, j, k] = mean_val + sigma_val * randn()
            end
        end
    end
    
# Calculate VaR and ES
VaR = zeros(Float32, n_samples, n_features)
ES = zeros(Float32, n_samples, n_features)

for i in 1:n_samples
    for j in 1:n_features
        # Extract samples for this point and feature
        point_samples = samples[i, j, :]
        
        # Calculate VaR as the negative of the quantile (for losses)
        VaR[i, j] = -quantile(point_samples, 1 - confidence_level)
        
        # Calculate ES as the mean of losses beyond VaR
        losses = -point_samples
        ES[i, j] = mean(losses[losses .>= -VaR[i, j]])
    end
end

return Dict("VaR" => VaR, "ES" => ES)
end

# Calculate risk metrics for test set
println("Calculating risk metrics...")
risk_metrics = calculate_risk_metrics(test_forecasts)

# Plot VaR and actual returns
for i in 1:N
p = plot(1:size(y_test_target, 1), y_test_target[:, i], 
    label="Actual returns", legend=:topleft)
plot!(p, 1:size(risk_metrics["VaR"], 1), -risk_metrics["VaR"][:, i], 
    label="Value-at-Risk (95%)", linestyle=:dash, linewidth=2, color=:red)

# Count and display VaR exceptions
exceptions = (y_test_target[:, i] .< -risk_metrics["VaR"][:, i])
exception_rate = sum(exceptions) / length(exceptions)

title!(p, "Series $i Returns vs VaR (Exception rate: $(round(exception_rate*100, digits=2))%)")
display(p)
end

# Calculate correlation forecasts from the posterior
function forecast_correlations(bnn, ch, x_test)
n_draws = min(size(ch, 2), 100)  # Use up to 100 draws to save computation
n_samples = size(x_test, 1)
n_features = size(x_test, 3)

correlations = zeros(Float32, n_samples, n_draws)

for i in 1:n_draws
    θnet = ch[:, i]
    θlike = θnet[bnn.start_θlike:end]  # Extract DCC parameters
    
    # Transform DCC parameters
    a, b = transform_ab(θlike[1], θlike[2])
    
    # Initialize network
    net = bnn.like.nc(θnet)
    
    # Use all training data for DCC recursion to get to the current state
    all_x = vcat(bnn.x, x_test[1:1, :, :])  # Add first test observation
    
    # Process all data with the network
    μ = Matrix{Float32}(undef, n_features, size(all_x, 1))
    σ = similar(μ)
    z = similar(μ)
    
    # Get standardized residuals for all observations
    for t in 1:size(all_x, 1)
        out = net(all_x[t, :, :])
        μ[:, t] = out[1:n_features]
        σ[:, t] = exp.(out[n_features+1:2*n_features] ./ 2)
        
        if t <= size(bnn.y, 1)
            # For training data, use actual values
            z[:, t] = (bnn.y[t, :] .- μ[:, t]) ./ σ[:, t]
        else
            # For first test point, use random normal
            z[:, t] = randn(n_features)
        end
    end
    
    # Compute unconditional correlation matrix
    Qbar = (z * z') / size(z, 2)
    
    # Ensure Qbar is correlation matrix
    d_inv = 1 ./ sqrt.(diag(Qbar))
    Qbar = Symmetric(Diagonal(d_inv) * Qbar * Diagonal(d_inv))
    
    # Initialize Q as the unconditional correlation
    Q = copy(Qbar)
    
    # Run DCC recursion through all training data
    for t in 1:size(bnn.x, 1)
        Q = (1-a-b) * Qbar + a * (z[:,t] * z[:,t]') + b * Q
    end
    
    # Now forecast correlations for test set
    for t in 1:n_samples
        if t > 1
            # Update Q using previous standardized residuals
            if t-1 == 1
                # Use the last training point's z
                prev_z = z[:, end]
            else
                # Use this forecast point's z
                out = net(x_test[t-1, :, :])
                μt = out[1:n_features]
                σt = exp.(out[n_features+1:2*n_features] ./ 2)
                prev_z = randn(n_features)  # For forecasting, use standard normal
            end
            
            Q = (1-a-b) * Qbar + a * (prev_z * prev_z') + b * Q
        end
        
        # Convert Q to correlation matrix
        d_inv = 1 ./ sqrt.(diag(Q))
        R = Diagonal(d_inv) * Q * Diagonal(d_inv)
        
        # Store correlation (for bivariate case, the off-diagonal element)
        correlations[t, i] = R[1, 2]
    end
end

# Calculate summary statistics
corr_mean = mean(correlations, dims=2)[:,1]
corr_lower = mapslices(x -> quantile(x, 0.025), correlations, dims=2)[:,1]
corr_upper = mapslices(x -> quantile(x, 0.975), correlations, dims=2)[:,1]

return Dict("mean" => corr_mean, "lower" => corr_lower, "upper" => corr_upper)
end

# If we have a bivariate series, forecast correlations
if N == 2
println("Forecasting correlations...")
corr_forecasts = forecast_correlations(bnn, ch, x_test)

# Plot correlation forecasts
p = plot(1:size(corr_forecasts["mean"], 1), corr_forecasts["mean"], 
    label="Mean correlation", legend=:topleft)
plot!(p, 1:size(corr_forecasts["mean"], 1), corr_forecasts["lower"], 
    label="Lower 95% CI", linestyle=:dash)
plot!(p, 1:size(corr_forecasts["mean"], 1), corr_forecasts["upper"], 
    label="Upper 95% CI", linestyle=:dash)
title!(p, "Forecasted Correlation")
display(p)
end

# Function to perform portfolio optimization using the forecasts
function optimize_portfolio(forecasts, corr_forecasts, risk_aversion=3.0)
n_samples = size(forecasts["means"]["mean"], 1)

# Only works for bivariate case (N=2)
weights = zeros(Float32, n_samples, 2)

for t in 1:n_samples
    # Get expected returns
    μ = forecasts["means"]["mean"][t, :]
    
    # Get variance-covariance matrix
    σ = forecasts["sigmas"]["mean"][t, :]
    ρ = corr_forecasts["mean"][t]
    
    Σ = [σ[1]^2 ρ*σ[1]*σ[2]; 
         ρ*σ[1]*σ[2] σ[2]^2]
    
    # Markowitz optimization with no constraints
    # w ∝ Σ^(-1) * μ
    w = inv(Σ) * μ / risk_aversion
    
    # Normalize weights to sum to 1 if they're positive
    if all(w .> 0)
        w = w / sum(w)
    else
        # If we have negative weights, constrain between 0 and 1
        w = max.(0, min.(1, w))
        w = w / sum(w)
    end
    
    weights[t, :] = w
end

return weights
end

# If we have a bivariate series, perform portfolio optimization
if N == 2
println("Performing portfolio optimization...")
portfolio_weights = optimize_portfolio(test_forecasts, corr_forecasts)

# Plot portfolio weights
p = plot(1:size(portfolio_weights, 1), portfolio_weights[:, 1], 
    label="Weight of Asset 1", legend=:topleft)
plot!(p, 1:size(portfolio_weights, 1), portfolio_weights[:, 2], 
    label="Weight of Asset 2")
title!(p, "Optimal Portfolio Weights")
display(p)

# Calculate portfolio returns
portfolio_returns = zeros(Float32, size(portfolio_weights, 1))

for t in 1:size(portfolio_weights, 1)
    portfolio_returns[t] = sum(portfolio_weights[t, :] .* y_test_target[t, :])
end

# Plot portfolio returns
p = plot(1:size(portfolio_returns, 1), portfolio_returns, 
    label="Portfolio returns", legend=:topleft)
title!(p, "Realized Portfolio Returns")
display(p)

# Calculate cumulative returns
cumulative_returns = cumprod(1 .+ portfolio_returns) .- 1

# Plot cumulative returns
p = plot(1:size(cumulative_returns, 1), cumulative_returns, 
    label="Portfolio", legend=:topleft)
plot!(p, 1:size(cumulative_returns, 1), cumprod(1 .+ y_test_target[:, 1]) .- 1, 
    label="Asset 1")
plot!(p, 1:size(cumulative_returns, 1), cumprod(1 .+ y_test_target[:, 2]) .- 1, 
    label="Asset 2")
title!(p, "Cumulative Returns")
display(p)
end

# Summary statistics 
println("\n----- Summary Statistics -----")
println("Training set volatility RMSE: ", rmse_vol)
println("Test set returns RMSE: ", test_rmse_returns)
println("Test set volatility RMSE: ", test_rmse_vol)

if N == 2
# Calculate portfolio performance metrics
portfolio_mean_return = mean(portfolio_returns)
portfolio_volatility = std(portfolio_returns)
portfolio_sharpe = portfolio_mean_return / portfolio_volatility * sqrt(252)  # Annualized

println("\n----- Portfolio Performance -----")
println("Mean daily return: ", portfolio_mean_return)
println("Daily volatility: ", portfolio_volatility)
println("Annualized Sharpe ratio: ", portfolio_sharpe)

# Calculate VaR coverage
var_coverage = mean((y_test_target .< -risk_metrics["VaR"]), dims=1)[1, :]
expected_coverage = 0.05  # 5% for 95% VaR

println("\n----- Risk Model Evaluation -----")
println("VaR coverage (expected $(expected_coverage*100)%): ", var_coverage * 100, "%")

# Perform Kupiec unconditional coverage test
function kupiec_test(exceptions, n, p=0.05)
    x = sum(exceptions)
    if x == 0
        return 0.0  # Can't perform test with no exceptions
    end
    
    likelihood_ratio = -2 * log((1-p)^(n-x) * p^x) + 2 * log((1-x/n)^(n-x) * (x/n)^x)
    p_value = 1 - cdf(Chisq(1), likelihood_ratio)
    
    return p_value
end

for i in 1:N
    exceptions = (y_test_target[:, i] .< -risk_metrics["VaR"][:, i])
    p_value = kupiec_test(exceptions, length(exceptions))
    
    println("Kupiec test p-value for series $i: ", p_value)
end
end

println("\nDCC-GARCH analysis complete!")