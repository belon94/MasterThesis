using Flux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors
using BayesFlux
Random.seed!(1212)

# Sample size
n = 500

# Define DCC-GARCH parameters
# For a bivariate series (N=2)
N = 2  # Number of series 
vol = [(20^2)/252, (15^2)/252]  # Different volatilities for each series
α = 0.3  # DCC parameter
β = 0.6  # DCC parameter
ρ_unconditional = 0.5  # Unconditional correlation

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
    
    # Initialize series
    L = zeros(n, N)  # Returns
    σ_2 = zeros(n+1, N)  # Conditional variances
    σ_2[1,:] = vol
    Qt = copy(Rbar)  # Initial Q matrix
    Rt = similar(Qt)  # Correlation matrix
    z = zeros(n, N)  # Standardized residuals
    
    # Generate data
    for t in 1:n
        # For first observation, use initial values
        if t == 1
            # Calculate correlation matrix
            d_inv = 1 ./ sqrt.(diag(Qt))
            Rt = Diagonal(d_inv) * Qt * Diagonal(d_inv)
            
            # Generate correlated standard normal variables
            std_noise = rand(MvNormal(zeros(N), Rt))
            
            # Compute returns
            for i in 1:N
                L[t,i] = sqrt(σ_2[t,i]) * std_noise[i]
                z[t,i] = std_noise[i]
            end
            
            # Update conditional variances for next period
            for i in 1:N
                σ_2[t+1,i] = ω[i] + α * L[t,i]^2 + β * σ_2[t,i]
            end
        else
            # Update Q matrix using DCC recursion
            Qt = (1-α-β) * Rbar + α * (z[t-1,:] * z[t-1,:]') + β * Qt
            
            # Calculate correlation matrix
            d_inv = 1 ./ sqrt.(diag(Qt))
            Rt = Diagonal(d_inv) * Qt * Diagonal(d_inv)
            
            # Generate correlated standard normal variables
            std_noise = rand(MvNormal(zeros(N), Rt))
            
            # Compute returns
            for i in 1:N
                L[t,i] = sqrt(σ_2[t,i]) * std_noise[i]
                z[t,i] = std_noise[i]
            end
            
            # Update conditional variances for next period
            for i in 1:N
                σ_2[t+1,i] = ω[i] + α * L[t,i]^2 + β * σ_2[t,i]
            end
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

# Prepare data for RNN
window_size = 5  # Lookback window
# Create sequences for training
x_train = zeros(Float32, window_size, size(y_train, 1)-window_size, N)
y_train_target = zeros(Float32, size(y_train, 1)-window_size, N)

for i in 1:(size(y_train, 1)-window_size)
    x_train[:, i, :] = y_train[i:(i+window_size-1), :]
    y_train_target[i, :] = y_train[i+window_size, :]
end

# Setup neural network for DCC-GARCH
# We need to output 2*N parameters (mean and log-variance for each series)
net = Chain(
    RNN(N, 10),  # RNN layer: N inputs, 10 hidden units
    Dense(10, 20, relu),  # Hidden layer
    Dense(20, 2*N)  # Output layer: mean and log-variance for each series
)

nc = destruct(net)

# Use DCCGarchNormal likelihood (assuming we've implemented it)
like = DCCGarchNormal(nc, Normal(0, 0.5), N)
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)

# Create BNN
bnn = BNN(x_train, y_train_target, like, prior, init)

# Find MAP estimate
opt = FluxModeFinder(bnn, Flux.RMSProp())
θmap = find_mode(bnn, 10, 2000, opt)

# Get predictions from MAP estimate
nethat = nc(θmap)
parameters = [nethat(x_train[:, i, :]) for i in 1:size(x_train, 2)]

# Extract means and log-variances
means = hcat([param[1:N] for param in parameters]...)
log_vars = hcat([param[N+1:2*N] for param in parameters]...)
σ_hat = exp.(log_vars ./ 2)  # Convert to standard deviations

# Calculate RMSE of volatility estimates
rmse_vol = zeros(N)
for i in 1:N
    rmse_vol[i] = sqrt(mean(abs2, simulated_σ[window_size+1:train_index[end], i] .- σ_hat[i, :]))
end

println("Volatility RMSE: ", rmse_vol)

# Plot actual vs. estimated volatility for each series
for i in 1:N
    p = plot(1:size(σ_hat, 2), simulated_σ[window_size+1:train_index[end], i], 
        label="Actual σ series $i", legend=:topleft)
    plot!(p, 1:size(σ_hat, 2), σ_hat[i, :], 
        label="Estimated σ series $i")
    display(p)
end

# Plot correlation
# We would need to extract correlation estimates from the DCC model
# This requires additional implementation of methods to extract correlation matrices

# MCMC sampling
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)
ch = mcmc(bnn, 10, 20_000, sampler, θstart=θmap)
ch = ch[:, end-10_000+1:end]  # Discard burn-in
chain = Chains(ch')

# Function to get volatility predictions from MCMC samples
function dcc_vol_prediction(bnn, draws::Array{T, 2}; x=bnn.x) where {T}
    vols = Array{T, 3}(undef, N, size(x, 2), size(draws, 2))
    Threads.@threads for i in 1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        for j in 1:size(x, 2)
            params = net(x[:, j, :])
            vols[:, j, i] = exp.(params[N+1:2*N] ./ 2)
        end
    end
    return vols
end

# Get volatility predictions from MCMC
vol_predictions = dcc_vol_prediction(bnn, ch)

# Calculate mean and quantiles of volatility predictions
vol_mean = mean(vol_predictions, dims=3)[:, :, 1]
vol_lower = mapslices(x -> quantile(x, 0.025), vol_predictions, dims=3)[:, :, 1]
vol_upper = mapslices(x -> quantile(x, 0.975), vol_predictions, dims=3)[:, :, 1]

# Plot volatility with uncertainty bands for each series
for i in 1:N
    p = plot(1:size(vol_mean, 2), simulated_σ[window_size+1:train_index[end], i], 
        label="Actual σ series $i", legend=:topleft)
    plot!(p, 1:size(vol_mean, 2), vol_mean[i, :], 
        label="Mean estimated σ")
    plot!(p, 1:size(vol_mean, 2), vol_lower[i, :], 
        label="Lower 95% CI", linestyle=:dash)
    plot!(p, 1:size(vol_mean, 2), vol_upper[i, :], 
        label="Upper 95% CI", linestyle=:dash)
    display(p)
end

# Function to forecast future values
function forecast_dcc(bnn, draws::Array{T, 2}, last_observations, steps_ahead) where {T}
    N = size(last_observations, 2)
    forecasts_mean = zeros(steps_ahead, N)
    forecasts_vol = zeros(steps_ahead, N)
    forecasts_corr = zeros(steps_ahead, N, N)
    
    # For each MCMC draw
    for i in 1:size(draws, 2)
        θnet = draws[:, i]
        θlike = θnet[bnn.start_θlike:end]  # Extract DCC parameters
        
        # Transform DCC parameters
        a, b = transform_ab(θlike[1], θlike[2])
        
        # Create forecasts using the trained model
        net = bnn.like.nc(θnet)
        
        # Initialize with last known observations
        current_obs = copy(last_observations)
        
        for step in 1:steps_ahead
            # Use the model to predict next step
            x = current_obs[end-window_size+1:end, :]
            params = net(x)
            
            means = params[1:N]
            log_vars = params[N+1:2*N]
            vols = exp.(log_vars ./ 2)
            
            # Store forecasts
            forecasts_mean[step, :] += means / size(draws, 2)
            forecasts_vol[step, :] += vols / size(draws, 2)
            
            # Generate next observation for recursive forecasting
            # (This is a simplified version; in practice, we would sample from the full multivariate distribution)
            next_obs = means + vols .* randn(N)
            current_obs = vcat(current_obs[2:end, :], reshape(next_obs, 1, N))
        end
    end
    
    return Dict(
        "mean" => forecasts_mean,
        "vol" => forecasts_vol
    )
end

# Last observations from training set
last_obs = y_train[end-window_size+1:end, :]

# Generate forecasts
forecast_horizon = 20
forecasts = forecast_dcc(bnn, ch, last_obs, forecast_horizon)

# Plot forecasts vs actual values
for i in 1:N
    p = plot(1:size(y_test, 1), y_test[:, i], 
        label="Actual returns", legend=:topleft)
    plot!(p, 1:forecast_horizon, forecasts["mean"][:, i], 
        label="Forecasted mean")
    display(p)
    
    p2 = plot(1:size(y_test, 1), simulated_σ[train_index[end]+1:end, i], 
        label="Actual volatility", legend=:topleft)
    plot!(p2, 1:forecast_horizon, forecasts["vol"][:, i], 
        label="Forecasted volatility")
    display(p2)
end