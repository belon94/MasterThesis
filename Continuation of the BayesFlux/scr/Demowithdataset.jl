using Flux
using BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors
using CSV, DataFrames

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

# Data preparation function for BayesFlux
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

# Implementation of the likelihood function for flattened input
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

# Posterior prediction implementation
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

# Load ETF data
etf_rf = ("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv")
df = CSV.read(etf_rf, DataFrame)
etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418",
"16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460",
"24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277",
"28278", "28279", "28280", "31372", "31466"]

# Data preprocessing function
function preprocess_data(df, etf_names)
    # Handle missing values by replacing with the mean of the column
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
     
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)
    
    # Calculate mean returns for each asset
    μ = mean(returns, dims=1)
    return etf_returns, rf_returns, returns, vec(μ)
end

# Preprocess the data
etf_returns, rf_returns, returns, mean_returns = preprocess_data(df, etf_names)

# Choose the number of ETFs to model (start with 5)
N = 2  # Number of ETFs to use
window_size = 5 

# Extract the selected ETFs
selected_returns = etf_returns[:, 1:N]

# Split into training and testing sets (80% train, 20% test)
n_samples = size(selected_returns, 1)
train_size = Int(floor(0.8 * n_samples))

y_train = Float32.(selected_returns[1:train_size, :])
y_test = Float32.(selected_returns[(train_size+1):end, :])
y_full = Float32.(selected_returns)

# Create BNN-ready data using the prepare_time_series_data function
x_train, y_train_target = prepare_time_series_data(y_train, window_size)

# Create neural network architecture adapted for the ETF data
net = Chain(
    Dense(N * window_size, 20, relu),  # Input layer handles flattened window
    Dense(20, 20, relu),               # Hidden layer
    Dense(20, 2*N)                     # Output: mean and log-variance for each ETF
)

nc = destruct(net)

# Use the DCCGarchNormal likelihood
like = DCCGarchNormal(nc, Normal(0, 0.5), N)
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)

# Create and train the BNN
bnn = BNN(x_train, y_train_target, like, prior, init)

# Find MAP estimate
opt = FluxModeFinder(bnn, Flux.ADAM())
θmap = find_mode(bnn, 2, 500, opt)  # 2 likelihood params, 500 iterations

# Get the trained network
nethat = nc(θmap)

# Get predictions for training set
means_train, sigmas_train = extract_predictions(nethat, x_train)

# Create test sequences
x_test, y_test_target = prepare_time_series_data(y_test, window_size)

# Get predictions for test set
means_test, sigmas_test = extract_predictions(nethat, x_test)

# Prepare indices for plotting (accounting for window size)
train_plot_indices = (window_size+1):length(y_train)
test_plot_indices = (1+window_size):(size(y_test, 1))

# Plot results for the first two ETFs (training data)
p1 = plot(train_plot_indices, y_train[train_plot_indices, 1], label="Actual Returns (Train)", color=:blue, 
          title="ETF $(etf_names[1])", xlabel="Time", ylabel="Returns")
plot!(p1, train_plot_indices, means_train[:, 1], label="Predicted Mean (Train)", color=:red)
plot!(p1, train_plot_indices, [means_train[:, 1] + 2*sigmas_train[:, 1], means_train[:, 1] - 2*sigmas_train[:, 1]], 
      label=["95% CI" nothing], color=:red, alpha=0.3, linestyle=:dash)

p2 = plot(train_plot_indices, y_train[train_plot_indices, 2], label="Actual Returns (Train)", color=:blue, 
          title="ETF $(etf_names[2])", xlabel="Time", ylabel="Returns")
plot!(p2, train_plot_indices, means_train[:, 2], label="Predicted Mean (Train)", color=:red)
plot!(p2, train_plot_indices, [means_train[:, 2] + 2*sigmas_train[:, 2], means_train[:, 2] - 2*sigmas_train[:, 2]], 
      label=["95% CI" nothing], color=:red, alpha=0.3, linestyle=:dash)

plot(p1, p2, layout=(2,1), size=(800, 600))

# Plot results for the first two ETFs (test data)
p3 = plot(test_plot_indices, y_test[test_plot_indices, 1], label="Actual Returns (Test)", color=:blue, 
          title="ETF $(etf_names[1]) - Test Set", xlabel="Time", ylabel="Returns")
plot!(p3, test_plot_indices, means_test[:, 1], label="Predicted Mean (Test)", color=:red)
plot!(p3, test_plot_indices, [means_test[:, 1] + 2*sigmas_test[:, 1], means_test[:, 1] - 2*sigmas_test[:, 1]], 
      label=["95% CI" nothing], color=:red, alpha=0.3, linestyle=:dash)

p4 = plot(test_plot_indices, y_test[test_plot_indices, 2], label="Actual Returns (Test)", color=:blue, 
          title="ETF $(etf_names[2]) - Test Set", xlabel="Time", ylabel="Returns")
plot!(p4, test_plot_indices, means_test[:, 2], label="Predicted Mean (Test)", color=:red)
plot!(p4, test_plot_indices, [means_test[:, 2] + 2*sigmas_test[:, 2], means_test[:, 2] - 2*sigmas_test[:, 2]], 
      label=["95% CI" nothing], color=:red, alpha=0.3, linestyle=:dash)

plot(p3, p4, layout=(2,1), size=(800, 600))

# Extract and analyze the DCC parameters
a, b = transform_ab(θmap[end-1], θmap[end])
println("Estimated DCC parameters: a = $a, b = $b")
println("Persistence (a + b) = $(a + b)")

# Calculate model performance metrics
function calculate_metrics(actual, predicted, sigmas)
    mse = mean((actual .- predicted).^2)
    rmse = sqrt(mse)
    mae = mean(abs.(actual .- predicted))
    
    # Calculate proportion of actual values within 95% CI
    lower_bound = predicted .- 1.96 .* sigmas
    upper_bound = predicted .+ 1.96 .* sigmas
    within_ci = mean((actual .>= lower_bound) .& (actual .<= upper_bound))
    
    return Dict("MSE" => mse, "RMSE" => rmse, "MAE" => mae, "Within_CI" => within_ci)
end

# Calculate metrics for each ETF in test set
for i in 1:N
    metrics = calculate_metrics(
        y_test[test_plot_indices, i], 
        means_test[:, i], 
        sigmas_test[:, i]
    )
    println("ETF $(etf_names[i]) Test Metrics: $metrics")
end

# You can also analyze correlation structures
# Extract the conditional correlation matrix at the final time point
final_inputs = x_train[:, end]
final_outputs = nethat(final_inputs)
final_μ = final_outputs[1:N]
final_σ = exp.(final_outputs[N+1:2*N] ./ 2)

println("Final Conditional Means:")
println(final_μ)
println("Final Conditional Volatilities:")
println(final_σ)

# Calculate the correlation matrix
correlation_matrix = zeros(N, N)
for i in 1:N
    for j in 1:N
        correlation_matrix[i, j] = final_μ[i] * final_μ[j] / (final_σ[i] * final_σ[j])
    end
end
println("Final Conditional Correlation Matrix:")
println(correlation_matrix)
# Plot the correlation matrix
heatmap(correlation_matrix, title="Final Conditional Correlation Matrix", 
        xlabel="Assets", ylabel="Assets", color=:viridis, aspect_ratio=1)




###############################

# DCC-GARCH Bayesian Neural Network for ETF Returns
using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames

Random.seed!(1212)

# Load data
etf_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"
df = CSV.read(etf_rf, DataFrame)

etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", 
             "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", 
             "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", 
             "28278", "28279", "28280", "31372", "31466"]

# Data preprocessing with mean imputation
function preprocess_data(df, etf_names)
    # Handle missing values by replacing with the mean of the column
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)
    
    # Calculate mean returns for each asset
    μ = mean(returns, dims=1)
    
    return etf_returns, rf_returns, returns, vec(μ)
end

# Helper functions from the example code
sigmoid(x) = 1/(1+exp(-x))
transform_ab(a,b) = let a_=sigmoid(a); b_=sigmoid(b)*(1-a_); (a_,b_) end
nearest_pd(A) = (A + A')/2 + 1e-4I  # Quick positive definite repair

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

# DCC-GARCH likelihood function (same as in the example)
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

# Main implementation

# Step 1: Preprocess the ETF data
etf_returns, rf_returns, returns, μ = preprocess_data(df, etf_names)

# Step 2: Select a subset of ETFs for computational efficiency
N = 5  # Use first 5 ETFs for demonstration (can be adjusted)
selected_etfs = etf_returns[:, 1:N]

# Step 3: Split data into training and testing sets
total_samples = size(selected_etfs, 1)
train_idx = 1:Int(round(0.8 * total_samples))
test_idx = (train_idx[end] + 1):total_samples

# Step 4: Prepare time series data with window size 22
window = 22
Xtr, Ytr = prepare_time_series_data(Float32.(selected_etfs[train_idx, :]), window)
Xte, Yte = prepare_time_series_data(Float32.(selected_etfs[test_idx, :]), window)

# Step 5: Build the neural network
net = Chain(
    Dense(N*window, 30, relu),
    Dense(30, 30, relu),
    Dense(30, 2N)  # Output: N means + N log variances
)

nc = destruct(net)
like = DCCGarchNormal(nc, Normal(0, 0.5), N)
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0f0, 0.5f0), like, prior)
bnn = BNN(Xtr, Ytr, like, prior, init)

# Step 6: Find the posterior mode
println("Finding posterior mode...")
θmap = find_mode(bnn, 50, 500, FluxModeFinder(bnn, Flux.ADAM()))
println("Posterior mode found.")



# Step 7: Posterior sampling with SGLD
println("Starting SGLD sampling...")
sgld = SGLD(Float32;
            stepsize_a = 5f-4,  # Smaller stepsize for stability
            stepsize_b = 0f0,
            stepsize_γ = 0.55f0)

draws = mcmc(bnn, 128, 10_000, sgld)
draws = draws[:, 5_001:end]  # Discard first 5000 as burn-in

# Step 8: Remove any draws containing NaN or Inf values
good = map(i -> all(isfinite, view(draws, :, i)), axes(draws, 2))
draws = draws[:, good]
samples = [draws[:, i] for i in axes(draws, 2)]

println("SGLD sampling completed. Number of valid samples: $(length(samples))")

# Step 9: Diagnostics
param_names = ["θ_$i" for i in 1:size(draws, 1)]
chn = Chains(permutedims(draws, (2, 1)), param_names)
println(summarystats(chn))


###############################################################################
#  DCC–GARCH Bayesian Sequential LSTM  (BayesFlux.jl) - REAL ETF DATA        #
###############################################################################
using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames

Random.seed!(1212)

# ───────────────────────── 1.  DCC-GARCH simulator (keep for reference) ──────────────────────────
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

# ───────────────────────── 4. Load and Preprocess Real ETF Data ────────────────────────────
# Load data
etf_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"
df = CSV.read(etf_rf, DataFrame)
etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", 
             "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", 
             "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", 
             "28278", "28279", "28280", "31372", "31466"]

# Data preprocessing with mean imputation
function preprocess_data(df, etf_names)
    # Handle missing values by replacing with the mean of the column
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)
    
    # Calculate mean returns for each asset
    μ = mean(returns, dims=1)
    
    return etf_returns, rf_returns, returns, vec(μ)
end

# Preprocess the data
etf_returns, rf_returns, returns, μ_returns = preprocess_data(df, etf_names)

# Use the combined returns (29 ETFs + 1 risk-free asset)
y = Float32.(returns)  # All 30 assets
n, N = size(y)         # n = number of time periods, N = 30 assets

println("=== Data Summary ===")
println("Number of observations: ", n)
println("Number of assets: ", N)
println("ETFs: ", length(etf_names))
println("Data range: ", size(y))
println("Mean returns: ", round.(μ_returns[1:5], digits=6), "... (showing first 5)")

# Define training set - use first 80% for training
train_size = Int(floor(0.8 * n))
train_idx = 1:train_size
test_idx = (train_size+1):n

println("Training observations: ", length(train_idx))
println("Test observations: ", length(test_idx))

# Prepare sequential data
Xtr, Ytr = prepare_sequential_data(y[train_idx,:])
Xte, Yte = prepare_sequential_data(y[test_idx,:])

println("Training data shape: X=", size(Xtr), ", Y=", size(Ytr))

# ─────────────────────── 5.  build & train Sequential LSTM BNN ────────────────
# Define Sequential LSTM architecture - adjusted for 30 assets
hidden_size = 64  # Increased hidden size for more complex data
net = Chain(
    LSTM(N => hidden_size),      # LSTM layer: 30 inputs to hidden_size outputs
    Dense(hidden_size => 2N)     # Output layer: mean and log variance for each of 30 series
)

println("=== Model Architecture ===")
println("Input dimension: ", N)
println("LSTM hidden size: ", hidden_size)
println("Output dimension: ", 2N, " (", N, " means + ", N, " log-variances)")

# Setup BNN components
nc = destruct(net)  # Network constructor
like = DCCGarchLSTMSequential(nc, Normal(0, 0.1), N)  # Sequential LSTM likelihood (smaller prior std)
prior = GaussianPrior(nc, 0.1f0)  # Tighter prior for real data
init = InitialiseAllSame(Normal(0f0, 0.1f0), like, prior)  # Parameter initialization
bnn = BNN(Xtr, Ytr, like, prior, init)  # BNN model

println("Total parameters: ", length(nc.θ))

# Find mode of posterior for initialization
println("Finding MAP estimate...")
θmap = find_mode(bnn, 100, 1000, FluxModeFinder(bnn, Flux.ADAM()))

# ─────────────────────── 6.  posterior sampling (SGNHTS) ─────────────────────
println("Starting MCMC sampling...")
# Smaller step size and adjusted parameters for real data
sampler = SGNHTS(5f-5, 1f0; xi = 1f0^2, μ = 5f0)  # Smaller step size
ch = mcmc(bnn, 20, 30_000, sampler)  # More chains, fewer samples due to computational cost
# Keep only the last 15,000 samples (discard 15,000 as burn-in)
ch = ch[:, end-15_000+1:end]

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
println("Generating predictions...")
yhats = naive_prediction_sequential_lstm(bnn, ch)
chain_yhat = Chains(yhats')

# Check convergence
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
println("Maximum R-hat: ", r_hat)

# ─────────────────────── 8. Sequential LSTM Posterior Predictive Checks ─────────────
function posterior_predict_sequential_lstm(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    n_samples = min(size(draws, 2), 1000)  # Limit samples for computational efficiency
    n_obs = size(x, 2)
    
    pred_samples = Array{T, 2}(undef, N * n_obs, n_samples)
    
    println("Generating posterior predictive samples...")
    Threads.@threads for i = 1:n_samples
        if i % 100 == 0
            println("Sample ", i, "/", n_samples)
        end
        
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
        d = 1 ./ sqrt.(max.(diag(Q̄), 1e-8))
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
            
            # Add regularization for numerical stability
            H_reg = H + 1e-4*I
            samples[:, t] = μ_all[:, t] + cholesky(Symmetric(H_reg)).L * randn(N)
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
    N = 30  # Updated for 30 assets
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
    
    if length(y_true_vector) > total_points
        y_true_vector = y_true_vector[1:total_points]
    end
    
    for i = 1:n_quantiles
        q = quantiles[i]
        threshold = [quantile(y_samples_reshaped[j, :], q) for j = 1:total_points]
        observed_quantiles[i] = sum(y_true_vector .< threshold) / total_points
    end
    
    return observed_quantiles
end

o_q = get_observed_quantiles(y[train_idx,:], posterior_yhat, t_q)

# Create quantile-quantile plot
p1 = plot(t_q, o_q, label = "Sequential LSTM Posterior Predictive", legend=:topleft,
     xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(p1, x->x, t_q, label = "45-degree line", linestyle=:dash) 
plot!(p1, title = "Sequential LSTM DCC-GARCH ETF Data Q-Q Plot", size = (800, 600))

# Display the plot
display(p1)

# ─────────────────────── 10. Additional Diagnostics ─────────────────────────
println("=== Final Model Summary ===")
println("Data points (training): ", size(Xtr, 2))
println("Assets: ", N, " (", length(etf_names), " ETFs + 1 risk-free)")
println("LSTM hidden size: ", hidden_size)
println("Total parameters: ", length(θmap))
dcc_params = transform_ab(mean(ch[end-1:end, :], dims=2)...)
println("DCC parameters (a, b): ", round.(dcc_params, digits=4))
println("Maximum R-hat: ", round(r_hat, digits=3))

# Plot some sample paths for first few ETFs
if r_hat < 1.2  # More lenient threshold for real data
    n_plot_samples = min(50, size(posterior_yhat, 2))
    sample_indices = rand(1:size(posterior_yhat, 2), n_plot_samples)
    
    # Plot first 3 ETFs
    for asset_idx in 1:min(3, N)
        p_asset = plot(title="ETF $(etf_names[asset_idx]) - Posterior Predictive", size=(800, 400))
        
        # Plot actual data
        actual_data = y[train_idx[2:end], asset_idx]
        plot!(p_asset, actual_data, label="Actual", color=:black, linewidth=2)
        
        # Plot sample predictions
        for i in 1:min(10, n_plot_samples)  # Show max 10 samples per asset
            y_sample = reshape(posterior_yhat[:, sample_indices[i]], N, :)
            plot!(p_asset, vec(y_sample[asset_idx, :]), alpha=0.2, color=:blue, 
                  label=i==1 ? "Predicted Samples" : "")
        end
        
        display(p_asset)
    end
end

println("=== Sequential LSTM DCC-GARCH ETF Model Complete ===")
