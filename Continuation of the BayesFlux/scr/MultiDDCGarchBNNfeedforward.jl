###############################################################################
#  DCC–GARCH Bayesian neural-network (BayesFlux.jl) - 2 Assets
###############################################################################
using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames

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

# ───────────────────────── 4.  Load and preprocess real data ────────────────────────
# Load data
etf_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"
df = CSV.read(etf_rf, DataFrame)

etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", 
             "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", 
             "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", 
             "28278", "28279", "28280", "31372", "31466"]

# Select first 2 assets (2 ETFs)
selected_assets = etf_names[1:2]  # First 2 ETFs
N = 2

# Data preprocessing with mean imputation
function preprocess_data_subset(df, selected_assets)
    # Handle missing values by replacing with the mean of the column
    for col in selected_assets
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    returns = Matrix{Float64}(df[!, selected_assets])  # Only selected assets
    μ = mean(returns, dims=1)
    
    return returns, vec(μ)
end

# Preprocess the data
returns, μ_returns = preprocess_data_subset(df, selected_assets)
y = Float32.(returns)  # Convert to Float32 for neural network
n, N = size(y)  # n = number of observations, N = 2 assets

println("="^50)
println("2-ASSET DCC-GARCH BNN ANALYSIS")
println("="^50)
println("Selected assets: $(selected_assets)")
println("Data dimensions: $n observations, $N assets")
println("Average returns: $(round.(μ_returns, digits=6))")

# Define training set (use 80% for training)
train_size = Int(floor(0.8 * n))
train_idx = 1:train_size
window = 22  # Use 22 lags

println("Training set: $(length(train_idx)) observations")

# Prepare training data
Xtr, Ytr = prepare_time_series_data(y[train_idx,:], window)
println("Training data prepared: X shape $(size(Xtr)), Y shape $(size(Ytr))")

# ─────────────────────── 5.  build & train BNN ──────────────────────────────
# Define neural network architecture (small for 2 assets)
net = Chain(
    Dense(N*window, 32, relu),    # Input layer (2*22=44 inputs)
    Dense(32, 16, relu),          # Hidden layer 1
    Dense(16, 2N)                 # Output layer (4 outputs: 2 means + 2 log-variances)
)

println("Neural network architecture:")
println("Input layer: $(N*window) → 32")
println("Hidden layer: 32 → 16") 
println("Output layer: 16 → $(2N)")

# Setup BNN components
nc    = destruct(net)
like  = DCCGarchNormal(nc, Normal(0, 0.2), N)  # Slightly larger prior for small N
prior = GaussianPrior(nc, 0.2f0)
init  = InitialiseAllSame(Normal(0f0, 0.2f0), like, prior)
bnn   = BNN(Xtr, Ytr, like, prior, init)

println("BNN components initialized for 2 assets")

# Find mode of posterior for initialization
println("Finding posterior mode...")
θmap = find_mode(bnn, 100, 1000, FluxModeFinder(bnn, Flux.ADAM(0.01)))
println("Posterior mode found")

# ─────────────────────── 6.  posterior sampling (SGNHTS) ─────────────────────
println("Starting MCMC sampling...")
sampler = SGNHTS(1f-4, 1f0; xi = 1f0^2, μ = 10f0)  # Larger step size for 2 assets
ch = mcmc(bnn, 10, 50_000, sampler)  # More samples since it's computationally cheap
ch = ch[:, end-20_000+1:end]  # Keep 20,000 samples

println("MCMC sampling completed")
println("Chain dimensions: $(size(ch))")

# Analysis functions (same as before but simplified output)
function naive_prediction(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    yhats = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        net = bnn.like.nc(draws[:, i])
        outs = map(t -> net(view(x, :, t)), 1:n_obs)
        μ = hcat(map(o -> o[1:N], outs)...)
        yhats[:, i] = vec(μ)
    end
    return yhats
end

yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])

println("Maximum R-hat: $(round(r_hat, digits=4))")
println("2-Asset Analysis Complete!")


###############################################################################
#  DCC–GARCH Bayesian neural-network (BayesFlux.jl) - 5 Assets  
###############################################################################
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

# Select first 5 assets (5 ETFs)
selected_assets = etf_names[1:5]  # First 5 ETFs
N = 5

# Data preprocessing with mean imputation
function preprocess_data_subset(df, selected_assets)
    for col in selected_assets
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    returns = Matrix{Float64}(df[!, selected_assets])
    μ = mean(returns, dims=1)
    return returns, vec(μ)
end

# Preprocess the data
returns, μ_returns = preprocess_data_subset(df, selected_assets)
y = Float32.(returns)
n, N = size(y)

println("="^50)
println("5-ASSET DCC-GARCH BNN ANALYSIS")
println("="^50)
println("Selected assets: $(selected_assets)")
println("Data dimensions: $n observations, $N assets")
println("Average returns: $(round.(μ_returns, digits=6))")

# Define training set
train_size = Int(floor(0.8 * n))
train_idx = 1:train_size
window = 22

println("Training set: $(length(train_idx)) observations")

# Prepare training data
Xtr, Ytr = prepare_time_series_data(y[train_idx,:], window)
println("Training data prepared: X shape $(size(Xtr)), Y shape $(size(Ytr))")

# ─────────────────────── 5.  build & train BNN ──────────────────────────────
# Define neural network architecture (medium for 5 assets)
net = Chain(
    Dense(N*window, 64, relu),    # Input layer (5*22=110 inputs)
    Dense(64, 32, relu),          # Hidden layer 1
    Dense(32, 16, relu),          # Hidden layer 2
    Dense(16, 2N)                 # Output layer (10 outputs: 5 means + 5 log-variances)
)

println("Neural network architecture:")
println("Input layer: $(N*window) → 64")
println("Hidden layers: 64 → 32 → 16") 
println("Output layer: 16 → $(2N)")

# Setup BNN components
nc    = destruct(net)
like  = DCCGarchNormal(nc, Normal(0, 0.15), N)
prior = GaussianPrior(nc, 0.15f0)
init  = InitialiseAllSame(Normal(0f0, 0.15f0), like, prior)
bnn   = BNN(Xtr, Ytr, like, prior, init)

println("BNN components initialized for 5 assets")

# Find mode of posterior for initialization
println("Finding posterior mode...")
θmap = find_mode(bnn, 100, 1000, FluxModeFinder(bnn, Flux.ADAM(0.005)))
println("Posterior mode found")

# ─────────────────────── 6.  posterior sampling (SGNHTS) ─────────────────────
println("Starting MCMC sampling...")
sampler = SGNHTS(5f-5, 1f0; xi = 1f0^2, μ = 10f0)  # Medium step size for 5 assets
ch = mcmc(bnn, 15, 40_000, sampler)
ch = ch[:, end-15_000+1:end]  # Keep 15,000 samples

println("MCMC sampling completed")
println("Chain dimensions: $(size(ch))")

# Analysis
yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])

println("Maximum R-hat: $(round(r_hat, digits=4))")
println("5-Asset Analysis Complete!")
println("="^50)

###############################################################################
#  DCC–GARCH Bayesian neural-network (BayesFlux.jl) - 10 Assets
###############################################################################
using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames

Random.seed!(1212)

# [Same helper functions as above - DCC simulator, helpers, likelihood]
# ... (copying the same functions to save space)

# ───────────────────────── 4.  Load and preprocess real data ────────────────────────
# Load data
etf_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"
df = CSV.read(etf_rf, DataFrame)

etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", 
             "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", 
             "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", 
             "28278", "28279", "28280", "31372", "31466"]

# Select first 10 assets (10 ETFs)
selected_assets = etf_names[1:10]  # First 10 ETFs
N = 10

# Data preprocessing with mean imputation
function preprocess_data_subset(df, selected_assets)
    for col in selected_assets
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    returns = Matrix{Float64}(df[!, selected_assets])
    μ = mean(returns, dims=1)
    return returns, vec(μ)
end

# Preprocess the data
returns, μ_returns = preprocess_data_subset(df, selected_assets)
y = Float32.(returns)
n, N = size(y)

println("="^50)
println("10-ASSET DCC-GARCH BNN ANALYSIS")
println("="^50)
println("Selected assets: $(selected_assets)")
println("Data dimensions: $n observations, $N assets")
println("Average returns: $(round.(μ_returns, digits=6))")

# Define training set
train_size = Int(floor(0.8 * n))
train_idx = 1:train_size
window = 22

println("Training set: $(length(train_idx)) observations")

# Prepare training data
Xtr, Ytr = prepare_time_series_data(y[train_idx,:], window)
println("Training data prepared: X shape $(size(Xtr)), Y shape $(size(Ytr))")

# ─────────────────────── 5.  build & train BNN ──────────────────────────────
# Define neural network architecture (larger for 10 assets)
net = Chain(
    Dense(N*window, 96, relu),    # Input layer (10*22=220 inputs)
    Dense(96, 48, relu),          # Hidden layer 1
    Dense(48, 24, relu),          # Hidden layer 2
    Dense(24, 2N)                 # Output layer (20 outputs: 10 means + 10 log-variances)
)

println("Neural network architecture:")
println("Input layer: $(N*window) → 96")
println("Hidden layers: 96 → 48 → 24") 
println("Output layer: 24 → $(2N)")

# Setup BNN components
nc    = destruct(net)
like  = DCCGarchNormal(nc, Normal(0, 0.12), N)
prior = GaussianPrior(nc, 0.12f0)
init  = InitialiseAllSame(Normal(0f0, 0.12f0), like, prior)
bnn   = BNN(Xtr, Ytr, like, prior, init)

println("BNN components initialized for 10 assets")

# Find mode of posterior for initialization
println("Finding posterior mode...")
θmap = find_mode(bnn, 100, 1200, FluxModeFinder(bnn, Flux.ADAM(0.003)))
println("Posterior mode found")

# ─────────────────────── 6.  posterior sampling (SGNHTS) ─────────────────────
println("Starting MCMC sampling...")
sampler = SGNHTS(2f-5, 1f0; xi = 1f0^2, μ = 10f0)  # Smaller step size for 10 assets
ch = mcmc(bnn, 15, 35_000, sampler)
ch = ch[:, end-12_000+1:end]  # Keep 12,000 samples

println("MCMC sampling completed")
println("Chain dimensions: $(size(ch))")

# Full analysis with posterior predictive checks
function my_posterior_predict(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    pred_samples = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        θnet = draws[1:end-2, i]
        θlike = draws[end-1:end, i]
        a, b = transform_ab(θlike...)
        
        net = bnn.like.nc(θnet)
        outs = map(t -> net(view(x, :, t)), 1:n_obs)
        μ = hcat(map(o -> o[1:N], outs)...)
        logσ2 = hcat(map(o -> o[N+1:2N], outs)...)
        σ = exp.(logσ2 ./ 2)
        
        z = (y .- μ) ./ σ
        Q̄ = (z * z') / n_obs
        d = 1 ./ sqrt.(diag(Q̄))
        Q̄ = Symmetric(diagm(d) * Q̄ * diagm(d))
        Q = copy(Q̄)
        
        samples = similar(μ)
        for t = 1:n_obs
            if t > 1
                zt = view(z, :, t-1)
                Q = (1-a-b) .* Q̄ .+ a .* (zt*zt') .+ b .* Q
            end
            
            d = 1 ./ sqrt.(max.(diag(Q), 1e-8))
            R = Symmetric(diagm(d) * Q * diagm(d))
            D = Diagonal(view(σ, :, t))
            H = D * R * D
            
            samples[:, t] = μ[:, t] + cholesky(Symmetric(H + 1e-6*I)).L * randn(N)
        end
        
        pred_samples[:, i] = vec(samples)
    end
    return pred_samples
end

# Generate predictions and assess calibration
yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])

posterior_yhat = my_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95

function get_observed_quantiles(y_true, y_samples, quantiles)
    N = size(y_true, 2)
    n_quantiles = length(quantiles)
    observed_quantiles = zeros(n_quantiles)
    
    y_true_vector = vec(y_true)
    n_time_points = div(size(y_samples, 1), N)
    total_points = N * n_time_points
    y_samples_reshaped = reshape(y_samples, total_points, :)
    
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

o_q = get_observed_quantiles(Ytr', posterior_yhat, t_q)

# Create Q-Q plot
p1 = plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
     xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(p1, x->x, t_q, label = "Perfect Calibration", linestyle=:dash) 
plot!(p1, title = "Q-Q Plot: 10 Assets DCC-GARCH BNN", size = (800, 600))

display(p1)

println("Maximum R-hat: $(round(r_hat, digits=4))")
println("Mean absolute deviation from perfect calibration: $(round(mean(abs.(o_q .- t_q)), digits=4))")
println("10-Asset Analysis Complete!")




###############################################################################
#  DCC–GARCH Bayesian neural-network (BayesFlux.jl) - Real ETF Data (30 Assets)
###############################################################################
using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames

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

# ───────────────────────── 4.  Load and preprocess real data ────────────────────────
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
    
    # Also handle missing values in risk-free rate
    if any(ismissing, df[!, "rf"])
        df[!, "rf"] = coalesce.(df[!, "rf"], mean(skipmissing(df[!, "rf"])))
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)  # 30 assets: 29 ETFs + 1 risk-free
    
    # Calculate mean returns for each asset
    μ = mean(returns, dims=1)
    
    return etf_returns, rf_returns, returns, vec(μ)
end

# Preprocess the data
etf_returns, rf_returns, returns, μ_returns = preprocess_data(df, etf_names)

# Use ALL 30 assets (29 ETFs + 1 risk-free rate) for the analysis
y = Float32.(returns)  # Convert to Float32 for neural network
n, N = size(y)  # n = number of observations, N = 30 assets

println("Data dimensions: $n observations, $N assets (29 ETFs + 1 risk-free)")
println("Sample period: $(n) days")
println("Average return across all assets: $(round(mean(μ_returns), digits=6))")
println("Average ETF return: $(round(mean(μ_returns[1:29]), digits=6))")
println("Average risk-free return: $(round(μ_returns[30], digits=6))")

# Define training set (use 80% for training)
train_size = Int(floor(0.8 * n))
train_idx = 1:train_size
window = 22  # Use 22 lags (approximately 1 month of trading days)

println("Training set: $(length(train_idx)) observations")
println("Test set: $(n - train_size) observations")

# Prepare training data
Xtr, Ytr = prepare_time_series_data(y[train_idx,:], window)
println("Training data prepared: X shape $(size(Xtr)), Y shape $(size(Ytr))")

# ─────────────────────── 5.  build & train BNN ──────────────────────────────
# Define neural network architecture (adjusted for 30 assets)
net = Chain(
    Dense(N*window, 128, relu),   # Input layer (30*22=660 inputs) - increased capacity
    Dense(128, 64, relu),         # Hidden layer 1
    Dense(64, 32, relu),          # Hidden layer 2  
    Dense(32, 2N)                 # Output layer (60 outputs: mean and log variance for each asset)
)

println("Neural network architecture:")
println("Input layer: $(N*window) → 128")
println("Hidden layers: 128 → 64 → 32") 
println("Output layer: 32 → $(2N)")

# Setup BNN components
nc    = destruct(net)  # Network constructor
like  = DCCGarchNormal(nc, Normal(0, 0.1), N)  # Custom likelihood for 30 assets
prior = GaussianPrior(nc, 0.1f0)  # Prior distribution (smaller variance for stability)
init  = InitialiseAllSame(Normal(0f0, 0.1f0), like, prior)  # Parameter initialization
bnn   = BNN(Xtr, Ytr, like, prior, init)  # BNN model

println("BNN components initialized for 30 assets")

# Find mode of posterior for initialization
println("Finding posterior mode...")
θmap = find_mode(bnn, 100, 1000, FluxModeFinder(bnn, Flux.ADAM()))
println("Posterior mode found")

# ─────────────────────── 6.  posterior sampling (SGNHTS) ─────────────────────
println("Starting MCMC sampling...")
# Use smaller step size for stability with 30-dimensional real data
sampler = SGNHTS(5f-6, 1f0; xi = 1f0^2, μ = 10f0)  # Even smaller step size for 30 assets
ch = mcmc(bnn, 20, 25_000, sampler)  # Reduced samples for computational efficiency
# Keep only the last 8,000 samples (discard 17,000 as burn-in)
ch = ch[:, end-8_000+1:end]

println("MCMC sampling completed")
println("Chain dimensions: $(size(ch))")

# Convert to MCMCChains format
chain = Chains(ch')

# ─────────────────────── 7. Naive prediction ─────────────────────────────────
function naive_prediction(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N  # Number of time series (30)
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

println("Generating naive predictions...")
# Generate naive predictions from the BNN using the MCMC samples
yhats = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats')

# Check convergence using R-hat statistic
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
println("Maximum R-hat: ", r_hat)

# ─────────────────────── 8. Posterior predictive checks ─────────────────────
function my_posterior_predict(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N  # Number of time series (30)
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

println("Generating posterior predictive samples...")
posterior_yhat = my_posterior_predict(bnn, ch)

# Define quantiles for evaluation
t_q = 0.05:0.05:0.95

# Get observed quantiles from posterior predictions
function get_observed_quantiles(y_true, y_samples, quantiles)
    N = size(y_true, 2)  # Number of time series (30)
    n_quantiles = length(quantiles)
    observed_quantiles = zeros(n_quantiles)
    
    # Reshape true values if needed
    if ndims(y_true) == 2 && size(y_true, 2) == N
        y_true_vector = vec(y_true)  # Flatten the matrix
    else
        y_true_vector = y_true  # Already in the right shape
    end
    
    # Get number of time points from the posterior samples
    n_time_points = div(size(y_samples, 1), N)
    total_points = N * n_time_points
    
    # Reshape samples for quantile calculation
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

println("Computing observed quantiles...")
o_q = get_observed_quantiles(Ytr', posterior_yhat, t_q)

# Create quantile-quantile plot to assess model calibration
println("Creating Q-Q plot...")
p1 = plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
     xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(p1, x->x, t_q, label = "Perfect Calibration", linestyle=:dash) 
plot!(p1, title = "Q-Q Plot: 30 Assets DCC-GARCH BNN\n(29 ETFs + Risk-Free Rate)", size = (800, 600))

display(p1)

# ─────────────────────── 9. Additional Analysis ─────────────────────────────
println("ANALYSIS SUMMARY - 30 ASSETS (29 ETFs + 1 RISK-FREE)")
println("Data: $(N) assets total")
println("  • $(N-1) ETFs")
println("  • 1 risk-free asset")
println("Observations: $(n) total")
println("Training period: $(train_size) observations")
println("Window size: $(window) lags")
println("Network parameters: $(length(θmap)-2) + 2 DCC parameters")
println("MCMC samples: $(size(ch, 2)) (after burn-in)")
println("Maximum R-hat: $(round(r_hat, digits=4))")
println("Mean absolute deviation from perfect calibration: $(round(mean(abs.(o_q .- t_q)), digits=4))")

# Analyze individual asset types
println("\nAsset-specific statistics:")
println("ETF returns - Mean: $(round(mean(μ_returns[1:29]), digits=6)), Std: $(round(std(μ_returns[1:29]), digits=6))")
println("Risk-free return: $(round(μ_returns[30], digits=6))")

# Save results
println("\nSaving results...")
# You can save the chain, predictions, etc. here if needed
# CSV.write("30_asset_predictions.csv", DataFrame(yhats', :auto))
# CSV.write("30_asset_mcmc_chain.csv", DataFrame(ch', :auto))
