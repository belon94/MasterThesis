###############################################################################
#  DCC–GARCH Bayesian Sequential LSTM - Multi-Asset Comparison (2, 5, 10)    #
###############################################################################
using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames

Random.seed!(1212)

# ───────────────────────── Helper Functions (same as before) ──────────────────────────
sigmoid(x) = 1/(1+exp(-x))
transform_ab(a,b) = let a_=sigmoid(a); b_=sigmoid(b)*(1-a_); (a_,b_) end
nearest_pd(A) = (A + A')/2 + 1e-4I

function prepare_sequential_data(data)
    X = data[1:end-1, :]'
    Y = data[2:end, :]'
    return Float32.(X), Float32.(Y)
end

# ─────────────────── Sequential LSTM DCC-GARCH likelihood (same as before) ─────────────────
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
    
    lstm_layer = net[1]
    dense_layer = net[2]
    
    hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
    h = zeros(T, hidden_size)
    c = zeros(T, hidden_size)
    
    outs = map(1:Tsteps) do t
        input_t = view(x, :, t)
        gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
        
        i_gate = sigmoid.(gates[1:hidden_size])
        f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
        g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
        o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
        
        c = f_gate .* c .+ i_gate .* g_gate
        h = o_gate .* tanh.(c)
        
        output = dense_layer(h)
        return output
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

# ───────────────────────── Data Loading and Preprocessing ────────────────────────────
etf_rf = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"
df = CSV.read(etf_rf, DataFrame)
etf_names = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", 
             "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", 
             "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", 
             "28278", "28279", "28280", "31372", "31466"]

function preprocess_data(df, etf_names)
    for col in etf_names
        if any(ismissing, df[!, col])
            df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col])))
        end
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)
    μ = mean(returns, dims=1)
    
    return etf_returns, rf_returns, returns, vec(μ)
end

etf_returns, rf_returns, returns, μ_returns = preprocess_data(df, etf_names)

# ───────────────────────── Multi-Asset Model Function ────────────────────────────
function run_sequential_lstm_model(n_assets::Int, asset_selection="first")
    println("RUNNING SEQUENTIAL LSTM DCC-GARCH WITH $n_assets ASSETS")
    
    # Select subset of assets
    if asset_selection == "first"
        selected_etfs = etf_names[1:n_assets-1]  # n_assets-1 ETFs + 1 risk-free
        y = Float32.(hcat(etf_returns[:, 1:n_assets-1], rf_returns))
    elseif asset_selection == "diversified"
        # Select more diversified assets (every few assets)
        step = max(1, length(etf_names) ÷ (n_assets-1))
        selected_indices = 1:step:length(etf_names)
        selected_indices = selected_indices[1:n_assets-1]
        selected_etfs = etf_names[selected_indices]
        y = Float32.(hcat(etf_returns[:, selected_indices], rf_returns))
    end
    
    n, N = size(y)
    
    println("Selected ETFs: ", selected_etfs)
    println("Number of observations: ", n)
    println("Number of assets: ", N)
    
    # Train/test split
    train_size = Int(floor(0.8 * n))
    train_idx = 1:train_size
    test_idx = (train_size+1):n
    
    Xtr, Ytr = prepare_sequential_data(y[train_idx,:])
    Xte, Yte = prepare_sequential_data(y[test_idx,:])
    
    println("Training data shape: X=", size(Xtr), ", Y=", size(Ytr))
    
    # Model architecture - scale hidden size with number of assets
    hidden_size = max(16, min(64, 8 * N))  # Adaptive hidden size
    net = Chain(
        LSTM(N => hidden_size),
        Dense(hidden_size => 2N)
    )
    
    println("LSTM hidden size: ", hidden_size)
    println("Total output dimension: ", 2N)
    
    # Setup BNN
    nc = destruct(net)
    like = DCCGarchLSTMSequential(nc, Normal(0, 0.1), N)
    prior = GaussianPrior(nc, 0.1f0)
    init = InitialiseAllSame(Normal(0f0, 0.1f0), like, prior)
    bnn = BNN(Xtr, Ytr, like, prior, init)
    
    println("Total parameters: ", length(nc.θ))
    
    # Find MAP
    println("Finding MAP estimate...")
    θmap = find_mode(bnn, 50, 500, FluxModeFinder(bnn, Flux.ADAM()))
    
    # MCMC sampling - adjust based on problem size
    println("Starting MCMC sampling...")
    if N <= 2
        sampler = SGNHTS(1f-4, 1f0; xi = 1f0^2, μ = 10f0)
        n_samples = 40_000
        n_chains = 10
    elseif N <= 5
        sampler = SGNHTS(5f-5, 1f0; xi = 1f0^2, μ = 8f0)
        n_samples = 30_000
        n_chains = 15
    else  # N = 10
        sampler = SGNHTS(2f-5, 1f0; xi = 1f0^2, μ = 5f0)
        n_samples = 25_000
        n_chains = 20
    end
    
    ch = mcmc(bnn, n_chains, n_samples, sampler)
    burn_in = n_samples ÷ 2
    ch = ch[:, end-burn_in+1:end]
    
    # Prediction functions
    function naive_prediction_sequential_lstm(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
        N = bnn.like.N
        n_samples = size(draws, 2)
        n_obs = size(x, 2)
        
        yhats = Array{T, 2}(undef, N * n_obs, n_samples)
        
        Threads.@threads for i = 1:n_samples
            net = bnn.like.nc(draws[:, i])
            lstm_layer = net[1]
            dense_layer = net[2]
            
            hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
            h = zeros(T, hidden_size)
            c = zeros(T, hidden_size)
            
            predictions = map(1:n_obs) do t
                input_t = view(x, :, t)
                gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
                
                i_gate = sigmoid.(gates[1:hidden_size])
                f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
                g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
                o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
                
                c = f_gate .* c .+ i_gate .* g_gate
                h = o_gate .* tanh.(c)
                
                final_output = dense_layer(h)
                return final_output[1:N]
            end
            
            yhats[:, i] = vcat(predictions...)
        end
        
        return yhats
    end
    
    function posterior_predict_sequential_lstm(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
        N = bnn.like.N
        n_samples = min(size(draws, 2), 500)  # Limit for efficiency
        n_obs = size(x, 2)
        
        pred_samples = Array{T, 2}(undef, N * n_obs, n_samples)
        
        Threads.@threads for i = 1:n_samples
            θnet = draws[1:end-2, i]
            θlike = draws[end-1:end, i]
            a, b = transform_ab(θlike...)
            
            net = bnn.like.nc(θnet)
            lstm_layer = net[1]
            dense_layer = net[2]
            
            hidden_size = size(lstm_layer.cell.Wi, 1) ÷ 4
            h = zeros(T, hidden_size)
            c = zeros(T, hidden_size)
            
            results = map(1:n_obs) do t
                input_t = view(x, :, t)
                gates = lstm_layer.cell.Wi * input_t + lstm_layer.cell.Wh * h .+ lstm_layer.cell.b
                
                i_gate = sigmoid.(gates[1:hidden_size])
                f_gate = sigmoid.(gates[hidden_size+1:2*hidden_size])
                g_gate = tanh.(gates[2*hidden_size+1:3*hidden_size])
                o_gate = sigmoid.(gates[3*hidden_size+1:4*hidden_size])
                
                c = f_gate .* c .+ i_gate .* g_gate
                h = o_gate .* tanh.(c)
                
                final_output = dense_layer(h)
                μ_t = final_output[1:N]
                σ_t = exp.(final_output[N+1:2N] ./ 2)
                return (μ_t, σ_t)
            end
            
            μ_all = hcat([r[1] for r in results]...)
            σ_all = hcat([r[2] for r in results]...)
            
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
                H = D * R * D + 1e-6*I
                
                samples[:, t] = μ_all[:, t] + cholesky(Symmetric(H)).L * randn(N)
            end
            
            pred_samples[:, i] = vec(samples)
        end
        
        return pred_samples
    end
    
    # Generate predictions
    println("Generating predictions...")
    yhats = naive_prediction_sequential_lstm(bnn, ch)
    chain_yhat = Chains(yhats')
    
    r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
    println("Maximum R-hat: ", round(r_hat, digits=3))
    
    # Posterior predictive checks
    posterior_yhat = posterior_predict_sequential_lstm(bnn, ch)
    
    # Quantile-quantile evaluation
    t_q = 0.05:0.05:0.95
    
    function get_observed_quantiles(y_true, y_samples, quantiles, N)
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
    
    o_q = get_observed_quantiles(y[train_idx,:], posterior_yhat, t_q, N)
    
    # Create plots
    p1 = plot(t_q, o_q, label = "$N Assets - Sequential LSTM", legend=:topleft,
         xlab = "Target Quantile", ylab = "Observed Quantile", linewidth=2)
    plot!(p1, x->x, t_q, label = "45-degree line", linestyle=:dash, color=:black) 
    plot!(p1, title = "Q-Q Plot: $N Assets Sequential LSTM DCC-GARCH", size = (800, 600))
    
    # Sample paths for first asset
    if r_hat < 1.3
        n_plot_samples = min(30, size(posterior_yhat, 2))
        sample_indices = rand(1:size(posterior_yhat, 2), n_plot_samples)
        
        p2 = plot(title="Asset 1 - $N Assets Model", size=(800, 400))
        actual_data = y[train_idx[2:end], 1]
        plot!(p2, actual_data, label="Actual", color=:black, linewidth=2)
        
        for i in 1:min(10, n_plot_samples)
            y_sample = reshape(posterior_yhat[:, sample_indices[i]], N, :)
            plot!(p2, vec(y_sample[1, :]), alpha=0.2, color=:blue, 
                  label=i==1 ? "Predicted Samples" : "")
        end
    end
    
    # Results summary
    dcc_params = transform_ab(mean(ch[end-1:end, :], dims=2)...)
    
    results = Dict(
        "n_assets" => N,
        "selected_etfs" => selected_etfs,
        "r_hat" => r_hat,
        "dcc_params" => dcc_params,
        "n_parameters" => length(nc.θ),
        "hidden_size" => hidden_size,
        "qq_plot" => p1,
        "observed_quantiles" => o_q,
        "target_quantiles" => t_q
    )
    
    println("\n=== $N Assets Model Summary ===")
    println("Selected ETFs: ", selected_etfs)
    println("LSTM hidden size: ", hidden_size)
    println("Total parameters: ", length(nc.θ))
    println("DCC parameters (a, b): ", round.(dcc_params, digits=4))
    println("Maximum R-hat: ", round(r_hat, digits=3))
    println("Convergence: ", r_hat < 1.1 ? "✓ Good" : r_hat < 1.2 ? "~ Acceptable" : "✗ Poor")
    
    return results, p1, (r_hat < 1.3 ? p2 : nothing)
end

# ───────────────────────── Run All Models ────────────────────────────
results_2, plot_2, path_2 = run_sequential_lstm_model(2, "first")
results_5, plot_5, path_5 = run_sequential_lstm_model(5, "first") 
results_10, plot_10, path_10 = run_sequential_lstm_model(10, "first")

# ───────────────────────── Comparative Analysis ────────────────────────────
println("COMPARATIVE ANALYSIS ACROSS DIFFERENT ASSET DIMENSIONS")

# Create comparison plots
all_plots = plot(plot_2, plot_5, plot_10, layout=(1,3), size=(1500, 500))
plot!(all_plots, plot_title="Sequential LSTM DCC-GARCH: Multi-Asset Comparison")
display(all_plots)

# Summary table
println("┌─────────┬────────────┬─────────────┬──────────────┬─────────────┐")
println("│ Assets  │ Parameters │ Hidden Size │ DCC (a, b)   │ R-hat       │")
println("├─────────┼────────────┼─────────────┼──────────────┼─────────────┤")

for (n_assets, results) in [(2, results_2), (5, results_5), (10, results_10)]
    a, b = results["dcc_params"]
    println("│ $(lpad(n_assets, 7)) │ $(lpad(results["n_parameters"], 10)) │ $(lpad(results["hidden_size"], 11)) │ $(lpad(round(a, digits=3), 4)), $(lpad(round(b, digits=3), 4)) │ $(lpad(round(results["r_hat"], digits=3), 11)) │")
end
println("└─────────┴────────────┴─────────────┴──────────────┴─────────────┘")

# QQ plot comparison
println("\n=== QUANTILE-QUANTILE PERFORMANCE ===")
for (n_assets, results) in [(2, results_2), (5, results_5), (10, results_10)]
    o_q = results["observed_quantiles"]
    t_q = results["target_quantiles"]
    
    # Calculate mean absolute deviation from diagonal
    mad = mean(abs.(o_q .- t_q))
    println("$n_assets Assets - Mean Absolute Deviation: $(round(mad, digits=4))")
end

# Display individual sample path plots if available
if path_2 !== nothing
    display(path_2)
end
if path_5 !== nothing
    display(path_5)
end
if path_10 !== nothing
    display(path_10)
end




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
