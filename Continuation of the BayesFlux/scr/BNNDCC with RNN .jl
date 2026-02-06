###############################################################################
#  DCC–GARCH Bayesian Sequential RNN  (BayesFlux.jl) - MULTI-ASSET ANALYSIS  #
###############################################################################
using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames

Random.seed!(1212)

# ───────────────────────── 1.  DCC-GARCH simulator (keep for reference) ─────
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

# ───────────────────────── 3.  Data Loading and Preprocessing ───────────────
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
    
    # Handle missing values in risk-free rate
    if any(ismissing, df[!, "rf"])
        df[!, "rf"] = coalesce.(df[!, "rf"], mean(skipmissing(df[!, "rf"])))
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)  # All assets: ETFs + RF
    
    # Calculate mean returns for each asset
    μ = mean(returns, dims=1)
    
    return etf_returns, rf_returns, returns, vec(μ)
end

# Process the real data
etf_returns, rf_returns, returns_full, mean_returns_full = preprocess_data(df, etf_names)

# ─────────────────── 4.  Sequential RNN DCC-GARCH likelihood ─────────────────
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
    
    # Initialize RNN hidden state
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

# ───────────────────────── 5. Prediction Functions ─────────────────────────
function naive_prediction_sequential_rnn(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    
    yhats = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        net = bnn.like.nc(draws[:, i])
        rnn_layer = net[1]
        dense_layer = net[2]
        
        # Initialize RNN hidden state
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

# ───────────────────────── 6. Analysis Function for Multiple Asset Sizes ─────────────────
function run_rnn_analysis(num_assets::Int, asset_selection="first")
    println("\n" * "="^80)
    println("SEQUENTIAL RNN DCC-GARCH ANALYSIS: $num_assets ASSETS")
    println("="^80)
    
    # Select assets based on strategy
    if asset_selection == "first"
        # Take first N-1 ETFs + risk-free rate
        if num_assets == 2
            selected_etfs = [1]  # First ETF only
            selected_names = [etf_names[1]]
        else
            selected_etfs = 1:(num_assets-1)
            selected_names = etf_names[1:(num_assets-1)]
        end
    end
    
    # Create subset of data: selected ETFs + risk-free rate
    returns_subset = hcat(etf_returns[:, selected_etfs], rf_returns)
    asset_names_subset = [selected_names..., "RF_Rate"]
    
    # Use the subset
    y = Float32.(returns_subset)
    n, N = size(y)
    
    println("Number of observations: ", n)
    println("Number of assets: ", N)
    println("Assets: ", asset_names_subset)
    
    # Train/test split
    train_split = 0.8
    train_size = Int(floor(n * train_split))
    train_idx = 1:train_size
    test_idx = (train_size+1):n
    
    Xtr, Ytr = prepare_sequential_data(y[train_idx,:])
    Xtest, Ytest = prepare_sequential_data(y[test_idx,:])
    
    println("Training observations: ", size(Xtr, 2))
    println("Test observations: ", size(Xtest, 2))
    
    # Build network - scale architecture with number of assets
    hidden_size = max(8, min(N*4, 64))  # Scale hidden size reasonably
    net = Chain(
        RNN(N => hidden_size),
        Dense(hidden_size => 2N)
    )
    
    println("RNN hidden size: ", hidden_size)
    println("Network parameters: ", sum(length(p) for p in Flux.params(net)))
    
    # Setup BNN components
    nc = destruct(net)
    like = DCCGarchRNNSequential(nc, Normal(0, 0.5), N)
    prior = GaussianPrior(nc, 0.5f0)
    init = InitialiseAllSame(Normal(0f0, 0.5f0), like, prior)
    bnn = BNN(Xtr, Ytr, like, prior, init)
    
    # Find MAP estimate
    println("Finding MAP estimate...")
    θmap = find_mode(bnn, 50, 1000, FluxModeFinder(bnn, Flux.ADAM()))
    
    # MCMC sampling - adjust iterations based on complexity
    mcmc_iterations = num_assets <= 5 ? 20_000 : (num_assets <= 10 ? 15_000 : 12_000)
    keep_samples = num_assets <= 5 ? 8_000 : (num_assets <= 10 ? 6_000 : 4_000)
    
    println("Starting MCMC sampling ($mcmc_iterations iterations)...")
    sampler = SGNHTS(1f-5, 1f0; xi = 1f0^2, μ = 25f0)
    ch = mcmc(bnn, 10, mcmc_iterations, sampler)
    ch = ch[:, end-keep_samples+1:end]
    
    chain = Chains(ch')
    
    # Predictions
    println("Generating predictions...")
    yhats = naive_prediction_sequential_rnn(bnn, ch)
    chain_yhat = Chains(yhats')
    
    # Check convergence
    r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
    println("Maximum R-hat: ", r_hat)
    
    # Posterior predictive checks
    println("Generating posterior predictive samples...")
    posterior_yhat = posterior_predict_sequential_rnn(bnn, ch)
    
    # QQ plot
    t_q = 0.05:0.05:0.95
    
    function get_observed_quantiles(y_true, y_samples, quantiles)
        N = size(y_true, 2)
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
    
    o_q = get_observed_quantiles(y[train_idx[2:end],:], posterior_yhat, t_q)
    
    # Create QQ plot
    p1 = plot(t_q, o_q, label = "RNN Posterior Predictive ($N Assets)", legend=:topleft,
         xlab = "Target Quantile", ylab = "Observed Quantile")
    plot!(p1, x->x, t_q, label = "45-degree line", linestyle=:dash) 
    plot!(p1, title = "Sequential RNN DCC-GARCH QQ Plot - $N Assets", size = (800, 600))
    display(p1)
    
    # Detailed Performance Analysis
    println("\n--- PERFORMANCE ANALYSIS FOR $N ASSETS ---")
    
    # Get the actual training data
    actual_data = y[train_idx[2:end], :]
    n_train_obs = size(actual_data, 1)
    
    # Calculate mean actual returns
    mean_actual = mean(actual_data, dims=1)
    
    # Properly reshape posterior predictions
    mean_posterior = mean(posterior_yhat, dims=2)
    mean_predicted_reshaped = reshape(vec(mean_posterior), N, n_train_obs)
    mean_predicted = mean(mean_predicted_reshaped, dims=2)
    
    # Calculate metrics for all assets
    correlations = Float64[]
    maes = Float64[]
    rmses = Float64[]
    
    println("\nAsset\t\t\tMean_Actual\tMean_Pred\tCorrelation\tMAE\t\tRMSE")
    println("="^85)
    
    for i in 1:N
        actual_series = actual_data[:, i]
        predicted_series = mean_predicted_reshaped[i, :]
        
        # Metrics
        corr_val = cor(actual_series, predicted_series)
        mae = mean(abs.(actual_series .- predicted_series))
        rmse = sqrt(mean((actual_series .- predicted_series).^2))
        
        push!(correlations, corr_val)
        push!(maes, mae)
        push!(rmses, rmse)
        
        asset_name = i == N ? "RF_Rate" : (i <= length(selected_names) ? selected_names[i] : "Asset_$i")
        println("$(rpad(asset_name, 15))\t$(round(mean_actual[i], digits=6))\t$(round(mean_predicted[i], digits=6))\t$(round(corr_val, digits=4))\t\t$(round(mae, digits=6))\t$(round(rmse, digits=6))")
    end
    
    # Summary statistics
    println("\n--- SUMMARY STATISTICS ---")
    println("Mean Correlation: $(round(mean(correlations), digits=4))")
    println("Median Correlation: $(round(median(correlations), digits=4))")
    println("Min Correlation: $(round(minimum(correlations), digits=4))")
    println("Max Correlation: $(round(maximum(correlations), digits=4))")
    println("Mean MAE: $(round(mean(maes), digits=6))")
    println("Mean RMSE: $(round(mean(rmses), digits=6))")
    println("DCC parameters (a, b): $(transform_ab(mean(ch[end-1:end, :], dims=2)...))")
    println("Convergence (R-hat): $(round(r_hat, digits=4))")
    
    # Return results for comparison
    return Dict(
        "num_assets" => N,
        "asset_names" => asset_names_subset,
        "correlations" => correlations,
        "maes" => maes,
        "rmses" => rmses,
        "mean_correlation" => mean(correlations),
        "mean_mae" => mean(maes),
        "mean_rmse" => mean(rmses),
        "r_hat" => r_hat,
        "dcc_params" => transform_ab(mean(ch[end-1:end, :], dims=2)...),
        "qq_target" => t_q,
        "qq_observed" => o_q
    )
end

# ───────────────────────── 7. Run Analysis for Multiple Asset Sizes ─────────────────
println("STARTING COMPREHENSIVE MULTI-ASSET ANALYSIS")
println("="^80)

# Run analyses for different numbers of assets
asset_counts = [2, 5, 10]
results = Dict()

for num_assets in asset_counts
    try
        println("\nProcessing $num_assets assets...")
        results[num_assets] = run_rnn_analysis(num_assets)
        println("✓ Completed analysis for $num_assets assets")
    catch e
        println("✗ Error with $num_assets assets: $e")
        results[num_assets] = nothing
    end
end

# ───────────────────────── 8. Comparative Analysis ─────────────────
println("COMPARATIVE ANALYSIS ACROSS DIFFERENT PORTFOLIO SIZES")

# Create comparison table
println("\nAssets\tMean_Corr\tMean_MAE\tMean_RMSE\tR_hat\t\tDCC_a\t\tDCC_b")

for num_assets in asset_counts
    if results[num_assets] !== nothing
        r = results[num_assets]
        dcc_a, dcc_b = r["dcc_params"]
        println("$(r["num_assets"])\t$(round(r["mean_correlation"], digits=4))\t\t$(round(r["mean_mae"], digits=6))\t$(round(r["mean_rmse"], digits=6))\t$(round(r["r_hat"], digits=4))\t\t$(round(dcc_a, digits=4))\t\t$(round(dcc_b, digits=4))")
    end
end

# Visualization: Comparative QQ plots
valid_results = filter(x -> x.second !== nothing, results)
if length(valid_results) > 1
    p_comp = plot(title="QQ Plot Comparison Across Portfolio Sizes", 
                  xlabel="Target Quantile", ylabel="Observed Quantile",
                  legend=:topleft, size=(800, 600))
    
    colors = [:red, :blue, :green, :purple]
    for (i, (num_assets, result)) in enumerate(valid_results)
        if result !== nothing
            plot!(p_comp, result["qq_target"], result["qq_observed"], 
                  label="$(num_assets) Assets", color=colors[i], linewidth=2)
        end
    end
    
    plot!(p_comp, x->x, 0:0.1:1, label="45-degree line", linestyle=:dash, color=:black)
    display(p_comp)
end

# Performance trends
if length(valid_results) >= 2
    asset_sizes = [r[1] for r in valid_results if r[2] !== nothing]
    mean_corrs = [r[2]["mean_correlation"] for r in valid_results if r[2] !== nothing]
    mean_maes = [r[2]["mean_mae"] for r in valid_results if r[2] !== nothing]
    mean_rmses = [r[2]["mean_rmse"] for r in valid_results if r[2] !== nothing]
    
    # Correlation trend
    p_corr_trend = plot(asset_sizes, mean_corrs, marker=:circle, linewidth=2,
                       title="Mean Correlation vs Portfolio Size",
                       xlabel="Number of Assets", ylabel="Mean Correlation",
                       legend=false, size=(600, 400))
    display(p_corr_trend)
    
    # Error trends
    p_error_trend = plot(asset_sizes, mean_maes, marker=:circle, linewidth=2, label="MAE",
                        title="Prediction Errors vs Portfolio Size",
                        xlabel="Number of Assets", ylabel="Error",
                        size=(600, 400))
    plot!(p_error_trend, asset_sizes, mean_rmses, marker=:square, linewidth=2, label="RMSE")
    display(p_error_trend)
end

println("MULTI-ASSET SEQUENTIAL RNN DCC-GARCH ANALYSIS COMPLETE")

# Final summary
println("\nKEY FINDINGS:")
for (num_assets, result) in sort(collect(valid_results))
    if result !== nothing
        println("• $num_assets assets: Correlation=$(round(result["mean_correlation"], digits=3)), MAE=$(round(result["mean_mae"], digits=5)), Convergence=$(round(result["r_hat"], digits=3))")
    end
end
###############################################################################
#  DCC–GARCH Bayesian Sequential RNN  (BayesFlux.jl) - REAL DATA (30 ASSETS) #
###############################################################################
using Flux, BayesFlux
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics
using CSV, DataFrames

Random.seed!(1212)

# ───────────────────────── 1.  DCC-GARCH simulator (keep for reference) ─────
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

# ───────────────────────── 3.  Data Loading and Preprocessing ───────────────
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
    
    # Handle missing values in risk-free rate
    if any(ismissing, df[!, "rf"])
        df[!, "rf"] = coalesce.(df[!, "rf"], mean(skipmissing(df[!, "rf"])))
    end
    
    etf_returns = Matrix{Float64}(df[!, etf_names])
    rf_returns = Vector{Float64}(df[!, "rf"])
    returns = hcat(etf_returns, rf_returns)  # 30 assets: 29 ETFs + 1 RF
    
    # Calculate mean returns for each asset
    μ = mean(returns, dims=1)
    
    return etf_returns, rf_returns, returns, vec(μ)
end

# Process the real data
etf_returns, rf_returns, returns, mean_returns = preprocess_data(df, etf_names)

# Use ALL 30 assets (29 ETFs + 1 risk-free rate)
y = Float32.(returns)
n, N = size(y)

println("=== Real Portfolio Data Summary ===")
println("Number of observations: ", n)
println("Number of assets (29 ETFs + 1 RF): ", N)
println("ETF names (first 5): ", etf_names[1:5])
println("Data shape: ", size(y))
println("Mean returns (first 5 ETFs): ", round.(mean_returns[1:5], digits=6))
println("Mean risk-free return: ", round(mean_returns[end], digits=6))

# ─────────────────── 4.  Sequential RNN DCC-GARCH likelihood ─────────────────
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
    
    # Initialize RNN hidden state
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

# ───────────────────────── 5.  prepare training data ────────────────────────
# Use 80% for training, 20% for testing
train_split = 0.8
train_size = Int(floor(n * train_split))
train_idx = 1:train_size
test_idx = (train_size+1):n

Xtr, Ytr = prepare_sequential_data(y[train_idx,:])
Xtest, Ytest = prepare_sequential_data(y[test_idx,:])

println("Training observations: ", size(Xtr, 2))
println("Test observations: ", size(Xtest, 2))

# ─────────────────────── 6.  build & train Sequential RNN BNN ────────────────
# Define Sequential RNN architecture - scale for 30 assets
hidden_size = min(60, N*2)  # Scale hidden size with number of assets, but cap it
net = Chain(
    RNN(N => hidden_size),       # RNN layer: 30 inputs to hidden_size outputs
    Dense(hidden_size => 2N)     # Output layer: mean and log variance for each of 30 assets
)

println("RNN hidden size: ", hidden_size)
println("Network parameters: ", sum(length(p) for p in Flux.params(net)))

# Setup BNN components
nc = destruct(net)
like = DCCGarchRNNSequential(nc, Normal(0, 0.5), N)  # Sequential RNN likelihood
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0f0, 0.5f0), like, prior)
bnn = BNN(Xtr, Ytr, like, prior, init)

# Find mode of posterior for initialization
println("Finding MAP estimate...")
θmap = find_mode(bnn, 50, 1000, FluxModeFinder(bnn, Flux.ADAM()))

# ─────────────────────── 7.  posterior sampling (SGNHTS) ─────────────────────
println("Starting MCMC sampling...")
sampler = SGNHTS(1f-5, 1f0; xi = 1f0^2, μ = 25f0)  # Adjust for 30-dimensional data
ch = mcmc(bnn, 10, 25_000, sampler)  # Adjusted iterations for 30 assets
ch = ch[:, end-12_000+1:end]  # Keep last 12k samples

chain = Chains(ch')

# ─────────────────────── 8. Sequential RNN Prediction Functions ─────────────
function naive_prediction_sequential_rnn(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    N = bnn.like.N
    n_samples = size(draws, 2)
    n_obs = size(x, 2)
    
    yhats = Array{T, 2}(undef, N * n_obs, n_samples)
    
    Threads.@threads for i = 1:n_samples
        net = bnn.like.nc(draws[:, i])
        rnn_layer = net[1]
        dense_layer = net[2]
        
        # Initialize RNN hidden state
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
println("Generating predictions...")
yhats = naive_prediction_sequential_rnn(bnn, ch)
chain_yhat = Chains(yhats')

# Check convergence
r_hat = maximum(summarystats(chain_yhat)[:, :rhat])
println("Maximum R-hat: ", r_hat)

# ─────────────────────── 9. Sequential RNN Posterior Predictive Checks ─────────────
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

println("Generating posterior predictive samples...")
posterior_yhat = posterior_predict_sequential_rnn(bnn, ch)

# ─────────────────────── 10. Model Evaluation ─────────────────────────────────
t_q = 0.05:0.05:0.95

function get_observed_quantiles(y_true, y_samples, quantiles)
    N = size(y_true, 2)
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

o_q = get_observed_quantiles(y[train_idx[2:end],:], posterior_yhat, t_q)

# Create quantile-quantile plot
p1 = plot(t_q, o_q, label = "Sequential RNN Posterior Predictive (30 Assets)", legend=:topleft,
     xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(p1, x->x, t_q, label = "45-degree line", linestyle=:dash) 
plot!(p1, title = "Sequential RNN DCC-GARCH QQ Plot - 30 Assets (29 ETFs + RF)", size = (800, 600))

display(p1)

# ─────────────────────── 11. Additional Diagnostics (CORRECTED) ─────────────────────────
# ─────────────────────── 11. Additional Diagnostics (ALL 30 ASSETS) ─────────────────────────
println("\n=== Sequential RNN Model Summary (30 Assets: 29 ETFs + 1 RF) ===")
println("Data points (training): ", size(Xtr, 2))
println("Total assets: ", N, " (29 ETFs + 1 Risk-Free)")
println("RNN hidden size: ", hidden_size)
println("Total parameters: ", length(θmap))
println("DCC parameters (a, b): ", transform_ab(mean(ch[end-1:end, :], dims=2)...))
println("Maximum R-hat: ", r_hat)

# Calculate performance metrics for ALL 30 ASSETS
println("\n=== Performance Metrics for All 30 Assets ===")

# Get the actual training data (excluding first observation for sequential prediction)
actual_data = y[train_idx[2:end], :]  # Shape: (n_train_obs-1, N)
n_train_obs = size(actual_data, 1)

# Calculate mean actual returns
mean_actual = mean(actual_data, dims=1)  # Shape: (1, N)

# Properly reshape posterior predictions and calculate mean
# posterior_yhat shape: (N * n_train_obs, n_samples)
mean_posterior = mean(posterior_yhat, dims=2)  # Average over samples: (N * n_train_obs, 1)
mean_predicted_reshaped = reshape(vec(mean_posterior), N, n_train_obs)  # Shape: (N, n_train_obs)
mean_predicted = mean(mean_predicted_reshaped, dims=2)  # Shape: (N, 1)

# Display mean returns for all assets
println("\n--- Mean Returns Comparison ---")
println("Asset\t\tActual\t\tPredicted\t\tDifference")
for i in 1:N-1  # ETFs
    actual_mean = mean_actual[i]
    pred_mean = mean_predicted[i]
    diff = pred_mean - actual_mean
    asset_name = i <= length(etf_names) ? etf_names[i] : "ETF_$i"
    println("$asset_name\t$(round(actual_mean, digits=6))\t\t$(round(pred_mean, digits=6))\t\t$(round(diff, digits=6))")
end
# Risk-free rate
println("RF_Rate\t\t$(round(mean_actual[end], digits=6))\t\t$(round(mean_predicted[end], digits=6))\t\t$(round(mean_predicted[end] - mean_actual[end], digits=6))")

# Calculate correlations between actual and predicted time series for ALL 30 ASSETS
println("\n--- Time Series Correlations ---")
correlations = Float64[]
maes = Float64[]
rmses = Float64[]

println("Asset\t\tCorrelation\tMAE\t\tRMSE")

for i in 1:N
    actual_series = actual_data[:, i]  # Time series for asset i
    predicted_series = mean_predicted_reshaped[i, :]  # Predicted time series for asset i
    
    if length(actual_series) == length(predicted_series)
        # Correlation
        corr_val = cor(actual_series, predicted_series)
        push!(correlations, corr_val)
        
        # Mean Absolute Error
        mae = mean(abs.(actual_series .- predicted_series))
        push!(maes, mae)
        
        # Root Mean Square Error
        rmse = sqrt(mean((actual_series .- predicted_series).^2))
        push!(rmses, rmse)
        
        # Display results
        if i <= length(etf_names)
            asset_name = etf_names[i]
        elseif i == N
            asset_name = "RF_Rate"
        else
            asset_name = "Asset_$i"
        end
        
        println("$asset_name\t$(round(corr_val, digits=4))\t\t$(round(mae, digits=6))\t$(round(rmse, digits=6))")
        
    else
        println("Warning: Length mismatch for asset $i")
        push!(correlations, NaN)
        push!(maes, NaN)
        push!(rmses, NaN)
    end
end

# Summary statistics across all assets
println("\n--- Summary Statistics Across All 30 Assets ---")
valid_correlations = correlations[.!isnan.(correlations)]
valid_maes = maes[.!isnan.(maes)]
valid_rmses = rmses[.!isnan.(rmses)]

println("Correlation Statistics:")
println("  Mean: $(round(mean(valid_correlations), digits=4))")
println("  Median: $(round(median(valid_correlations), digits=4))")
println("  Min: $(round(minimum(valid_correlations), digits=4))")
println("  Max: $(round(maximum(valid_correlations), digits=4))")
println("  Std: $(round(std(valid_correlations), digits=4))")

println("\nMAE Statistics:")
println("  Mean: $(round(mean(valid_maes), digits=6))")
println("  Median: $(round(median(valid_maes), digits=6))")
println("  Min: $(round(minimum(valid_maes), digits=6))")
println("  Max: $(round(maximum(valid_maes), digits=6))")

println("\nRMSE Statistics:")
println("  Mean: $(round(mean(valid_rmses), digits=6))")
println("  Median: $(round(median(valid_rmses), digits=6))")
println("  Min: $(round(minimum(valid_rmses), digits=6))")
println("  Max: $(round(maximum(valid_rmses), digits=6))")

# Identify best and worst performing assets
println("\n--- Best and Worst Performing Assets ---")
best_corr_idx = argmax(valid_correlations)
worst_corr_idx = argmin(valid_correlations)

println("Highest Correlation:")
if best_corr_idx <= length(etf_names)
    println("  $(etf_names[best_corr_idx]): $(round(correlations[best_corr_idx], digits=4))")
else
    println("  RF_Rate: $(round(correlations[end], digits=4))")
end

println("Lowest Correlation:")
if worst_corr_idx <= length(etf_names)
    println("  $(etf_names[worst_corr_idx]): $(round(correlations[worst_corr_idx], digits=4))")
else
    println("  RF_Rate: $(round(correlations[end], digits=4))")
end


# Plot sample paths for selected assets (now showing more variety)
if r_hat < 1.2
    n_plot_samples = min(50, size(posterior_yhat, 2))
    sample_indices = rand(1:size(posterior_yhat, 2), n_plot_samples)
    
    # Plot assets with highest, median, and lowest correlations for variety
    selected_indices = [argmax(correlations), 
                       sortperm(correlations)[div(N,2)],  # median
                       argmin(correlations),
                       N]  # risk-free rate
    
    for (plot_num, asset_idx) in enumerate(selected_indices)
        if asset_idx <= length(etf_names)
            asset_name = etf_names[asset_idx]
            title_prefix = "ETF"
        elseif asset_idx == N
            asset_name = "RF_Rate"
            title_prefix = "Risk-Free Rate"
        else
            asset_name = "Asset_$(asset_idx)"
            title_prefix = "Asset"
        end
        
        p_asset = plot(title="$title_prefix $asset_name (Corr: $(round(correlations[asset_idx], digits=3)))", 
                      size=(800, 400))
        
        # Extract actual time series
        actual_series = actual_data[:, asset_idx]
        plot!(p_asset, actual_series, label="Actual Returns", color=:black, linewidth=2)
        
        # Plot sample predictions
        for i in 1:min(10, n_plot_samples)
            sample_predictions = reshape(posterior_yhat[:, sample_indices[i]], N, n_train_obs)
            predicted_series = sample_predictions[asset_idx, :]
            
            color = asset_idx == N ? :blue : :red  # Blue for RF, red for ETFs
            plot!(p_asset, predicted_series, alpha=0.15, color=color, 
                  label=i==1 ? "RNN Predicted Samples" : "")
        end
        
        display(p_asset)
    end
end

# Save results to a summary table (optional - for export)
results_df = DataFrame(
    Asset = [etf_names[1:min(length(etf_names), N-1)]..., "RF_Rate"],
    Mean_Actual = vec(mean_actual),
    Mean_Predicted = vec(mean_predicted),
    Correlation = correlations,
    MAE = maes,
    RMSE = rmses
)

println(results_df)

