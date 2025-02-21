##########################  Single Input  ############################

# This Julia script is designed for backtesting Bayesian neural network models on ETF return data. 
# It includes functions for data loading, preprocessing, model building, and Bayesian inference. 
# The script focuses on estimating and comparing value at risk (VaR) using different methodologies like GARCH, MAP, BNN, and BBB. 
# It assesses model performance through RMSE calculations and quantile comparison plots, 
# and consolidates the results into an Excel file for comprehensive analysis. 
# This is particularly useful for financial risk analysis and predictions using advanced statistical and machine learning methods.

#*** Ensure that all the necessary functions are executed before running the code in the 'Run Main Function' section.

include("BayesFlux.jl")
include("Thesis_Var_Backtest.jl")
using Random, Distributions, LinearAlgebra, Plots
using MCMCChains, Bijectors, Statistics, Flux, StatsBase
using .BayesFlux, ARCHModels 
using CSV,DataFrames,ExcelFiles,Plots,XLSX,Dates, DataStructures


# Load data
function load_data_portfolio(file_path)
    data = CSV.read(file_path, DataFrame)
    data[!, "Date"] = Date.(data[!, "Date"], "dd/mm/yyyy")
    # Replace missing value with the mean (note that there is only one missing value in 1477x18)
    data = replace_missing_with_mean(data)
    # Convert date to numerical representation
    #data[!, :Date] = Dates.value.(data[!, :Date])
    return data
end

# Preprocess data
# Scaling the input data 
function preprocess_data_portfolio(lag, data, train_index, val_index, test_index)
    # Convert DataFrame to Matrix
    matrix_data = Matrix(data)
    matrix_data = Matrix{Float32}(matrix_data[:,2:31])
    
    # Calculate row-wise mean of assets / first raw needs to be = -.009415702
    portfolio = mean(matrix_data, dims=2)
    portfolio
    #------
    #Scaling the input data 
    # Get the mean and standard deviation of each column
    scaled_train_index = train_index .- minimum(train_index) .+ 1
    scaled_val_index = val_index .- minimum(train_index) .+ 1
    scaled_test_index = test_index .- minimum(train_index) .+ 1
    scaled_full_index = (minimum(train_index):maximum(test_index)) .- minimum(train_index) .+ 1

    means = mean(portfolio[train_index])
    stddevs = std(portfolio[train_index])
    # Normalize the columns by subtracting the mean and dividing by the standard deviation
    portfolio = (portfolio .- means) ./ stddevs
    #------

    # Create portfolio matrix
    
    full_index = minimum(train_index):maximum(test_index)
    portfolio = portfolio[full_index]
    
    #first 5 observation will be missed
    data = DataFrame( Returns = portfolio[(lag + 1):end])

    data[!, :MAP_μ] = fill(Float64(1), size(data,1))
    data[!, :MAP_σ] = fill(Float64(1), size(data,1))
    
    data[!, :BNN_Mean_μ] = fill(Float64(2), size(data,1))
    data[!, :BNN_Mean_σ] = fill(Float64(2), size(data,1))

    data[!, :BNN_Median_μ] = fill(Float64(3), size(data,1))
    data[!, :BNN_Median_σ] = fill(Float64(3), size(data,1))

    data[!, :BBB_Mean_μ] = fill(Float64(4), size(data,1))
    data[!, :BBB_Mean_σ] = fill(Float64(4), size(data,1))

    data[!, :BBB_Median_μ] = fill(Float64(5), size(data,1))
    data[!, :BBB_Median_σ] = fill(Float64(5), size(data,1))

    data[!, :Garch_μ] = fill(Float64(6), size(data,1))
    data[!, :Garch_σ] = fill(Float64(6), size(data,1))

    df_train_index = (train_index .- minimum(train_index) .+ 1)[1:end-lag]
    df_validation_index = val_index .- minimum(val_index) .+ 1 .+ maximum(df_train_index)
    df_test_index = test_index .- minimum(test_index) .+ 1 .+ maximum(df_validation_index)

    y = portfolio 

    y_train, y_val, y_test, y_full = y[scaled_train_index], y[scaled_val_index], y[scaled_test_index], y[scaled_full_index]
    
    #arrange a train set for NN
    x_train = make_rnn_tensor(reshape(y_train, :, 1), lag + 1)
    y_train = vec(x_train[end, :, :])
    x_train = x_train[1:end-1, :, :]
        
    return x_train, y_train, y_val, y_test, y_full, data, df_train_index, df_validation_index, df_test_index
end

# Calculate Quantile
function calculate_quantile(degrees_f, quantiles)
    t_dist = TDist(degrees_f)
    quant = quantile(t_dist, quantiles)
    return quant
end

#get bnn
function get_bnn_data(net::Flux.Chain{T},x_train, y_train, degrees_f) where {T}
    bnn = get_bnn(net, x_train,(y_train),degrees_f)
    return bnn
end

# Get Bnn
function get_bnn(net::Flux.Chain{T},x, y,df) where {T}
    nc = destruct(net)
    like = GarchSeqToOneTDist(nc, Normal(0, 0.5),df)
    prior = GaussianPrior(nc, 1.5f0)
    init = InitialiseAllSame(Normal(0.0f0, 1.5f0), like, prior)
    bnn = BNN(x, y, like, prior, init)
    return bnn
end

# Calculate MAP Estimation
function calculate_map_estimation(bnn, x_train)
    opt = FluxModeFinder(bnn, Flux.RMSProp())
    θmap = find_mode(bnn, 10, 1000, opt)
    nethat = bnn.like.nc(θmap)
    parameters = [nethat(xx) for xx in eachslice(x_train; dims =1 )][end]
    μ_hat = parameters[1, :]
    log_σ = parameters[2, :]
    σ_hat = exp.(log_σ)
    return θmap, μ_hat, σ_hat
end

# Calculate Test Set MAP Estimation
function calculate_test_map_estimation(bnn, train_index,val_index, test_index, θmap, y_full)
    μ_hat_val, σ_hat_val , μ_hat_test, σ_hat_test = estimate_test_σ(bnn, train_index, val_index, test_index, θmap, y_full)
    return μ_hat_val, σ_hat_val, μ_hat_test, σ_hat_test
end

 # Function to estimate σs for BNN
 function get_bnn_predictions(bnn, θmap, y_full::Array{Float32, 1}, train_index,val_index, test_index, quant)

    sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)

    # Sampling 
    ch = mcmc(bnn, 10, 50_000, sampler, θstart = θmap)
    ch = ch[:, end-20_000+1:end]
    #chain = Chains(ch')

    # Training-set Bayesian Neural Network (BNN) mean/median VaRs estimation
    μhats, σhats = naive_train_bnn_σ_prediction_recurrent(bnn,ch)

    #VaRs_bnn = bnn_var_prediction(μhats,σhats,quant)

    # Test set estimation - optimized
    μhats_validation, σhats_validation , μhats_test, σhats_test = naive_test_bnn_σ_prediction_recurrent(bnn, y_full, train_index, val_index, test_index,ch)
    #VaRs_validation_bnn = bnn_var_prediction(μhats_validation,σhats_validation,quant)
    #VaRs_test_bnn = bnn_var_prediction(μhats_test,σhats_test,quant)

    return ch, μhats, σhats, μhats_validation, σhats_validation , μhats_test, σhats_test
end

# Function to estimate σs for BBB
function get_bbb_predictions(bnn, y_full::Array{Float32, 1}, train_index, val_index, test_index, quant)
    
    q, params, losses = bbb(bnn, 10, 2_000; mc_samples = 1, opt = Flux.RMSProp(), n_samples_convergence = 10)

    ch_bbb = rand(q, 20_000)
    
    # Training-set Bayesian Neural Network (BNN) mean/median VaRs estimation
    μhats_bbb, σhats_bbb = naive_train_bnn_σ_prediction_recurrent(bnn,ch_bbb)
    #VaRs_bbb = bnn_var_prediction(μhats_bbb, σhats_bbb, quant)

    # Test set estimation - computationally expensive
    μhats_validation_bbb, σhats_validation_bbb , μhats_test_bbb, σhats_test_bbb = naive_test_bnn_σ_prediction_recurrent(bnn, y_full, train_index, val_index, test_index,ch_bbb)
    #VaRs_validation_bbb = bnn_var_prediction(μhats_validation_bbb, σhats_validation_bbb,quant)
    #VaRs_test_bbb = bnn_var_prediction(μhats_test_bbb, σhats_test_bbb, quant)

    return ch_bbb, μhats_bbb, σhats_bbb, μhats_validation_bbb, σhats_validation_bbb, μhats_test_bbb, σhats_test_bbb
end

function save_qq_plot(train_index, val_index, test_index, y_full, y_train, y_val, y_test, bnn, ch, ch_bbb, new_dir)

    # Shift the train index
    train_index = train_index .- minimum(train_index) .+ 1
    shifted_index = train_index[:] .+ length(val_index) .+ length(test_index)

    # Make RNN tensor
    x_shifted = make_rnn_tensor(reshape(y_full[shifted_index], :, 1), lag + 1)
    x_shifted = x_shifted[1:end-1, :, :]

    # Sample from the posterior predictive distribution
    posterior_y_train = sample_posterior_predict(bnn, ch)    
    posterior_y_shift = sample_posterior_predict(bnn, ch; x = x_shifted)
    posterior_y_val = posterior_y_shift[end-length(test_index)-(length(val_index)-1):end-length(test_index),:]
    posterior_y_test = posterior_y_shift[end-(length(test_index)-1):end,:]

    posterior_bbb_y_train = sample_posterior_predict(bnn, ch_bbb)
    posterior_bbb_y_shift = sample_posterior_predict(bnn, ch_bbb; x = x_shifted)
    posterior_bbb_y_val = posterior_bbb_y_shift[end-length(test_index)-(length(val_index)-1):end-length(test_index),:]
    posterior_bbb_y_test = posterior_bbb_y_shift[end-(length(test_index)-1):end,:]

    # Plot quantile comparison and save
    p = plot_quantile_comparison(y_train, posterior_y_train)
    plotfile_posterior_y_train = joinpath(new_dir, "posterior_y_train.png")
    savefig(p, plotfile_posterior_y_train)

    p = plot_quantile_comparison(y_val, posterior_y_val)
    plotfile_posterior_y_val = joinpath(new_dir, "posterior_y_val.png")
    savefig(p, plotfile_posterior_y_val)

    p = plot_quantile_comparison(y_test, posterior_y_test)
    plotfile_posterior_y_test = joinpath(new_dir, "posterior_y_test.png")
    savefig(p, plotfile_posterior_y_test)

    p = plot_quantile_comparison(y_train, posterior_bbb_y_train)
    plotfile_posterior_bbb_y_train = joinpath(new_dir, "posterior_bbb_y_train.png")
    savefig(p, plotfile_posterior_bbb_y_train)

    p = plot_quantile_comparison(y_val, posterior_bbb_y_val)
    plotfile_posterior_bbb_y_val = joinpath(new_dir, "posterior_bbb_y_val.png")
    savefig(p, plotfile_posterior_bbb_y_val)

    p = plot_quantile_comparison(y_test, posterior_bbb_y_test)
    plotfile_posterior_bbb_y_test = joinpath(new_dir, "posterior_bbb_y_test.png")
    savefig(p, plotfile_posterior_bbb_y_test)
end

function plot_and_save(y_values, VaRs, title, filename, new_dir, label="5% VaR")
    plot(1:length(y_values), y_values, label="Returns", title=title)
    plot!(1:length(y_values), VaRs, label=label)
    plotfile = joinpath(new_dir, filename)
    savefig(plotfile)
end

function plot_and_save_combined(data_frame, raw_data, train_index, test_index, val_index, new_dir, plot_types, plot_titles, ylabels)
    plots = []
    df_full = (minimum(train_index):maximum(test_index))[(lag + 1):end]
    
    for i in 1:length(plot_types)
        p = plot(raw_data[df_full,:Date], data_frame[:, Symbol(plot_types[i])],
        title = plot_titles[i],
        xlab = "Date",
        ylab = ylabels[i],
        legend = false,
        linewidth = 2,
        linecolor = :blue)
    
        date1 = raw_data[minimum(val_index),:Date]
        date2 = raw_data[minimum(test_index),:Date]

        plot!(p, [date1, date1], [minimum(data_frame[:, Symbol(plot_types[i])]), maximum(data_frame[:, Symbol(plot_types[i])])], line=:dash, linewidth=2, linecolor=:red)
        plot!(p, [date2, date2], [minimum(data_frame[:, Symbol(plot_types[i])]), maximum(data_frame[:, Symbol(plot_types[i])])], line=:dash, linewidth=2, linecolor=:red)

        push!(plots, p)
        
        #plotfile = joinpath(new_dir, "$(plot_types[i])_over_Time")
        #savefig(p, plotfile)
    end

    combined_plot = plot(plots..., layout = (length(plot_types), 1), size = (600, 300 * length(plot_types)))
    savefig(combined_plot, joinpath(new_dir, "Combined_plot"))
end

####### -----------------------------------------------------------------------------------------#########
#######                                       Helper Functions                                   #########
####### -----------------------------------------------------------------------------------------#########


function replace_missing_with_mean(data)
    for col in names(data)
        if eltype(data[!, col]) <: Union{Real, Missing}  # Only process numerical columns
            #println(col)
            col_mean = mean(skipmissing(data[!, col]))  # Compute mean, ignoring missing
            data[!, col] = coalesce.(data[!, col], col_mean)  # Replace missing values with the mean
        end
    end
    return data
end

# Function to create BNN Var predictions for mean and median 
function bnn_var_prediction(μhats::Array{Float32,2}, σhats::Array{Float32,2}, quant) where {T}
    VaRs = μhats .+ (σhats .* reshape(quant, 1, 1, :)) 
    
    # Compute the final matrices
    final_matrix_mean = dropdims(mapslices(mean, VaRs, dims=2), dims=2)
    final_matrix_median = dropdims(mapslices(median, VaRs, dims=2), dims=2)

    # Reshape the matrices to add a third dimension
    final_matrix_mean = reshape(final_matrix_mean, size(final_matrix_mean)..., 1)
    final_matrix_median = reshape(final_matrix_median, size(final_matrix_median)..., 1)

    # Combine both results
    final_matrix = cat(final_matrix_mean, final_matrix_median, dims=3)
    return Float32.(final_matrix)
end    

function bnn_var_prediction(μhats, σhats, quant) where {T}
    result = hcat([σhats .* q for q in quant]...)
    return Float32.(μhats .+ result)
end

# Function to estimate σ for MAP validation and test set estimation for multi input model
function estimate_test_σ(bnn, train_index, val_index, test_index, θmap, y_full::Array{Float32, 1})
    train_index = train_index .- minimum(train_index) .+ 1
    nethat = bnn.like.nc(θmap)
    shifted_index = train_index[:] .+ length(val_index) .+ length(test_index)
    x_shifted = make_rnn_tensor(reshape(y_full[shifted_index], :, 1), lag + 1)
    x_shifted = x_shifted[1:end-1, :, :]
    parameters_whole = [nethat(xx) for xx in eachslice(x_shifted; dims =1 )][end]
    μ_whole = parameters_whole[1, :]
    log_σ_whole = parameters_whole[2, :]
    σ_hat_test = exp.(log_σ_whole)
    return  μ_whole[end-length(test_index)-(length(val_index)-1):end-length(test_index)], σ_hat_test[end-length(test_index)-(length(val_index)-1):end-length(test_index)],μ_whole[end-(length(test_index)-1):end], σ_hat_test[end-(length(test_index)-1):end]
end


# Function to estimate BNN train set σ estimation
function naive_train_bnn_σ_prediction_recurrent(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    μhats = Array{T, 2}(undef, length(y), size(draws, 2))
    log_σhats = Array{T, 2}(undef, length(y), size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])   
        parameters = [net(xx) for xx in eachslice(x; dims =1 )][end]
        μhats[:,i] = parameters[1, :]
        log_σhats[:,i] = parameters[2, :]
    end
    σhats = exp.(log_σhats)
    return μhats, σhats
end

# Function to estimate σ for BNN/BBB validation and test set estimation for multi input model
function naive_test_bnn_σ_prediction_recurrent(bnn,y_full::Array{Float32, 1},train_index, val_index, test_index, draws::Array{T, 2}) where {T}
    μ_whole = Array{T, 2}(undef, length(train_index)-lag, size(draws, 2))
    log_σ_whole = Array{T, 2}(undef, length(train_index)-lag, size(draws, 2))
    train_index = train_index .- minimum(train_index) .+ 1
    
    shifted_index = train_index[:] .+ length(val_index) .+ length(test_index)
    x_shifted = make_rnn_tensor(reshape(y_full[shifted_index], :, 1), lag + 1)
    x_shifted = x_shifted[1:end-1, :, :]
    Threads.@threads for j=1:size(draws, 2)
        nethat = bnn.like.nc(draws[:, j])
        parameters_whole = [nethat(xx) for xx in eachslice(x_shifted; dims =1 )][end]
        μ_whole[:,j] = parameters_whole[1, :]
        log_σ_whole[:,j] = parameters_whole[2, :]
        end
    σhats = exp.(log_σ_whole)
    return  μ_whole[end-length(test_index)-(length(val_index)-1):end-length(test_index),:], σhats[end-length(test_index)-(length(val_index)-1):end-length(test_index),:],μ_whole[end-(length(test_index)-1):end,:], σhats[end-(length(test_index)-1):end,:]
end

# Function to plot calibration for BNN/BBB
function plot_quantile_comparison(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    observed_q = get_observed_quantiles(y, posterior_yhat, target_q)
    plot(target_q, observed_q, label = "Observed", legend_position = :topleft, xlab = "Quantile of Posterior Draws", ylab = "Percent Observations below")
    plot!(x -> x, minimum(target_q), maximum(target_q), label = "Theoretical")
end

# Function to get observed quantiles for BNN/BBB
function get_observed_quantiles(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    qs = [quantile(yr, target_q) for yr in eachrow(posterior_yhat)]
    qs = reduce(hcat, qs)
    observed_q = mean(reshape(y, 1, :) .< qs; dims = 2)
    return observed_q
end

# Update DataFrame
function update_dataframe(df, df_train_index,df_validation_index, df_test_index, μ_hat, σ_hat, μ_hat_val, σ_hat_val, μ_hat_test, σ_hat_test,μhats, σhats, μhats_validation, σhats_validation , μhats_test, σhats_test,μhats_bbb, σhats_bbb, μhats_validation_bbb, σhats_validation_bbb, μhats_test_bbb, σhats_test_bbb,garch_μ, garch_σ)


    df[df_train_index, :"MAP_μ"] = convert(Vector{Float64}, μ_hat)
    df[df_validation_index, :"MAP_μ"] = convert(Vector{Float64}, μ_hat_val)
    df[df_test_index, :"MAP_μ"] = convert(Vector{Float64}, μ_hat_test)
    df[df_train_index, :"MAP_σ"] = convert(Vector{Float64}, σ_hat)
    df[df_validation_index, :"MAP_σ"] = convert(Vector{Float64}, σ_hat_val)
    df[df_test_index, :"MAP_σ"] = convert(Vector{Float64}, σ_hat_test)

    df[df_train_index, :"BNN_Mean_μ"] = convert(Vector{Float64}, calc_mean_median(μhats)[1])
    df[df_validation_index, :"BNN_Mean_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_validation)[1])
    df[df_test_index, :"BNN_Mean_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_test)[1])
    df[df_train_index, :"BNN_Mean_σ"] = convert(Vector{Float64}, calc_mean_median(σhats)[1])
    df[df_validation_index, :"BNN_Mean_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_validation)[1])
    df[df_test_index, :"BNN_Mean_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_test)[1])

    df[df_train_index, :"BNN_Median_μ"] = convert(Vector{Float64}, calc_mean_median(μhats)[2])
    df[df_validation_index, :"BNN_Median_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_validation)[2])
    df[df_test_index, :"BNN_Median_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_test)[2])
    df[df_train_index, :"BNN_Median_σ"] = convert(Vector{Float64}, calc_mean_median(σhats)[2])
    df[df_validation_index, :"BNN_Median_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_validation)[2])
    df[df_test_index, :"BNN_Median_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_test)[2])

    df[df_train_index, :"BBB_Mean_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_bbb)[1])
    df[df_validation_index, :"BBB_Mean_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_validation_bbb)[1])
    df[df_test_index, :"BBB_Mean_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_test_bbb)[1])
    df[df_train_index, :"BBB_Mean_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_bbb)[1])
    df[df_validation_index, :"BBB_Mean_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_validation_bbb)[1])
    df[df_test_index, :"BBB_Mean_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_test_bbb)[1])

    df[df_train_index, :"BBB_Median_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_bbb)[2])
    df[df_validation_index, :"BBB_Median_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_validation_bbb)[2])
    df[df_test_index, :"BBB_Median_μ"] = convert(Vector{Float64}, calc_mean_median(μhats_test_bbb)[2])
    df[df_train_index, :"BBB_Median_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_bbb)[2])
    df[df_validation_index, :"BBB_Median_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_validation_bbb)[2])
    df[df_test_index, :"BBB_Median_σ"] = convert(Vector{Float64}, calc_mean_median(σhats_test_bbb)[2])

    df[:, :"Garch_μ"] = convert(Vector{Float64}, fill(garch_μ, length(garch_σ)))
    df[:, :"Garch_σ"] = convert(Vector{Float64}, garch_σ)
    return df
end
 
function calc_mean_median(mat)
    mean_vec = mean(mat, dims=2)
    median_vec = median(mat, dims=2)

    return vec(mean_vec), vec(median_vec)
end

# Function to estimate GARCH volatility
function fittedVol(L, w_hat, a_hat, b_hat)
    sigmaHead = zeros(Float64, length(L))
    sigmaHead[1] = w_hat / (1 - a_hat - b_hat)
    for j in 2:length(L)
        sigmaHead[j] = w_hat + a_hat * L[j-1]^2 + b_hat * sigmaHead[j-1]
    end
    return sqrt.(sigmaHead)
end

# Function to calculate RMSE 
function calc_rmse(df, indices, proxy_vol, predict_vals)
    rmse = Dict()

    for (dataset, index) in indices
        for (method, pred_val) in predict_vals
            rmse[(method, dataset)] = sqrt(mean(abs2, proxy_vol[index] .- pred_val[index].^2))
        end
    end
    
    return rmse
end

# Function to create RMSE DataFrame
function create_rmse_dataframe(rmse_values, methods, datasets)
    train_rmse = [rmse_values[(method, "train")] for method in methods]
    val_rmse = [rmse_values[(method, "validation")] for method in methods]
    test_rmse = [rmse_values[(method, "test")] for method in methods]

    df_rmse = DataFrame(
        Method = methods,
        Train = train_rmse,
        Validation = val_rmse,
        Test = test_rmse
    )
    
    return df_rmse
end

#convert dictionary to DataFrame 
function convert_dict_to_df(dict)
    df = DataFrame(quantile = Float64[], 
                   PF = Float64[], 
                   TUFF = Int[],
                   LRTUFF = Float64[],
                   LRUC = Float64[],
                   LRIND = Float64[],
                   LRCC = Float64[],
                   BASEL = Int[])

    for (k, v) in dict
        push!(df, (Float64(k), Float64(v.PF), v.TUFF, Float64(v.LRTUFF), Float64(v.LRUC), Float64(v.LRIND), Float64(v.LRCC), v.BASEL))
    end
    return df
end

function add_empty_row(df::DataFrame)
    empty_row = DataFrame([[missing] for _ in 1:size(df, 2)], names(df))
    return vcat(df, empty_row)
end

function convert_dict_to_df_with_name(dict, name)
    df = convert_dict_to_df(dict)
    df[!, :source] = fill(name, size(df, 1))
    df = add_empty_row(df) # Add an empty row at the end
    return df
end

function create_and_concatenate_dfs(y_dict, VaRs_dict, quantiles)
    df_list = []
    for key in keys(y_dict)
        risk = VaR(y_dict[key], VaRs_dict[key], quantiles)
        df = convert_dict_to_df_with_name(risk, key)
        push!(df_list, df)
    end
    
    combined_df = vcat(df_list...)
    
    return combined_df
end

# Function to convert all Float32 columns to Float64 in a DataFrame
function convert_float32_to_float64!(df::DataFrame)
    for col in names(df)
        if eltype(df[!, col]) == Float32
            df[!, col] = convert.(Float64, df[!, col])
        end
    end
    return df
end

####### ------------------------------------------------------------------------------#########
#######                               Main Function                                   #########
####### ------------------------------------------------------------------------------#########

function backtest_and_save_to_excel(
    net::Flux.Chain{T},
    train_index::UnitRange{Int},
    val_index::UnitRange{Int},
    test_index::UnitRange{Int},
    degrees_f::Float32,
    quantiles::Vector{Float32},
    lag
) where {T}

    # Specify the directory you want to create
    new_dir = "VaR_Single_$net"

    # Create the directory
    if !isdir(new_dir)
    mkdir(new_dir)
    end

    #load data
    file_path = "src/data/etfReturns.csv"
    raw_data = load_data_portfolio(file_path)
    #data processing
    x_train, y_train, y_val, y_test, y_full, data, df_train_index, df_validation_index, df_test_index = preprocess_data_portfolio(raw_data, train_index, val_index, test_index)
    #t_dist quantile
    quant = calculate_quantile(degrees_f, quantiles)

    ###GARCH estimation
    # formal test for the presence of volatility clustering is Engle's (1982) 
    ARCHLMTest(y_train, 1)
    am = fit(GARCH{1, 1}, y_train,dist=StdT);
    # Now we can get the coefficients
    coefficients = coef.(am)
    garch_σ = fittedVol(y_full, coefficients[1], coefficients[2], coefficients[3])[(lag + 1):end]
    garch_μ = coefficients[5]
    # σ^2 as a proxy
    proxy_vol_2 = y_full[(lag + 1):end].^2

    #construct bnn
    bnn = get_bnn_data(net,x_train, y_train, degrees_f)
    #Map estimated network parameters and network ouptputs at training set
    θmap, μ_hat, σ_hat = calculate_map_estimation(bnn, x_train)
    #Map estimation network parameters and network ouptputs at validation and test sets
    μ_hat_val, σ_hat_val, μ_hat_test, σ_hat_test = calculate_test_map_estimation(bnn, train_index, val_index, test_index, θmap, y_full)

    #Bnn estimated network parameters and network ouptputs for all sets   
    ch, μhats, σhats, μhats_validation, σhats_validation , μhats_test, σhats_test = get_bnn_predictions(bnn, θmap, y_full, train_index, val_index, test_index, quant)
 
    #Bbb estimated network parameters and network ouptputs for all sets   
    ch_bbb, μhats_bbb, σhats_bbb, μhats_validation_bbb, σhats_validation_bbb, μhats_test_bbb, σhats_test_bbb = get_bbb_predictions(bnn, y_full, train_index,val_index, test_index, quant)

    #Store the μ and σ estimations of the t distribution using GARCH,MAP,BNN,BBB
    df = update_dataframe(data, df_train_index,df_validation_index, df_test_index, μ_hat, σ_hat, μ_hat_val, σ_hat_val, μ_hat_test, σ_hat_test,μhats, σhats, μhats_validation, σhats_validation , μhats_test, σhats_test,μhats_bbb, σhats_bbb, μhats_validation_bbb, σhats_validation_bbb, μhats_test_bbb, σhats_test_bbb,garch_μ, garch_σ)

    ### Create a compact RMSE Data Frame 
    indices = Dict("train" => df_train_index, "validation" => df_validation_index, "test" => df_test_index)
    predict_vals = Dict(
        "MAP" => df[:, :MAP_σ], 
        "GARCH" => df[:, :Garch_σ], 
        "BNN_Mean" => df[:, :BNN_Mean_σ],
        "BNN_Median" => df[:, :BNN_Median_σ],
        "BBB_Mean" => df[:, :BBB_Mean_σ],
        "BBB_Median" => df[:, :BBB_Median_σ]
    )

    rmse_values = calc_rmse(df, indices, proxy_vol_2, predict_vals)
    methods = ["MAP", "GARCH", "BNN_Mean", "BNN_Median", "BBB_Mean", "BBB_Median"]
    datasets = ["train", "validation", "test"]

    df_rmse = create_rmse_dataframe(rmse_values, methods, datasets)
    
    ##### Save QQ plots
    save_qq_plot(train_index, val_index, test_index, y_full, y_train, y_val, y_test, bnn, ch, ch_bbb, new_dir)

    ##### VaR calculations for 1%, 5% and 10% level for all estimation methods
    VaRs_Garch = bnn_var_prediction(coefficients[5], garch_σ, quant)

    VaRs_MAP = bnn_var_prediction(μ_hat, σ_hat, quant)
    VaRs_val_MAP = bnn_var_prediction(μ_hat_val, σ_hat_val,quant)
    VaRs_test_MAP = bnn_var_prediction(μ_hat_test, σ_hat_test, quant)

    VaRs_bnn = bnn_var_prediction(μhats,σhats,quant)
    VaRs_validation_bnn = bnn_var_prediction(μhats_validation,σhats_validation,quant)
    VaRs_test_bnn = bnn_var_prediction(μhats_test,σhats_test,quant)

    VaRs_bbb = bnn_var_prediction(μhats_bbb, σhats_bbb, quant)
    VaRs_validation_bbb = bnn_var_prediction(μhats_validation_bbb, σhats_validation_bbb,quant)
    VaRs_test_bbb = bnn_var_prediction(μhats_test_bbb, σhats_test_bbb, quant)

    #### get combined_risk_df
    y_dict = OrderedDict(
        "risk_Garch_train" => y_train,
        "risk_Garch_validation" => y_val,
        "risk_Garch_test" => y_test,
        "risk_map_train" => y_train,
        "risk_map_validation" => y_val,
        "risk_map_test" => y_test,
        "risk_bnn_train_mean" => y_train,
        "risk_bnn_validation_mean" => y_val,
        "risk_bnn_test_mean" => y_test,
        "risk_bnn_train_median" => y_train,
        "risk_bnn_validation_median" => y_val,
        "risk_bnn_test_median" => y_test,
        "risk_bbb_train_mean" => y_train,
        "risk_bbb_validation_mean" => y_val,
        "risk_bbb_test_mean" => y_test,
        "risk_bbb_train_median" => y_train,
        "risk_bbb_validation_median" => y_val,
        "risk_bbb_test_median" => y_test
    )

    VaRs_dict = OrderedDict(
        "risk_Garch_train" => VaRs_Garch[df_train_index,:],
        "risk_Garch_validation" => VaRs_Garch[df_validation_index,:],
        "risk_Garch_test" => VaRs_Garch[df_test_index,:],
        "risk_map_train" => VaRs_MAP,
        "risk_map_validation" => VaRs_val_MAP,
        "risk_map_test" => VaRs_test_MAP,
        "risk_bnn_train_mean" => VaRs_bnn[:,:,1],
        "risk_bnn_validation_mean" => VaRs_validation_bnn[:,:,1],
        "risk_bnn_test_mean" => VaRs_test_bnn[:,:,1],
        "risk_bnn_train_median" => VaRs_bnn[:,:,2],
        "risk_bnn_validation_median" => VaRs_validation_bnn[:,:,2],
        "risk_bnn_test_median" => VaRs_test_bnn[:,:,2],
        "risk_bbb_train_mean" => VaRs_bbb[:,:,1],
        "risk_bbb_validation_mean" => VaRs_validation_bbb[:,:,1],
        "risk_bbb_test_mean" => VaRs_test_bbb[:,:,1],
        "risk_bbb_train_median" => VaRs_bbb[:,:,2],
        "risk_bbb_validation_median" => VaRs_validation_bbb[:,:,2],
        "risk_bbb_test_median" => VaRs_test_bbb[:,:,2]
    )

    combined_risk_df = create_and_concatenate_dfs(y_dict, VaRs_dict, quantiles)
    
    # Define model parameters
    parameters = Dict(
        "net" => string(net),  # Convert net to a string to store it
        "train_index" => string(train_index),
        "validation_index" => string(val_index),
        "test_index" => string(test_index),
        "degrees_f" => degrees_f,
        "quantiles" => join(quantiles, ", "),  # Convert array to a comma-separated string
        "lag" => lag
    )

    # Convert the parameters dictionary into a DataFrame
    parameters_df = DataFrame(parameters)
    parameters_df = DataFrame([eltype(c) == Float32 ? convert(Vector{Float64}, c) : c for c in eachcol(parameters_df)], names(parameters_df))


    ##### Plot VaR values with returns to see visually 
    plot_and_save(y_train, VaRs_Garch[df_train_index,2], "GARCH-Driven VaR Assessment on Training Set", "garch_train_plot.png", new_dir)
    plot_and_save(y_val, VaRs_Garch[df_validation_index,2], "GARCH-Driven VaR Assessment on Validation Set", "garch_validation_plot.png", new_dir)
    plot_and_save(y_test, VaRs_Garch[df_test_index,2], "GARCH-Driven VaR Assessment on Test Set", "garch_test_plot.png", new_dir)

    plot_and_save(y_train, VaRs_MAP[:,2], "MAP-Driven VaR Assessment on Training Data", "MAP_train_plot.png", new_dir)
    plot_and_save(y_val, VaRs_val_MAP[:,2], "MAP-Driven VaR Assessment on Validation Data", "MAP_validation_plot.png", new_dir)
    plot_and_save(y_test, VaRs_test_MAP[:,2], "MAP-Driven VaR Assessment on Test Data", "MAP_test_plot.png", new_dir)

    plot_and_save(y_train, VaRs_bnn[:,2,1], "BNN-Driven VaR Assessment on Training Data", "BNN_train_plot.png", new_dir)
    plot_and_save(y_val, VaRs_validation_bnn[:,2,1], "BNN-Driven VaR Assessment on Validation Data", "BNN_validation_plot.png", new_dir)
    plot_and_save(y_test, VaRs_test_bnn[:,2,1], "BNN-Driven VaR Assessment on Test Data", "BNN_test_plot.png", new_dir)

    plot_and_save(y_train, VaRs_bbb[:,2,1], "BBB-Driven VaR Assessment on Training Data", "BBB_train_plot.png", new_dir)
    plot_and_save(y_val, VaRs_validation_bbb[:,2,1], "BBB-Driven VaR Assessment on Validation Data", "BBB_validation_plot.png", new_dir)
    plot_and_save(y_test, VaRs_test_bbb[:,2,1], "BBB-Driven VaR Assessment on Test Data", "BBB_test_plot.png", new_dir)


    ####Plot and Save Volatility Estimations and Returns
    plot_and_save_combined(
        df, 
        raw_data, 
        train_index, 
        test_index, 
        val_index, 
        new_dir, 
        ["Returns", "Garch_σ", "MAP_σ", "BNN_Mean_σ", "BBB_Mean_σ"], 
        ["Return over Time", "Garch-Volatility over Time", "MAP-Volatility over Time", "SGNT-S-Volatility over Time", "BBB-Volatility over Time"], 
        ["Returns", "Volatility", "Volatility", "Volatility", "Volatility"]
    )

    # Specify the filename, including the new directory
    filename = joinpath(new_dir, "my_results.xlsx")

    # Convert Float32 columns to Float64 in all dataframes
    df = convert_float32_to_float64!(df)
    df_rmse = convert_float32_to_float64!(df_rmse)
    combined_risk_df = convert_float32_to_float64!(combined_risk_df)

    # Write DataFrames to new worksheets in a workbook
    XLSX.openxlsx(filename, mode="w") do xf
    # Make sure all DataFrames are converted to compatible types here

    sheet1 = XLSX.addsheet!(xf, "Parameters")
    XLSX.writetable!(sheet1, Tables.columntable(parameters_df))

    sheet2 = XLSX.addsheet!(xf, "Return & Parameter Estimations")
    XLSX.writetable!(sheet2, Tables.columntable(df))

    sheet3 = XLSX.addsheet!(xf, "RMSE")
    XLSX.writetable!(sheet3, Tables.columntable(df_rmse))
      
    sheet4 = XLSX.addsheet!(xf, "Risk Metrics")
    XLSX.writetable!(sheet4, Tables.columntable(combined_risk_df))

    # ... repeat for each DataFrame
    end

end   

#################### Run Main Function ####################

#load data
file_path = "src/data/etfReturns.csv"
raw_data = load_data_portfolio(file_path)

train_index = 110:3450
val_index = 3451:3700
test_index = 3701:3950
degrees_f = Float32(5)  # Degrees of freedom,
quantiles = Float32.([0.01,0.05,0.1])
lag = 5

#network structures
network_structures = [
    Flux.Chain(RNN(1, 2), Dense(2, 2)), 
    Flux.Chain(RNN(1, 4), Dense(4, 2)), 
    Flux.Chain(RNN(1, 6), Dense(6, 2)), 
    Flux.Chain(RNN(1, 10), Dense(10, 2)),
    Flux.Chain(RNN(1, 2), Dense(2, 2, sigmoid), Dense(2, 2)), 
    Flux.Chain(RNN(1, 6), Dense(6, 6, sigmoid), Dense(6, 2)), 
    Flux.Chain(RNN(1, 2), Dense(2, 2, relu), Dense(2, 2)), 
    Flux.Chain(RNN(1, 6), Dense(6, 6, relu), Dense(6, 2)), 
    Flux.Chain(LSTM(1, 2), Dense(2, 2)), 
    Flux.Chain(LSTM(1, 4), Dense(4, 2)), 
    Flux.Chain(LSTM(1, 6), Dense(6, 2)), 
    Flux.Chain(LSTM(1, 10), Dense(10, 2)), 
    Flux.Chain(LSTM(1, 2), Dense(2, 2, sigmoid), Dense(2, 2)), 
    Flux.Chain(LSTM(1, 6), Dense(6, 6, sigmoid), Dense(6, 2)), 
    Flux.Chain(LSTM(1, 2), Dense(2, 2, relu), Dense(2, 2)), 
    Flux.Chain(LSTM(1, 6), Dense(6, 6, relu), Dense(6, 2)), 
]

#run main function 
for net in network_structures
    try
        backtest_and_save_to_excel(net, train_index, val_index, test_index, degrees_f, quantiles,lag)
    catch e
        println("An error occurred: ", e)
        continue
    end
end
























# ###### Some additional code after final feedback for better visualizaion purposes#######


# df_full = (minimum(train_index):maximum(test_index))[(lag + 1):end]


# p = plot(raw_data[df_full,:Date], percantage_portfolio[df_full],
# title = "Portfolio Return",
# xlab = "Date",
# ylab = "returns",
# legend = false,
# linewidth = 2,
# linecolor = :blue)

# date1 = raw_data[minimum(val_index),:Date]
# date2 = raw_data[minimum(test_index),:Date]

# plot!(p, [date1, date1], [minimum(percantage_portfolio[df_full]), maximum(percantage_portfolio[df_full])], line=:dash, linewidth=2, linecolor=:red)
# plot!(p, [date2, date2], [minimum(percantage_portfolio[df_full]), maximum(percantage_portfolio[df_full])], line=:dash, linewidth=2, linecolor=:red)

# savefig(p, "full_data")

# train_index = 1:2100
# val_index = 2101:2300
# test_index = 2301:4280
# 1800-3500-3700


# raw_data[minimum(train_index),:Date]
# raw_data[minimum(val_index),:Date]
# raw_data[minimum(test_index),:Date]

# df_full
# raw_data[df_full]
# portfolio[df_full]

# mean_value = mean(percantage_portfolio)       # Mean
# median_value = median(percantage_portfolio)   # Median
# std_deviation = std(percantage_portfolio)     # Standard Deviation
# variance = var(percantage_portfolio)          # Variance
# minimum_value = minimum(percantage_portfolio) # Minimum
# maximum_value = maximum(percantage_portfolio) # Maximum

# raw_data[3950,:Date]
# portfolio[109]
# portfolio[3450]
# portfolio[3700]




# using StatsBase

# percantage_portfolio
# percantage_portfolio = vec(percantage_portfolio)

# # Calculate various tail quantiles
# q_1 = quantile(percantage_portfolio, 0.01)
# q_5 = quantile(percantage_portfolio, 0.05)
# q_25 = quantile(percantage_portfolio, 0.25)
# q_75 = quantile(percantage_portfolio, 0.75)
# q_95 = quantile(percantage_portfolio, 0.95)
# q_99 = quantile(percantage_portfolio, 0.99)
# # Calculate skewness and kurtosis
# skew = skewness(percantage_portfolio)
# kurt = kurtosis(percantage_portfolio)

# # Display the results
# println("1th percentile: $q_1")
# println("5th percentile: $q_5")
# println("25th percentile: $q_25")
# println("75th percentile: $q_75")
# println("95th percentile: $q_95")
# println("99th percentile: $q_99")
# println("Skewness: $skew")
# println("Kurtosis: $kurt")