using CSV, DataFrames
using Dates,Statistics
using JLD2
using ProgressMeter
using XLSX
using ARCHModels

####### -----------------------------------------------------------------------------------------#########
#######                                       Data Preperation                                   #########
####### -----------------------------------------------------------------------------------------#########
Random.seed!(1212)

#network structures
network_structures = [
    Flux.Chain(RNN(1, 2), Dense(2, 1)), 
    Flux.Chain(RNN(1, 4), Dense(4, 1)), 
    Flux.Chain(RNN(1, 6), Dense(6, 1)), 
    Flux.Chain(RNN(1, 10), Dense(10, 1)),
    Flux.Chain(RNN(1, 2), Dense(2, 2, sigmoid), Dense(2, 1)), 
    Flux.Chain(RNN(1, 6), Dense(6, 6, sigmoid), Dense(6, 1)), 
    Flux.Chain(RNN(1, 2), Dense(2, 2, relu), Dense(2, 1)), 
    Flux.Chain(RNN(1, 6), Dense(6, 6, relu), Dense(6, 1)), 
    Flux.Chain(LSTM(1, 2), Dense(2, 1)), 
    Flux.Chain(LSTM(1, 4), Dense(4, 1)), 
    Flux.Chain(LSTM(1, 6), Dense(6, 1)), 
    Flux.Chain(LSTM(1, 10), Dense(10, 1)), 
    Flux.Chain(LSTM(1, 2), Dense(2, 2, sigmoid), Dense(2, 1)), 
    Flux.Chain(LSTM(1, 6), Dense(6, 6, sigmoid), Dense(6, 1)), 
    Flux.Chain(LSTM(1, 5), Dense(5, 5, relu), Dense(5, 1)), 
    Flux.Chain(LSTM(1, 10), Dense(10, 10, relu), Dense(10, 1)), 
]

function load_data_portfolio(file_path)
    data = CSV.read(file_path, DataFrame)
    data[!, "Date"] = Date.(data[!, "Date"], "dd/mm/yyyy")
    # Replace missing value with the mean (note that there is only one missing value in 1477x18)
    data = replace_missing_with_mean(data)
    # Convert date to numerical representation
    data[!, :Date] = Dates.value.(data[!, :Date])
    return data
end


# Preprocess data
#Scaling the input data 
function preprocess_data_portfolio(data, train_index, test_index)
    # Convert DataFrame to Matrix
    matrix_data = Matrix(data)
    matrix_data = Matrix{Float32}(matrix_data)

    # Extract date column
    date_column = matrix_data[:, 1]

    # Calculate row-wise mean of assets / first raw needs to be = -.009415702
    portfolio = mean(matrix_data[:, 2:end], dims=2)

    # Create portfolio matrix
    portfolio_matrix = hcat(date_column, portfolio)
    
    full_index = minimum(train_index):maximum(test_index)
    # #------
    # #Scaling the input data 
    # # Get the mean and standard deviation of each column
    # scaled_train_index = train_index .- minimum(train_index) .+ 1

    # means = mean(X[scaled_train_index], dims=1)
    # stddevs = std(X[scaled_train_index], dims=1)

    # # Normalize the columns by subtracting the mean and dividing by the standard deviation
    # X = (X .- means) ./ stddevs
    #------
    #multiply the return to avoid numerical problems !!!!!!!!! do not forget !!!!!!
    X = matrix_data[:,2:end] .* 100
    y = portfolio .* 100
    
    X_train, y_train, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
    X_full, y_full = X[full_index,:], y[full_index]
    #arrange a train and test set properly
    x_train = make_rnn_tensor(reshape(y_train, :, 1), 5 + 1)
    y_train = vec(x_train[end, :, :])
    x_train = x_train[1:end-1, :, :]
    
    
 
    # data = data[full_index,:]
    # data = data[6:end,:]

    # data[!, :MAP] = fill(Float64(1), size(data, 1))
    # data[!, :Mean] = fill(Float64(2), size(data, 1))
    # data[!, :Median] = fill(Float64(3), size(data, 1))

    # df_train_index = (train_index .- minimum(train_index) .+ 1)[1:end-5]
    # df_test_index = test_index .- minimum(test_index) .+ 1 .+ maximum(df_train_index)

    # y = (returns .- means) ./ stddevs
    # X = hcat(y,X)

    # X_train, y_train, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
    # X_full, y_full = X[full_index,:], y[full_index]

    # X_train = make_rnn_tensor(X_train, 5 + 1)
    # y_train = vec(X_train[end, 1, :])
    # X_train = X_train[1:end-1, 2:end, :]

    return x_train, y_train, y_test, y_full #, data, df_train_index, df_test_index
end

#--done
# Calculate Quantile
function calculate_quantile(degrees_f, quantiles)
    t_dist = TDist(degrees_f)
    quant = quantile(t_dist, quantiles)
    return quant
end


# Calculate MAP Estimation
function calculate_map_estimation(bnn, x_train,quant)
    opt = FluxModeFinder(bnn, Flux.RMSProp())
    θmap = find_mode(bnn, 10, 10000, opt)
    nethat = bnn.like.nc(θmap)
    log_σ  = vec([nethat(xx) for xx in eachslice(x_train; dims =1 )][end])
    σ_hat = exp.(log_σ)
    VaRs_MAP = bnn_var_prediction(σ_hat,θmap,quant)
    return θmap, σ_hat, VaRs_MAP
end




# Calculate Test Set MAP Estimation
function calculate_test_map_estimation(bnn, train_index, test_index, θmap, y_full,quant)
    σ_hat_test = estimate_test_σ(bnn, train_index, test_index, θmap, y_full)
    VaRs_test_MAP = bnn_var_prediction(σ_hat_test,θmap,quant)
    return σ_hat_test, VaRs_test_MAP
end


#######   BNN Estimation   ######

function get_bnn_predictions(bnn, θmap, y_full, train_index, test_index, quant)
    sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)

    # Sampling 
    ch = mcmc(bnn, 10, 50_000, sampler, θstart = θmap)
    ch = ch[:, end-20_000+1:end]
    #chain = Chains(ch')

    # Training-set Bayesian Neural Network (BNN) mean/median VaRs estimation
    σhats = naive_train_bnn_σ_prediction_recurrent(bnn,ch)
    VaRs_bnn = bnn_var_prediction(σhats,ch,quant)

    # Test set estimation - computationally expensive
    σhats_test = naive_test_bnn_σ_prediction_recurrent(bnn,y_full,train_index,test_index,ch)
    VaRs_test_bnn = bnn_var_prediction(σhats_test,ch,quant)

    return σhats, σhats_test, VaRs_bnn, VaRs_test_bnn
end

function get_bbb_predictions(bnn, θmap, y_full, train_index, test_index, quant)
    q, params, losses = bbb(bnn, 10, 2_000; mc_samples = 1, opt = Flux.RMSProp(), n_samples_convergence = 10)

    ch_bbb = rand(q, 20_000)
    
    # Training-set Bayesian Neural Network (BNN) mean/median VaRs estimation
    σhats_bbb = naive_train_bnn_σ_prediction_recurrent(bnn,ch_bbb)
    VaRs_bnn_bbb = bnn_var_prediction(σhats,ch_bbb,quant)
        
    # Test set estimation - computationally expensive
    σhats_test_bbb = naive_test_bnn_σ_prediction_recurrent(bnn,y_full,train_index,test_index,ch_bbb)
    VaRs_test_bnn_bbb = bnn_var_prediction(σhats_test,ch_bbb,quant)

    return σhats_bbb, σhats_test_bbb, VaRs_bnn_bbb, VaRs_test_bnn_bbb
end



function backtest_and_save_to_excel_thesis(
    net::Flux.Chain{T},
    train_index::UnitRange{Int},
    test_index::UnitRange{Int},
    degrees_f::Float32,
    quantiles::Vector{Float32},
) where {T}

    # Specify the directory you want to create
    new_dir = "$net"

    # Create the directory
    if !isdir(new_dir)
    mkdir(new_dir)
    end

    file_path = "src/data/etfReturns.csv"
    # The value to find the quantile for, replace ... with the actual value.
    #sampler

    data = load_data_portfolio(file_path)
    x_train, y_train, y_test, y_full = preprocess_data_portfolio(data, train_index, test_index)
    quant = calculate_quantile(degrees_f, quantiles)
    bnn = get_bnn(net,x_train,y_train,df)
    θmap, σ_hat, VaRs_MAP = calculate_map_estimation(bnn, x_train,quant)
    σ_hat_test, VaRs_test_MAP = calculate_test_map_estimation(bnn, train_index, test_index, θmap, y_full,quant)
    σhats, σhats_test, VaRs_bnn, VaRs_test_bnn = get_bnn_predictions(bnn, θmap, y_full, train_index, test_index, quant)
    σhats_bbb, σhats_test_bbb, VaRs_bnn_bbb, VaRs_test_bnn_bbb = get_bbb_predictions(bnn, θmap, y_full, train_index, test_index, quant)

    # Define model parameters
    parameters = Dict(
        "net" => string(net),  # Convert net to a string to store it
        "train_index" => string(train_index),
        "test_index" => string(test_index),
        "degrees_f" => degrees_f,
        "quantiles" => join(quantiles, ", "),  # Convert array to a comma-separated string
    )

    # Convert the parameters dictionary into a DataFrame
    parameters_df = DataFrame(parameters)
    parameters_df = DataFrame([eltype(c) == Float32 ? convert(Vector{Float64}, c) : c for c in eachcol(parameters_df)], names(parameters_df))

    #Analysis

    ### MAP
    #train
    Var_Map_Train = VaRLR(y_train,VaRs_MAP,quantiles)
    Var_Map_Train_df = convert_dict_to_df(Var_Map_Train)

    #test
    Var_Map_Test = VaRLR(y_test,VaRs_test_MAP,quantiles)
    Var_Map_Test_df = convert_dict_to_df(Var_Map_Test)

    #BNN
    #train 
    Var_BNN_Train_Mean = VaRLR(y_train,VaRs_bnn[:,:,1],quantiles) #mean
    Var_BNN_Train_Mean_df = convert_dict_to_df(Var_BNN_Train_Mean)

    Var_BNN_Train_Median = VaRLR(y_train,VaRs_bnn[:,:,2],quantiles) #median
    Var_BNN_Train_Median_df = convert_dict_to_df(Var_BNN_Train_Median)

    #test 
    Var_BNN_Test_Mean = VaRLR(y_test,VaRs_test_bnn[:,:,1],quantiles) #mean
    Var_BNN_Test_Mean_df = convert_dict_to_df(Var_BNN_Test_Mean)

    Var_BNN_Test_Median = VaRLR(y_test,VaRs_test_bnn[:,:,2],quantiles) #median
    Var_BNN_Test_Median_df = convert_dict_to_df(Var_BNN_Test_Median)

    #VI
    #train 
    Var_BBB_Train_Mean = VaRLR(y_train,VaRs_bnn_bbb[:,:,1],quantiles) #mean
    Var_BBB_Train_Mean_df = convert_dict_to_df(Var_BBB_Train_Mean)

    Var_BBB_Train_Median = VaRLR(y_train,VaRs_bnn_bbb[:,:,2],quantiles) #median
    Var_BBB_Train_Median_df = convert_dict_to_df(Var_BBB_Train_Median)


    #test 
    Var_BBB_Test_Mean = VaRLR(y_test,VaRs_test_bnn_bbb[:,:,1],quantiles) #mean
    Var_BBB_Test_Mean_df = convert_dict_to_df(Var_BBB_Test_Mean)

    Var_BBB_Test_Median = VaRLR(y_test,VaRs_test_bnn_bbb[:,:,2],quantiles) #median
    Var_BBB_Test_Median_df = convert_dict_to_df(Var_BBB_Test_Median)



    # Specify the filename, including the new directory
    filename = joinpath(new_dir, "my_results.xlsx")

    # Write DataFrames to new worksheets in a workbook
    XLSX.openxlsx(filename, mode="w") do xf
        # Make sure all DataFrames are converted to compatible types here

        sheet1 = XLSX.addsheet!(xf, "Parameters")
        XLSX.writetable!(sheet1, Tables.columntable(parameters_df))

        sheet2 = XLSX.addsheet!(xf, "Var_Map_Train_df")
        XLSX.writetable!(sheet2, Tables.columntable(Var_Map_Train_df))

        sheet3 = XLSX.addsheet!(xf, "Var_Map_Test_df")
        XLSX.writetable!(sheet3, Tables.columntable(Var_Map_Test_df))

        sheet4 = XLSX.addsheet!(xf, "Var_BNN_Train_Mean_df")
        XLSX.writetable!(sheet4, Tables.columntable(Var_BNN_Train_Mean_df))

        sheet5 = XLSX.addsheet!(xf, "Var_BNN_Train_Median_df")
        XLSX.writetable!(sheet5, Tables.columntable(Var_BNN_Train_Median_df))

        sheet6 = XLSX.addsheet!(xf, "Var_BNN_Test_Mean_df")
        XLSX.writetable!(sheet6, Tables.columntable(Var_BNN_Test_Mean_df))

        sheet7 = XLSX.addsheet!(xf, "Var_BNN_Test_Median_df")
        XLSX.writetable!(sheet7, Tables.columntable(Var_BNN_Test_Median_df))

        sheet8 = XLSX.addsheet!(xf, "Var_BBB_Train_Mean_df")
        XLSX.writetable!(sheet8, Tables.columntable(Var_BBB_Train_Mean_df))
        
        sheet9 = XLSX.addsheet!(xf, "Var_BBB_Train_Median_df")
        XLSX.writetable!(sheet9, Tables.columntable(Var_BBB_Train_Median_df))

        sheet10 = XLSX.addsheet!(xf, "Var_BBB_Test_Mean_df")
        XLSX.writetable!(sheet10, Tables.columntable(Var_BBB_Test_Mean_df))
        
        sheet11 = XLSX.addsheet!(xf, "Var_BBB_Test_Median_df")
        XLSX.writetable!(sheet11, Tables.columntable(Var_BBB_Test_Median_df))
        # ... repeat for each DataFrame

    end
    # Save plots as images
    plot(1:length(y_train), y_train, label="Actual")
    plot!(1:length(y_train),σ_hat, label="Estimated")
    # Specify the filename, including the new directory
    plotfile_train = joinpath(new_dir, "train_plot.png")
    savefig(plotfile_train)

    plot(1:length(y_test), y_test, label="Actual")
    plot!(1:length(y_test),σ_hat_test, label="Estimated")
    # Specify the filename, including the new directory
    plotfile_test = joinpath(new_dir, "test_plot.png")
    savefig(plotfile_test)
end
network_structures[1]
train_index = 1:2000
test_index = 2000:2001
degrees_f = Float32(5)  # Degrees of freedom, replace ... with the actual value.
quantiles = Float32.([0.01,0.05,0.1]) 

 backtest_and_save_to_excel_thesis(network_structures[1],train_index, test_index, degrees_f,quantiles)

# @save "θmap_full_data_LSTM(1, 2), Dense(2, 1).jld2" θmap 
# @load "θmap_full_data_LSTM(1, 2), Dense(2, 1).jld2" θmap
# θmap

#very good
# @save "θmap_2008_data_LSTM(1, 2), Dense(2, 1) 10k with *100 t=3000:4000,t=3001:4100.jld2" θmap 
# @load "θmap_2008_data_LSTM(1, 2), Dense(2, 1) 10k with *100 t=3000:4000,t=3001:4100.jld2" θmap
# θmap



######## Helper Functions

function get_bnn(net::Flux.Chain{T},x, y,df) where {T}
    #net = Chain(LSTM(2, 6), Dense(6, 1))  # last layer is linear output layer
    nc = destruct(net)
    like = ArchSeqToOneTDist(nc, Normal(0, 2.5),df)
    prior = GaussianPrior(nc, 2.5f0)
    init = InitialiseAllSame(Normal(0.0f0, 2.5f0), like, prior)
    bnn = BNN(x, y, like, prior, init)
    return bnn
end

function naive_train_bnn_σ_prediction_recurrent(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    log_σhats = Array{T, 2}(undef, length(y), size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        σh = vec([net(xx) for xx in eachslice(x; dims = 1)][end])
        log_σhats[:,i] = σh    
    end
    σhats = exp.(log_σhats)
    return σhats
end
function bnn_var_prediction(σhats, draws::Array{T, 2}, quant) where {T}
    VaRs = reshape(draws[end,:], 1, :) .+ (σhats .* reshape(quant, 1, 1, :)) 
    
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

function bnn_var_prediction(σhats, θmap::Array{T, 1}, quant) where {T}
    result = hcat([σhats .* q for q in quant]...)
    return Float32.(θmap[end] .+ result)
end


function estimate_test_σ(bnn, train_index, test_index, θmap, y_full::Array{Float32, 1})
    train_index = train_index .- minimum(train_index) .+ 1
    nethat = bnn.like.nc(θmap)
    shifted_index = train_index[:] .+ length(test_index)
    x_shifted = make_rnn_tensor(reshape(y_full[shifted_index], :, 1), 5 + 1)
    x_shifted = x_shifted[1:end-1, :, :]
    log_σ_whole  = vec([nethat(xx) for xx in eachslice(x_shifted; dims =1 )][end]) 
    σ_hat_test = exp.(log_σ_whole)
    return σ_hat_test[end-(length(test_index)-1):end]
end

function naive_test_bnn_σ_prediction_recurrent(bnn,y_full::Array{Float32, 1},train_index,test_index, draws::Array{T, 2}) where {T}
    log_σ_whole = Array{T, 2}(undef, length(train_index)-5, size(draws, 2))
    train_index = train_index .- minimum(train_index) .+ 1

    shifted_index = train_index[:] .+ length(test_index)
    x_shifted = make_rnn_tensor(reshape(y_full[shifted_index], :, 1), 5 + 1)
    x_shifted = x_shifted[1:end-1, :, :]
    Threads.@threads for j=1:size(draws, 2)
        net = bnn.like.nc(draws[:, j])
        σh = vec([net(xx) for xx in eachslice(x_shifted; dims = 1)][end])
        log_σ_whole[:,j] = σh    
        end
    σhats = exp.(log_σ_whole[end-(length(test_index)-1):end,:])
    return σhats
end

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







file_path = "src/data/etfReturns.csv"
    # The value to find the quantile for, replace ... with the actual value.
    #sampler

    data = load_data_portfolio(file_path)
    x_train, y_train, y_test, y_full = preprocess_data_portfolio(data, train_index, test_index)
    quant = calculate_quantile(degrees_f, quantiles)
    bnn = get_bnn(net,x_train,y_train,df)
    θmap, σ_hat, VaRs_MAP = calculate_map_estimation(bnn, x_train,quant)
    σ_hat_test, VaRs_test_MAP = calculate_test_map_estimation(bnn, train_index, test_index, θmap, y_full,quant)
    σhats, σhats_test, VaRs_bnn, VaRs_test_bnn = get_bnn_predictions(bnn, θmap, y_full, train_index, test_index, quant)
    σhats_bbb, σhats_test_bbb, VaRs_bnn_bbb, VaRs_test_bnn_bbb = get_bbb_predictions(bnn, θmap, y_full, train_index, test_index, quant)
    
    # Define model parameters
    parameters = Dict(
        "net" => string(net),  # Convert net to a string to store it
        "train_index" => string(train_index),
        "test_index" => string(test_index),
        "degrees_f" => degrees_f,
        "quantiles" => join(quantiles, ", "),  # Convert array to a comma-separated string
    )

    # Convert the parameters dictionary into a DataFrame
    parameters_df = DataFrame(parameters)
    parameters_df = DataFrame([eltype(c) == Float32 ? convert(Vector{Float64}, c) : c for c in eachcol(parameters_df)], names(parameters_df))

    #Analysis

    ### MAP
    #train
    Var_Map_Train = VaRLR(y_train,VaRs_MAP,quantiles)
    Var_Map_Train_df = convert_dict_to_df(Var_Map_Train)

    #test
    Var_Map_Test = VaRLR(y_test,VaRs_test_MAP,quantiles)
    Var_Map_Test_df = convert_dict_to_df(Var_Map_Test)

    #BNN
    #train 
    Var_BNN_Train_Mean = VaRLR(y_train,VaRs_bnn[:,:,1],quantiles) #mean
    Var_BNN_Train_Mean_df = convert_dict_to_df(Var_BNN_Train_Mean)

    Var_BNN_Train_Median = VaRLR(y_train,VaRs_bnn[:,:,2],quantiles) #median
    Var_BNN_Train_Median_df = convert_dict_to_df(Var_BNN_Train_Median)

    #test 
    Var_BNN_Test_Mean = VaRLR(y_test,VaRs_test_bnn[:,:,1],quantiles) #mean
    Var_BNN_Test_Mean_df = convert_dict_to_df(Var_BNN_Test_Mean)

    Var_BNN_Test_Median = VaRLR(y_test,VaRs_test_bnn[:,:,2],quantiles) #median
    Var_BNN_Test_Median_df = convert_dict_to_df(Var_BNN_Test_Median)

    #VI
    #train 
    Var_BBB_Train_Mean = VaRLR(y_train,VaRs_bnn_bbb[:,:,1],quantiles) #mean
    Var_BBB_Train_Mean_df = convert_dict_to_df(Var_BBB_Train_Mean)

    Var_BBB_Train_Median = VaRLR(y_train,VaRs_bnn_bbb[:,:,2],quantiles) #median
    Var_BBB_Train_Median_df = convert_dict_to_df(Var_BBB_Train_Median)


    #test 
    Var_BBB_Test_Mean = VaRLR(y_test,VaRs_test_bnn_bbb[:,:,1],quantiles) #mean
    Var_BBB_Test_Mean_df = convert_dict_to_df(Var_BBB_Test_Mean)

    Var_BBB_Test_Median = VaRLR(y_test,VaRs_test_bnn_bbb[:,:,2],quantiles) #median
    Var_BBB_Test_Median_df = convert_dict_to_df(Var_BBB_Test_Median)



