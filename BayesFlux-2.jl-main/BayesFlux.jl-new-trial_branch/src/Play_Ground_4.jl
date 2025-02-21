using CSV, DataFrames
using Dates,Statistics
using JLD2
####### -----------------------------------------------------------------------------------------#########
#######                                       Data Preperation                                   #########
####### -----------------------------------------------------------------------------------------#########

filename = "src/data/etfReturns.csv" # replace with your actual file name

data = CSV.read(filename, DataFrame)  # read the CSV file into a DataFrame

data[!, "Date"] = Date.(data[!, "Date"], "dd/mm/yyyy")

# Replace missing value with the mean (note that there is only one missing value in 1477x18)
data = replace_missing_with_mean(data)

# Convert date to numerical representation
data[!, :Date] = Dates.value.(data[!, :Date])

# Convert DataFrame to Matrix
matrix_data = Matrix(data)
matrix_data = Matrix{Float32}(matrix_data)

# Extract date column
date_column = matrix_data[:, 1]

# Calculate row-wise mean of assets / first raw needs to be = -.009415702
portfolio = mean(matrix_data[:, 2:end], dims=2)

# Create portfolio matrix
portfolio_matrix = hcat(date_column, portfolio)

#select the time interval
# train_index = 1:924
# test_index = 925:1155
# full_index = 1:1155
train_index = 1:400
test_index = 401:500
full_index = 1:500

#multiply with 100 the return to avoid numerical problems !!!!!!!!! do not forget !!!!!!
X = matrix_data[:,2:end] .* 100
y = portfolio .* 100
X = hcat(y,X)

X_train, y_train, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
X_full, y_full = X[full_index,:], y[full_index]
#arrange a train and test set properly

X_train = make_rnn_tensor(X_train, 5 + 1)
y_train = vec(X_train[end, 1, :])
X_train = X_train[1:end-1, 2:end, :]

df = Float32(5)  # Degrees of freedom, replace ... with the actual value.
quantiles = Float32.([0.01,0.05,0.1])   # The value to find the quantile for, replace ... with the actual value.

# Creating t-distribution with given degrees of freedom
t_dist = TDist(df)

# Calculating the quantile
quant = quantile(t_dist, quantiles)

#Get Bnn
bnn = get_bnn(X_train,(y_train))

############      MAP Estimation    #########


#find MAP estimation for the likelihood and network parameters
opt = FluxModeFinder(bnn, Flux.RMSProp())
θmap = find_mode(bnn, 10, 10000, opt)

#setup the network with the MAP estimation
nethat = bnn.like.nc(θmap)

#Training-set estimation
log_σ  = vec([nethat(xx) for xx in eachslice(X_train; dims =1 )][end])
σ_hat = exp.(log_σ)
VaRs_MAP = bnn_var_prediction(σ_hat,θmap,quant)

#Training-set plot
plot(1:length(y_train), y_train, label="Actual")
plot!(1:length(y_train),σ_hat, label="Estimated")

#Test-set MAP estimation
σ_hat_test = estimate_test_σ(bnn, train_index, test_index, θmap, X_full)
VaRs_test_MAP = bnn_var_prediction(σ_hat_test,θmap,quant)

#Test-set plot
plot(1:length(y_test), y_test, label="Actual")
plot!(1:length(y_test),σ_hat_test, label="Estimated")

####-----------------

#######   BNN Estimation   ######

#training-set BNN 

#sampler
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)

#sampling 
ch = mcmc(bnn, 10, 50_000, sampler,θstart = θmap)
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')

#training-set BNN mean/median VaRs estimation
σhats = naive_train_bnn_σ_prediction_recurrent(bnn,ch)
VaRs_bnn = bnn_var_prediction(σhats,ch,quant)

#Test set estimation -computationaly expensive
σhats_test = naive_test_bnn_σ_prediction_recurrent(bnn,X_full,train_index,test_index,ch)
VaRs_test_bnn = bnn_var_prediction(σhats_test,ch,quant)


#Analysis

### MAP
#train
VaRLR(y_train,VaRs_MAP,quantiles)
#test
VaRLR(y_test,VaRs_test_MAP,quantiles)

#BNN
#train 
VaRLR(y_train,VaRs_bnn[:,:,1],quantiles) #mean
VaRLR(y_train,VaRs_bnn[:,:,2],quantiles) #median

#test 
VaRLR(y_test,VaRs_test_bnn[:,:,1],quantiles) #mean
VaRLR(y_test,VaRs_test_bnn[:,:,2],quantiles) #median











kkk = [12,12]
kkk
























# @save "θmap_full_data_LSTM(1, 2), Dense(2, 1).jld2" θmap 
# @load "θmap_full_data_LSTM(1, 2), Dense(2, 1).jld2" θmap
# θmap




# @save "θmap_2008_data_LSTM(30, 15), Dense(15, 1).jld2" θmap 
# @load "θmap_2008_data_LSTM(30, 15), Dense(15, 1).jld2" θmap
# θmap

# @save "θmap_2008_data_LSTM(30, 45), Dense(45, 1).jld2" θmap 
# @load "θmap_2008_data_LSTM(30, 45), Dense(45, 1).jld2" θmap
# θmap



# @save "θmap_2008_data_LSTM(30, 15),RNN(15, 45), Dense(45, 1) 50k.jld2" θmap 
# @load "θmap_2008_data_LSTM(30, 15),RNN(15, 45), Dense(45, 1) 50k.jld2" θmap
# θmap

# @save "θmap_2008_data_LSTM(30, 15),RNN(15, 45), Dense(45, 1) 10k without *100.jld2" θmap 
# @load "θmap_2008_data_LSTM(30, 15),RNN(15, 45), Dense(45, 1) 10k without *100.jld2" θmap
# θmap










# function estimate_test_σ(bnn, train_index, test_index, θmap, X_full::Array{Float32, 2})
#     log_σ_test = []
#     train_index = train_index .- minimum(train_index) .+ 1
#     for i in 1:(length(test_index))
#         nethat = bnn.like.nc(θmap)
#         shifted_index = train_index[:] .+ i
#         X_shifted = make_rnn_tensor(X_full[shifted_index,:], 5 + 1)
#         X_shifted = X_shifted[1:end-1, 2:end, :]
#         log_σ_whole  = vec([nethat(xx) for xx in eachslice(X_shifted; dims =1 )][end]) 
#         push!(log_σ_test, log_σ_whole[end])
#     end
#     σ_hat_test = exp.(log_σ_test)
#     return σ_hat_test
# end

# function naive_test_bnn_σ_prediction_recurrent(bnn,X_full::Array{Float32, 2},train_index,test_index, draws::Array{T, 2}) where {T}
#     σhats = Array{T, 2}(undef, length(test_index), size(draws, 2))
#     log_σ_whole = Array{T, 2}(undef, length(train_index)-5, size(draws, 2))
#     train_index = train_index .- minimum(train_index) .+ 1
#     # Create the progress bar
#     p = Progress(length(test_index), 1, "Processing...", 50)

#     for i in 1:length(test_index)
#         next!(p) # Update the progress bar
#         shifted_index = train_index[:] .+ i
#         X_shifted = make_rnn_tensor(X_full[shifted_index,:], 5 + 1)
#         X_shifted = X_shifted[1:end-1, 2:end, :]
#         Threads.@threads for j=1:size(draws, 2)
#             net = bnn.like.nc(draws[:, j])
#             σh = vec([net(xx) for xx in eachslice(X_shifted; dims = 1)][end])
#             log_σ_whole[:,j] = σh    
#         end
#         σhats[i,:] = exp.(log_σ_whole[end,:])
#     end    
#     return σhats
# end


function estimate_test_σ(bnn, train_index, test_index, θmap, X_full::Array{Float32, 2})
    train_index = train_index .- minimum(train_index) .+ 1
    nethat = bnn.like.nc(θmap)
    shifted_index = train_index[:] .+ length(test_index)
    X_shifted = make_rnn_tensor(X_full[shifted_index,:], 5 + 1)
    X_shifted = X_shifted[1:end-1, 2:end, :]
    log_σ_whole  = vec([nethat(xx) for xx in eachslice(X_shifted; dims =1 )][end]) 
    σ_hat_test = exp.(log_σ_whole)
    return σ_hat_test[end-(length(test_index)-1):end]
end


function naive_test_bnn_σ_prediction_recurrent(bnn,X_full::Array{Float32, 2},train_index,test_index, draws::Array{T, 2}) where {T}
    log_σ_whole = Array{T, 2}(undef, length(train_index)-5, size(draws, 2))
    train_index = train_index .- minimum(train_index) .+ 1
    
    shifted_index = train_index[:] .+ length(test_index)
    X_shifted = make_rnn_tensor(X_full[shifted_index,:], 5 + 1)
    X_shifted = X_shifted[1:end-1, 2:end, :]
    Threads.@threads for j=1:size(draws, 2)
        net = bnn.like.nc(draws[:, j])
        σh = vec([net(xx) for xx in eachslice(X_shifted; dims = 1)][end])
        log_σ_whole[:,j] = σh    
        end
    σhats = exp.(log_σ_whole[end-(length(test_index)-1):end,:])
    return σhats
end

 