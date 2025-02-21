using CSV, DataFrames, ShiftedArrays
using Dates,Statistics
using JLD2

# Load the data
data = CSV.read("src/data/SPY_data.csv", DataFrame)

# Compute the log returns
data[!, "Log Return"] = log.(data[!, "Adj Close"]) .- log.(circshift(data[!, "Adj Close"], 1))

# Delete the first row
delete!(data, 1)

# Replace Inf and -Inf with missing values
data[!, "Log Return"] = replace(data[!, "Log Return"], Inf=>missing, -Inf=>missing)

# Drop the missing values
dropmissing!(data)
df = rename(data, :"Adj Close" => :"Price")

# Convert DataFrame to Matrix
matrix_data = Matrix(data)
matrix_data = Matrix{Float32}(matrix_data[:,2:3])

# Create portfolio matrix
returns = matrix_data[:,2]
r_squared = returns.^2
X = hcat(returns, r_squared)

#select the time interval
# train_index = 1:924
# test_index = 925:1155
# full_index = 1:1155
train_index = 600:1000
test_index = 1001:1100
full_index = minimum(train_index):maximum(test_index)
df = df[full_index,:]
# Delete the first 5 rows
df = df[6:end,:]
# Add new columns filled with constant values
df[!, :MAP] = fill(Float64(1), size(df, 1))
df[!, :Mean] = fill(Float64(2), size(df, 1))
df[!, :Median] = fill(Float64(3), size(df, 1))

df_train_index = (train_index .- minimum(train_index) .+ 1)[1:end-5]
df_test_index = test_index .- minimum(test_index) .+ 1 .+ maximum(df_train_index)

#multiply the return to avoid numerical problems !!!!!!!!! do not forget !!!!!!

###Find a way to adjust here properly 
#X = matrix_data[:,2:end] .* 100
y = returns 
X = hcat(y,X)

X_train, y_train, X_test, y_test = X[train_index,:], y[train_index], X[test_index,:], y[test_index]
X_full, y_full = X[full_index,:], y[full_index]
#arrange a train and test set properly
X_train
X_train = make_rnn_tensor(X_train, 5 + 1)
y_train = vec(X_train[end, 1, :])
X_train = X_train[1:end-1, 2:end, :]

degrees_f = Float32(5)  # Degrees of freedom, replace ... with the actual value.
quantiles = Float32.([0.01,0.05,0.1])   # The value to find the quantile for, replace ... with the actual value.

# Creating t-distribution with given degrees of freedom
t_dist = TDist(degrees_f)

# Calculating the quantile
quant = quantile(t_dist, quantiles)

#Get Bnn
bnn = get_bnn(X_train,(y_train),degrees_f)

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
# insert new values into rows 10:19 of column :new_column
df[df_train_index, :"MAP"] = convert(Vector{Float64}, σ_hat)


#Training-set plot
plot(1:length(y_train), y_train, label="Actual")
plot!(1:length(y_train),σ_hat, label="Estimated")

#Test-set MAP estimation
σ_hat_test = estimate_test_σ(bnn, train_index, test_index, θmap, X_full)
VaRs_test_MAP = bnn_var_prediction(σ_hat_test,θmap,quant)
df[df_test_index, :"MAP"] = convert(Vector{Float64}, σ_hat_test)

#Test-set plot
plot(1:length(y_test), y_test, label="Actual")
plot!(1:length(y_test),σ_hat_test, label="Estimated")

####-----------------

#######   BNN Estimation   ######

#training-set BNN 

#sampler
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)

#sampling 
ch = mcmc(bnn, 10, 50_0, sampler,θstart = θmap)
ch = ch[:, end-20_0+1:end]
chain = Chains(ch')

#training-set BNN mean/median VaRs estimation
σhats = naive_train_bnn_σ_prediction_recurrent(bnn,ch)
VaRs_bnn = bnn_var_prediction(σhats,ch,quant)
df[df_train_index, :"Mean"] = convert(Vector{Float64}, vec(mapslices(mean, σhats, dims = 2)))
df[df_train_index, :"Median"] = convert(Vector{Float64}, vec(mapslices(median, σhats, dims = 2)))



#Test set estimation -computationaly expensive
σhats_test = naive_test_bnn_σ_prediction_recurrent(bnn,X_full,train_index,test_index,ch)
VaRs_test_bnn = bnn_var_prediction(σhats_test,ch,quant)
df[df_test_index, :"Mean"] = convert(Vector{Float64}, vec(mapslices(mean, σhats_test, dims = 2)))
df[df_test_index, :"Median"] = convert(Vector{Float64}, vec(mapslices(mean, σhats_test, dims = 2)))
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






# Define volatility estimation methods
vol_estimation_methods = [:MAP, :Mean, :Median]

# Perform the backtests
initial_investment = 100000.0

for vol_estimation_method in vol_estimation_methods
    portfolio_directional = backtest_strategy(df[df_test_index,:], initial_investment, directional_strategy, vol_estimation_method)
    portfolio_mean_reversion = backtest_strategy(df[df_test_index,:], initial_investment, mean_reversion_strategy, vol_estimation_method)

    # Calculate the final portfolio values
    final_value_directional = portfolio_directional.cash + portfolio_directional.etf_units * df[maximum(df_test_index), :Price]
    final_value_mean_reversion = portfolio_mean_reversion.cash + portfolio_mean_reversion.etf_units * df[maximum(df_test_index), :Price]

    # Print the final portfolio values
    println("Final portfolio value for the directional strategy with ", vol_estimation_method, ": ", final_value_directional)
    println("Final portfolio value for the mean-reversion strategy with ", vol_estimation_method, ": ", final_value_mean_reversion)
end

etf_units_hold = initial_investment / df[minimum(df_test_index), :Price]  # Buy as much as we can at the beginning

# Update the value of the holding strategy
portfolio_value_hold = etf_units_hold *  df[maximum(df_test_index), :Price]







    portfolio_directional = backtest_strategy(df[df_train_index,:], initial_investment, directional_strategy, :MAP)
    portfolio_mean_reversion = backtest_strategy(df[df_train_index,:], initial_investment, mean_reversion_strategy, :MAP)

    # Calculate the final portfolio values
    final_value_directional = portfolio_directional.cash + portfolio_directional.etf_units * df[maximum(df_train_index), :Price]
    final_value_mean_reversion = portfolio_mean_reversion.cash + portfolio_mean_reversion.etf_units * df[maximum(df_train_index), :Price]

    # Print the final portfolio values
    println("Final portfolio value for the directional strategy with ", vol_estimation_method, ": ", final_value_directional)
    println("Final portfolio value for the mean-reversion strategy with ", vol_estimation_method, ": ", final_value_mean_reversion)


c = 10/3 


floor.(Int, c) 