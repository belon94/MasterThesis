include("BayesFlux.jl")
using  Flux
using  Random, Distributions,  LinearAlgebra, Plots
using  MCMCChains, Bijectors
using  .BayesFlux#Example: Feedforward NN Regression
Random.seed!(1212)


# Number of simulations
#nr.sim = 100
# Sample size
n = 500
# Volatility
vol = (20^2)/252 #Implying a volatility of 20% per year
# Degrees of freedom
#df = 8
# Right tail Var and ES probability
#p = 0.01
# GARCH parameters
α = 0.5
β = 0.3




#"Function to Simulate GARCH process."
function garchNDraws(n::Int, α::Float64, β::Float64, vol::Float64)
    ω = vol*(1-α-β)
    d = TDist(5)  # Create a t-distribution with 10 degrees of freedom
    ϵ = rand(d,n)
    L = similar(ϵ)
    L[1] = sqrt(vol) * ϵ[1]
    σ_2 = zeros(n+1)
    σ_2[1] = vol

    # Generate series
    for j in 2:n
        σ_2[j] = ω + α * L[j-1]^2 + β * σ_2[j-1]
        L[j] = sqrt(σ_2[j]) * ϵ[j]
    end

    # Get one period ahead value of sigma (not that randomness is involved ϵ vector)
    σ_2[n+1] = ω + α * L[n]^2 + β * σ_2[n]

    # Return output
    output = Dict("L" => L, "sigma_squared" => σ_2[1:n], "nextSigma" => σ_2[n+1])
    return output
end

simulated_Garch = garchNDraws(n, α, β, vol)

y = Float32.(simulated_Garch["L"])
simulated_σ = sqrt.(Float32.(simulated_Garch["sigma_squared"]))

#y = y .^2
#select the time interval
# train_index = 1:924
# test_index = 925:1155
# full_index = 1:1155
train_index = 1:400
test_index = 401:500
full_index = 1:500



y_train, y_test = y[train_index], y[test_index]
y_full = y[full_index]
#arrange a train and test set properly
x_train = make_rnn_tensor(reshape(y_train, :, 1), 5 + 1)
y_train = vec(x_train[end, :, :])
x_train = x_train[1:end-1, :, :]
#arrange a train and test set properly

####### New likelihood Trial


net = Chain(RNN(1, 2), Dense(2, 2))  # last layer is linear output layer
nc = destruct(net)
like =GarchSeqToOneTDist(nc, Normal(0, 0.5),Float32(5))
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)





bnn = BNN(x_train, y_train, like, prior, init)
opt = FluxModeFinder(bnn, Flux.RMSProp())
θmap = find_mode(bnn, 10, 5000, opt)

nethat = nc(θmap)
parameters = [nethat(xx) for xx in eachslice(x_train; dims =1 )][end]
μ = parameters[1, :]
log_σ = parameters[2, :]
σ_hat = exp.(log_σ)


sqrt(mean(abs2, simulated_σ[6:n] .- σ_hat))



σ_hat_testt = estimate_test_σ(bnn, train_index, test_index, θmap, y_full)
σ_hat_testt
nethat = bnn.like.nc(θmap)
shifted_index = train_index[:] .+ length(test_index)
x_shifted = make_rnn_tensor(reshape(y_full[shifted_index], :, 1), 5 + 1)
x_shifted = x_shifted[1:end-1, :, :]
log_σ_whole  = vec([nethat(xx) for xx in eachslice(x_shifted; dims =1 )][end]) 
σ_hat_whole = exp.(log_σ_whole)

σ_hat_test

length(test_index)
σ_hat_whole[end-99:end]

count(x -> x, σ_hat_testt .!= σ_hat_whole[end-99:end])

##
last_sequence = x[end, 1:end, :]  # Taking the last sequence from your training data

k = nethat(last_sequence)[end]
k[end]

# plot the actual and estimated series
plot(1:length(σ_hat), simulated_σ[6:400], label="Actual")
plot!(1:length(σ_hat),σ_hat, label="Estimated")

# add labels and legend
xlabel!("Time")
ylabel!("Std.")
title!("Actual vs. Estimated Series")

var = θmap[end] .+ (σ_hat .* 1.645)
var = Float32.(var)
es = θmap[end] .+ (σ_hat .* 2.062713)
es =Float32.(es)
y

# Compare element-wise
comparison_var = y .> var
# Count how many times elements in x are greater than those in y
count_bigger_var = sum(comparison_var)
persantage = count_bigger_var/ length(y)


###Samplerss##
sampler = AdaptiveMH(diagm(ones(Float32, bnn.num_total_params)), 1000, 0.5f0, 1f-4)

sadapter = DualAveragingStepSize(1f-9; target_accept = 0.55f0, adapt_steps = 10000)
madapter = DiagCovMassAdapter(5000, 1000)
sampler = HMC(1f-9, 5; sadapter = sadapter)

sadapter = DualAveragingStepSize(1f-9; target_accept = 0.55f0, adapt_steps = 10000)
sampler = GGMC(Float32; β = 0.1f0, l = 1f-9, sadapter = sadapter)

sampler = SGLD(Float32; stepsize_a = 10f-2, stepsize_b = 0.01f0, stepsize_γ = 0.55f0)
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)


ch = mcmc(bnn, 10, 50_000, SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0))
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')



function naive_prediction_recurrent(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    yhats = Array{T, 2}(undef, length(y), size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        yh = vec([net(xx) for xx in eachslice(x; dims = 1)][end])
        yhats[:,i] = yh
    end
    return yhats
end

function get_observed_quantiles(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    qs = [quantile(yr, target_q) for yr in eachrow(posterior_yhat)]
    qs = reduce(hcat, qs)
    observed_q = mean(reshape(y, 1, :) .< qs; dims = 2)
    return observed_q
end


log_σhats = naive_prediction_recurrent(bnn, ch)
σhats = exp.(log_σhats)
chain_σhat = Chains(σhats')
maximum(summarystats(chain_σhat)[:, :rhat])

posterior_log_σhat = sample_posterior_predict(bnn, ch)
posterior_σhat = exp.(posterior_log_σhat)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y_train, posterior_log_σhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")



##Variation Inference

q, params, losses = bbb(bnn, 10, 2_000; mc_samples = 1, opt = Flux.ADAM(), n_samples_convergence = 10)
ch = rand(q, 20_000)
posterior_log_σhat = sample_posterior_predict(bnn, ch)
posterior_σhat = exp.(posterior_log_σhat)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(simulated_σ[6:500], posterior_σhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")




# #### Default Recurrent estimation

# Random.seed!(6150533)
# gamma = 0.8
# N = 500
# burnin = 1000
# y = zeros(N + burnin + 1)
# for t=2:(N+burnin+1)
#     y[t] = gamma*y[t-1] + randn()
# end
# y = Float32.(y[end-N+1:end])


# net = Chain(RNN(1, 1), Dense(1, 1))  # last layer is linear output layer
# nc = destruct(net)
# like = SeqToOneNormal(nc, Gamma(2.0, 0.5))
# prior = GaussianPrior(nc, 0.5f0)
# init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)


# x = make_rnn_tensor(reshape(y, :, 1), 5 + 1)
# y = vec(x[end, :, :])
# x = x[1:end-1, :, :]



# bnn = BNN(x, y, like, prior, init)
# opt = FluxModeFinder(bnn, Flux.RMSProp())
# θmap = find_mode(bnn, 10, 1000, opt)


# nethat = nc(θmap)
# yhat = vec([nethat(xx) for xx in eachslice(x; dims =1 )][end])
# sqrt(mean(abs2, y .- yhat))

ch

chain

size(ch, 2)


# Define a matrix of arrays
A = [randn(5) for i in 1:3]
A
# Concatenate the arrays horizontally using reduce and hcat
B = reduce(hcat, A)

# Print the original matrix and the concatenated matrix
println("A: ", A)
println("B: ", B)

qs = [quantile(yr, t_q) for yr in eachrow(posterior_yhat)]
qs = reduce(hcat, qs)


####### Assuming we have a function posterior_density(θ) that returns the posterior 
#density p(θ|y) for a given value of θ, we can calculate 
#the predictive distribution p(z|y) for a specific value of z as follows:

function predictive_distribution(posterior_density, z, sampling_distribution, lower_bound, upper_bound)
    integrand(θ) = pdf(sampling_distribution(θ), z) * posterior_density(θ)
    p_z_given_y, _ = quadgk(integrand, lower_bound, upper_bound)
    return p_z_given_y
end


#Here, sampling_distribution is a function that takes θ as input and returns the appropriate 
#distribution for p(z|θ). lower_bound and upper_bound are the limits of integration for the
# numerical integration. You may need to adjust them depending on the range of θ in your 
#problem.

function posterior_density(θ)
    # Replace this with your actual posterior density function
    return pdf(Normal(0, 1), θ)
end

sampling_distribution(θ) = Normal(θ, 1)

#To compute the predictive distribution p(z|y) for a specific value of z,
# call the predictive_distribution function:

z = 1.5
lower_bound = -10
upper_bound = 10

p_z_given_y = predictive_distribution(posterior_density, z, sampling_distribution, lower_bound, upper_bound)

#Now p_z_given_y contains the predictive distribution value p(z|y) for the given value of z.
# Adjust the posterior_density function, the sampling distribution, and the limits of 
#integration according to your problem.





#training-set BNN 

#sampler
sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)

#sampling 
ch = mcmc(bnn, 10, 50_000, sampler,θstart = θmap)
ch = ch[:, end-20_000+1:end]
chain = Chains(ch')

#training-set BNN mean/median VaRs estimation
σhats = naive_train_bnn_σ_prediction_recurrent(bnn,ch)


#Test set estimation -computationaly expensive
σhats_test = naive_test_bnn_σ_prediction_recurrent(bnn,y_full,train_index,test_index,ch)
VaRs_test_bnn = bnn_var_prediction(σhats_test,ch,quant)


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
bnn_trial = naive_test_bnn_σ_prediction_recurrent(bnn,y_full,train_index,test_index,ch)