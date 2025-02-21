include("BayesFlux.jl")
using  Flux
using  Random, Distributions,  LinearAlgebra, Plots
using  MCMCChains, Bijectors
using  .BayesFlux#Example: Feedforward NN Regression
Random.seed!(1212)


"Simulate GARCH process."
using Distributions



k = 5
n = 500
x = randn(Float32, k, n);
β = randn(Float32, k);
y = x'*β + randn(Float32, n);

net = Chain(Dense(k, k, relu), Dense(k, k, relu), Dense(k, 1))

nc = destruct(net)

like = FeedforwardNormal(nc, Gamma(2.0, 0.5))

prior = GaussianPrior(nc, 0.5f0)

init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)

bnn = BNN(x, y, like, prior, init)

opt = FluxModeFinder(bnn, Flux.ADAM())  # We will use ADAM
θmap = find_mode(bnn, 10, 500, opt)  # batchsize 10 with 500 epochs

nethat = nc(θmap)
yhat = vec(nethat(x))
sqrt(mean(abs2, y .- yhat))

+ logpdf(tdist, θlike[1])
tdist = transformed(Gamma(2.0, 0.5))
sigma = invlink(Gamma(2.0, 0.5), θlike[1])
##MCMC - SGLD

sampler = SGNHTS(1f-2, 2f0; xi = 2f0^2, μ = 50f0)
ch = mcmc(bnn, 10, 50_000, sampler)
ch = ch[:, end-20_000+1:end]

chain = Chains(ch')
plot(chain)

function naive_prediction(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    yhats = Array{T, 2}(undef, length(y), size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        net = bnn.like.nc(draws[:, i])
        yh = vec(net(x))
        yhats[:,i] = yh
    end
    return yhats
end

yhats_mcmc = naive_prediction(bnn, ch)
chain_yhat = Chains(yhats_mcmc')
maximum(summarystats(chain_yhat)[:, :rhat])


function get_observed_quantiles(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    qs = [quantile(yr, target_q) for yr in eachrow(posterior_yhat)]
    qs = reduce(hcat, qs)
    observed_q = mean(reshape(y, 1, :) .< qs; dims = 2)
    return observed_q
end

posterior_yhat = sample_posterior_predict(bnn, ch)
t_q = 0.05:0.05:0.95
o_q = get_observed_quantiles(y, posterior_yhat, t_q)
plot(t_q, o_q, label = "Posterior Predictive", legend=:topleft,
    xlab = "Target Quantile", ylab = "Observed Quantile")
plot!(x->x, t_q, label = "Target")





##########check which funtion it uses##########
isa(net, Flux.Dense)

typeof(net)

@which FeedforwardNormal(nc, Gamma(2.0, 0.5))

@edit  FeedforwardNormal(nc, Gamma(2.0, 0.5))

methods(FeedforwardNormal)

like.num_params_like


###############################################
#Prior 

isa(net, Flux.Dense)

typeof(prior)

@which GaussianPrior(nc, 0.5f0)

@edit  GaussianPrior(nc, 0.5f0)

methods(GaussianPrior)

prior.num_params_hyper

#Initialization

@which InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)

@edit  InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)

methods(InitialiseAllSame)

###BNN

@which BNN(x, y, like, prior, init)

@edit BNN(x, y, like, prior, init)

#opt

@which FluxModeFinder(bnn, Flux.ADAM())

@edit FluxModeFinder(bnn, Flux.ADAM())

#MAP
@edit find_mode(bnn, 10, 500, opt)


@edit nethat(x)

nethat(x)


@edit destruct(net)

@edit nc(θmap)

bnn.like()

k = Int(3.0)















k = 5
n = 500
x = randn(Float32, k, n);
β = randn(Float32, k);
y = x'*β + randn(Float32, n);


net = Chain(Dense(k, k),Dense(k,1))  # k inputs and one output

nc = destruct(net)

θ = randn(Float32, nc.num_params_network)

###we will define a prior for all parameters of the network. Since weight decay is a popular regularisation method in standard ML estimation, we will be using a Gaussian prior, which is the Bayesian weight decay:
prior = GaussianPrior(nc, 0.5f0)  # the last value is the standard deviation

##We also need a likelihood and a prior on all parameters the likelihood introduces to the model. We will go for a Gaussian likelihood, which introduces the standard deviation of the model. BFlux currently implements Gaussian and Student-t likelihoods for Feedforward and Seq-to-one cases but more can easily be implemented.

like = FeedforwardNormal(nc, Gamma(2.0, 0.5))  # Second argument is prior for standard deviation.

init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)  # First argument is dist we draw parameters from.

bnn = BNN(x, y, like, prior, init)


opt = FluxModeFinder(bnn, Flux.ADAM())  # We will use ADAM
θmap = find_mode(bnn, 10, 500, opt)  # batchsize 10 with 500 epochs


nethat = nc(θmap)
yhat = vec(nethat(x))
sqrt(mean(abs2, y .- yhat))

arr1 = collect(1:10)
arr2 = collect(2:11)


sqrt(mean(abs2, y .- yhat))

arr2 - arr1 

mean(abs2, arr2 - arr1)

y .- yhat

y










####Plot Distributions


x = range(-10, stop=10, length=1000)
y = pdf.(Bijectors.log{0}())

plot(x, y, xlabel="x", ylabel="Density", title="Normal Distribution")



tdist = transformed(Gamma(2.0, 0.5))
sigma = invlink(Gamma(2.0, 0.5), θlike[1])
##MCMC - SGLD


min 49