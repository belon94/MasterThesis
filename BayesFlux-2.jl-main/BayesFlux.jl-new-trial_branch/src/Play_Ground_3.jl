
# Set the number of observations and seed for reproducibility
T = 1000
Random.seed!(123)

# Initialize the series
x1 = zeros(T)
x2 = zeros(T)
y = zeros(T)

# Generate the series
for t in 2:T
    x1[t] = 0.5*randn()   # Replace 0.5 with the mean of x1
    x2[t] = 0.5*randn()   # Replace 0.5 with the mean of x2
    y[t] = x1[t-1] + x2[t-1] + randn()  # VAR(1) equation
end


x = hcat(y,x1,x2)
# Get the dimensions of the matrix
x = x[2:end, :]
x = convert(Array{Float32, 2}, x)


net = Chain(RNN(2, 2), Dense(2, 1))
nc = destruct(net)
like = ArchSeqToOneNormal(nc, Normal(0, 0.5))
prior = GaussianPrior(nc, 0.5f0)
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)

x = make_rnn_tensor(x, 5 + 1)
y = vec(x[end, 1, :])
x = x[1:end-1, 2:3, :]

bnn = BNN(x, y, like, prior, init)
opt = FluxModeFinder(bnn, Flux.RMSProp())
θmap = find_mode(bnn, 10, 1000, opt)

nethat = nc(θmap)

log_σ  = vec([nethat(xx) for xx in eachslice(x; dims =1 )][end])
σ_hat = exp.(log_σ)

plot(1:length(σ_hat), y, label="Actual")
plot!(1:length(σ_hat),σ_hat, label="Estimated")

sampler = SGNHTS(1f-2, 1f0; xi = 1f0^1, μ = 1f0)

ch = mcmc(bnn, 10, 50_000, sampler)
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