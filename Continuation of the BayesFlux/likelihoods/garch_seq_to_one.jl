using Distributions
using LinearAlgebra

###############################################################################
# Sequence to one Normal Multiple Output
################################################################################
#Time varying μ and σ
######################################################################
"""
    GArchSeqToOneNormal(nc::NetConstructor{T, F}, prior_μ::D) where {T, F, D<:Distribution}

Use a Gaussian/Normal likelihood for a Seq-to-One architecture with a multiple output.

Assumes is two outputs. Thus, the last layer must have output size two. 

# Argumentsß
ßß
- `nc`: obtained using [`destruct`](@ref)
"""

struct GarchSeqToOneNormal{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    #prior for the distribution paramaters is not used, but kept it for consistency of the package
    prior_μ::D
end

function GarchSeqToOneNormal(nc::NetConstructor{T,F}, prior_μ::D) where {T,F,D<:Distributions.Distribution}
    return GarchSeqToOneNormal(1, nc, prior_μ)
end


function (l::GarchSeqToOneNormal{T,F,D})(x::Array{T,3}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    garch_params = [net(xx) for xx in eachslice(x; dims=1)][end]
    μ = garch_params[1, :]
    log_σ = garch_params[2, :]
    n = length(y)
    σ = exp.(log_σ)
    σ = T.(σ)
    # Using reparameterised likelihood 
    # Usually results in faster gradients
    return logpdf(MvNormal(zeros(n), I), (y .- μ) ./ σ) - sum(log.(σ)) 
end

function posterior_predict(l::GarchSeqToOneNormal{T,F,D}, x::Array{T,3}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}

    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    garch_params = [net(xx) for xx in eachslice(x; dims=1)][end]
    μ = garch_params[1, :]
    log_σ = garch_params[2, :]
    σ = exp.(log_σ)
    σ = T.(σ)

    ypp = rand(MvNormal(μ, σ^2 * I))
    return ypp
end


################################################################################
# Sequence to one TDIST
################################################################################

"""
GarchSeqToOneTDist(nc::NetConstructor{T, F}, prior_μ::D, ν::T) where {T, F, D}

Use a Student-T likelihood for a Seq-to-One architecture with a multiple(two) outputs
and known degress of freedom.

Assumes that multiple(two) outputs. Thus, the last layer must have output size of two. 

# Arguments

- `nc`: obtained using [`destruct`](@ref)
- `prior_μ`: a prior distribution for the standard deviation
- `ν`: degrees of freedom

Note that #prior for the distribution paramaters is not used, but kept it for consistency of the package

"""
struct GarchSeqToOneTDist{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior_μ::D
    ν::T
end
function GarchSeqToOneTDist(nc::NetConstructor{T,F}, prior_μ::D, ν::T) where {T,F,D}
    return GarchSeqToOneTDist(1, nc, prior_μ, ν)
end

function (l::GarchSeqToOneTDist{T,F,D})(x::Array{T,3}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    garch_params = [net(xx) for xx in eachslice(x; dims=1)][end]
    μ = garch_params[1, :]
    log_σ = garch_params[2, :]
    σ = exp.(log_σ)
    σ = T.(σ)

    return sum(logpdf.(TDist(l.ν), (y .- μ) ./ σ)) - sum(log.(σ))
end


function posterior_predict(l::GarchSeqToOneTDist{T,F,D}, x::Array{T,3}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)


    net = l.nc(θnet)
    garch_params = [net(xx) for xx in eachslice(x; dims=1)][end]
    μ = garch_params[1, :]
    log_σ = garch_params[2, :]
    σ = exp.(log_σ)
    σ = T.(σ)
    n = length(σ)

    ypp = σ .* rand(TDist(l.ν), n) .+ μ
    return ypp
end



