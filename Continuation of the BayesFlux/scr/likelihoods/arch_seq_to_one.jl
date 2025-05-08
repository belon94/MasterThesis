using Distributions
using LinearAlgebra

###############################################################################
# Sequence to one Normal 
################################################################################
#Time varying σ and constant μ

"""
    ArchSeqToOneNormal(nc::NetConstructor{T, F}, prior_μ::D) where {T, F, D<:Distribution}

Use a Gaussian/Normal likelihood for a Seq-to-One architecture with a single output.

Assumes is a single output. Thus, the last layer must have output size one. 

# Argumentsß
ßß
- `nc`: obtained using [`destruct`](@ref)
- `prior_μ`: a prior distribution for the standard deviation

"""

struct ArchSeqToOneNormal{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
# TODO: rename the prior(μ) 
    prior_μ::D
end

function ArchSeqToOneNormal(nc::NetConstructor{T,F}, prior_μ::D) where {T,F,D<:Distributions.Distribution}
    return ArchSeqToOneNormal(1, nc, prior_μ)
end


function (l::ArchSeqToOneNormal{T,F,D})(x::Array{T,3}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    log_σ = vec([net(xx) for xx in eachslice(x; dims=1)][end])
    n = length(y)
    σ = exp.(log_σ)
    σ = T.(σ)
    # Using reparameterised likelihood 
    # Usually results in faster gradients
    return logpdf(MvNormal(zeros(n), I), (y .- θlike[1]) ./ σ) - sum(log.(σ)) + logpdf(l.prior_μ, θlike[1])
end

function posterior_predict(l::ArchSeqToOneNormal{T,F,D}, x::Array{T,3}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}

    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    log_σ = vec([net(xx) for xx in eachslice(x; dims=1)][end])
    σ = exp.(log_σ)
    σ = T.(σ)

    ypp = rand(MvNormal(θlike[1], σ^2 * I))
    return ypp
end


################################################################################
# Sequence to one TDIST
################################################################################

"""
ArchSeqToOneTDist(nc::NetConstructor{T, F}, prior_μ::D, ν::T) where {T, F, D}

Use a Student-T likelihood for a Seq-to-One architecture with a single output
and known degress of freedom.

Assumes is a single output. Thus, the last layer must have output size one. 

# Arguments

- `nc`: obtained using [`destruct`](@ref)
- `prior_μ`: a prior distribution for the standard deviation
- `ν`: degrees of freedom

"""
struct ArchSeqToOneTDist{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int
    nc::NetConstructor{T,F}
    prior_μ::D
    ν::T
end
function ArchSeqToOneTDist(nc::NetConstructor{T,F}, prior_μ::D, ν::T) where {T,F,D}
    return ArchSeqToOneTDist(1, nc, prior_μ, ν)
end

function (l::ArchSeqToOneTDist{T,F,D})(x::Array{T,3}, y::Vector{T}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)

    net = l.nc(θnet)
    log_σ = vec([net(xx) for xx in eachslice(x; dims=1)][end])
    σ = exp.(log_σ)
    σ = T.(σ)

    return sum(logpdf.(TDist(l.ν), (y .- θlike[1]) ./ σ)) - sum(log.(σ)) + logpdf(l.prior_μ, θlike[1])
end


function posterior_predict(l::ArchSeqToOneTDist{T,F,D}, x::Array{T,3}, θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)


    net = l.nc(θnet)
    log_σ = vec([net(xx) for xx in eachslice(x; dims=1)][end])
    σ = exp.(log_σ)
    σ = T.(σ)
    n = length(σ)

    ypp = σ .* rand(TDist(l.ν), n) .+ θlike[1]
    return ypp
end



