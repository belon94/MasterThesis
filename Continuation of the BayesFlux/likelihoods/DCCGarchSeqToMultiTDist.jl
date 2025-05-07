### The likelihood for a DCC-GARCH model with a t-Student distribution


# ─────────────────────────────────────────────────────────────────────────────
#  DCC‑GARCH likelihood with multivariate Student‑t errors inside a Bayesian
#  neural‑network framework.
#
#  This version fixes the *predictive‑covariance scaling* issue: the scale
#  matrix passed to `MvTDist` is now Σ = H_{T+1} * (ν‑2)/ν so that
#      Cov[y_{T+1}] = H_{T+1}.
# ─────────────────────────────────────────────────────────────────────────────

using Distributions, LinearAlgebra, Random, StatsBase

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

"""
    transform_ab(a_raw, b_raw; ϵ = 1e-6)

Maps unconstrained reals (a_raw, b_raw) to (a, b) such that
    ϵ < a < 1‑2ϵ ,  ϵ < b < 1‑2ϵ ,  a + b < 1‑ϵ.
Guarantees covariance‑stationarity of the DCC recursion.
"""
function transform_ab(a_raw, b_raw; ϵ = 1e-10)
    σ(x) = 1 / (1 + exp(-x))
    a    = ϵ + (1 - 2ϵ) * σ(a_raw)
    braw = σ(b_raw)
    b    = (1 - a - ϵ) * braw
    return a, b
end

function cholesky(A; jitter = 1e-10)
    try
        return cholesky(A; check = true)
    catch
        δ = jitter * tr(A) / size(A,1)
        return cholesky(A + δ * I, check = false)
    end
end

# -----------------------------------------------------------------------------
# Likelihood type
# -----------------------------------------------------------------------------

struct DCCGarchTDist{T,F,D<:Distribution} <: BNNLikelihood
    num_params_like::Int          # 3: (a, b, ν)
    nc::NetConstructor{T,F}       # neural‑network constructor
    prior_μ::D                    # prior on the mean outputs
    N::Int                        # dimension of the series
end

function DCCGarchTDist(nc::NetConstructor{T,F}, prior_μ::D, N::Int) where {T,F,D<:Distribution}
    DCCGarchTDist{T,F,D}(3, nc, prior_μ, N)
end

# -----------------------------------------------------------------------------
# Log‑likelihood  ℓ(θ | data)
# -----------------------------------------------------------------------------

function (l::DCCGarchTDist{T,F,D})(x::Array{T,3},      
                                   y::Matrix{T},        
                                   θnet::AbstractVector,
                                   θlike::AbstractVector) where {T,F,D}
    # 1. Parameter transforms ------------------------------------------------------------
    θnet  = T.(θnet)
    θlike = T.(θlike)

    a, b  = transform_ab(θlike[1], θlike[2])
    ν     = 2 + exp(θlike[3])                # ν > 2 (finite variance)
    scale = sqrt((ν - 2)/ν)                 # t‑standardisation factor

    # 2. BNN forward pass ---------------------------------------------------------------
    net     = l.nc(θnet)
    N, Tobs = l.N, size(x,1)

    μ   = Matrix{T}(undef, N, Tobs)
    σ2  = similar(μ)
    z   = similar(μ)

    @inbounds for t in 1:Tobs
        out = net(@view x[t,:,:])
        length(out) == 2 * N || error("Network output must be 2N scalars.")
        μ[:,t]  .= @view out[1:N]
        σ2[:,t] .= exp.(@view out[N+1:2*N])
        z[:,t]  .= scale .* (y[:,t] .- μ[:,t]) ./ sqrt.(σ2[:,t])
    end

    # 3. Unconditional correlation ------------------------------------------------------
    Qbar = (z * z') / Tobs; cov2cor!(Qbar)  # diag == 1

    # 4. Initialise recursion -----------------------------------------------------------
    Q        = copy(Qbar)                   # Q₁
    logl     = zero(T)
    const_ll = lgamma((ν + N)/2) - lgamma(ν/2) - (N/2)*log(π*ν)

    diff = similar(y[:,1])                  

    # 5. DCC likelihood loop  (Q_t ⇒ ℓ_t, then Q_{t+1}) --------------------------------
    @inbounds for t in 1:Tobs
        # --- Likelihood step ---------------------------------------------------------
        d_inv   = 1 ./ sqrt.(diag(Q))
        R_t     = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
        H_t     = Diagonal(sqrt.(σ2[:,t])) * R_t * Diagonal(sqrt.(σ2[:,t]))

        chol    = cholesky(H_t)
        logdetH = 2sum(log, diag(chol.U))

        diff .= y[:,t] .- μ[:,t]
        quad    = dot(diff, chol \ diff)

        logl   += const_ll - 0.5*logdetH - ((ν + N)/2) * log1p(quad/ν)

        # --- Recursion step: build Q_{t+1} -----------------------------------------
        if t < Tobs
            Q_next = (1 - a - b) .* Qbar .+ a .* (z[:,t] * z[:,t]') .+ b .* Q
            Q      .= Q_next
        end
    end

    logl += sum(logpdf.(Ref(l.prior_μ), μ))
    return logl
end

# -----------------------------------------------------------------------------
# Posterior predictive draw  y_{T+1} | θ, data
# -----------------------------------------------------------------------------

function posterior_predict(l::DCCGarchTDist{T,F,D},
                           x::Array{T,3},
                           θnet::AbstractVector,
                           θlike::AbstractVector,
                           y_hist::Matrix{T}) where {T,F,D}
    θnet  = T.(θnet)
    θlike = T.(θlike)

    a, b  = transform_ab(θlike[1], θlike[2])
    ν     = 2 + exp(θlike[3])
    scale = sqrt((ν - 2)/ν)

    # BNN forward pass ---------------------------------------------------------------
    net     = l.nc(θnet)
    N, Tobs = l.N, size(x,1)

    μ   = Matrix{T}(undef, N, Tobs)
    σ2  = similar(μ)
    z   = similar(μ)

    @inbounds for t in 1:Tobs
        out = net(@view x[t,:,:])
        μ[:,t]  .= @view out[1:N]
        σ2[:,t] .= exp.(@view out[N+1:2*N])
        z[:,t]  .= scale .* (y_hist[:,t] .- μ[:,t]) ./ sqrt.(σ2[:,t])
    end

    # DCC recursion up to Q_T ----------------------------------------------------------
    Qbar = (z * z') / Tobs; cov2cor!(Qbar)
    Q    = copy(Qbar)

    for t in 1:Tobs-1
        Q_next = (1 - a - b) .* Qbar .+ a .* (z[:,t] * z[:,t]') .+ b .* Q
        Q      .= Q_next
    end

    # Build Q_{T+1}
    Q = (1 - a - b) .* Qbar .+ a .* (z[:,Tobs] * z[:,Tobs]') .+ b .* Q

    # Predictive covariance ------------------------------------------------------------
    d_inv = 1 ./ sqrt.(diag(Q))
    R_T1  = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
    H_T1  = Diagonal(sqrt.(σ2[:,end])) * R_T1 * Diagonal(sqrt.(σ2[:,end]))

    # ---- SCALE CORRECTION -----------------------------------------------------------
    Σ = H_T1 * (ν - 2) / ν            # ensures Cov = H_T1

    μ_T1 = μ[:,end]
    return rand(MvTDist(ν, μ_T1, Σ))
end


export DCCGarchTDist, posterior_predict, transform_ab

