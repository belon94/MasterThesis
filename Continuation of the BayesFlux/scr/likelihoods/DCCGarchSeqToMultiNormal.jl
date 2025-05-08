# DCC-GARCH Normal Likelihood for Bayesian Neural Networks
# This code implements a DCC-GARCH model with normal errors for Bayesian neural networks.   
# It includes functions for transforming parameters, computing the log-likelihood,
using LinearAlgebra, Statistics, Distributions

struct DCCGarchNormal{T,F,D<:Distributions.Distribution} <: BNNLikelihood
    num_params_like::Int  # For DCC parameters (a,b)
    nc::NetConstructor{T,F}
    prior_μ::D
    N::Int  # Dimension of multivariate output
end

function DCCGarchNormal(nc::NetConstructor{T,F}, prior_μ::D, N::Int) where {T,F,D<:Distributions.Distribution}
    return DCCGarchNormal(2, nc, prior_μ, N)  # 2 params: a and b
end

# Helper function for nearest positive definite matrix
function nearest_pd(A)
    # Compute symmetric part
    B = (A + A') / 2
    
    # Eigendecomposition
    F = eigen(B)
    
    # Replace negative eigenvalues with small positive values
    D = Diagonal(max.(F.values, 1e-10))
    
    # Reconstruct matrix
    return F.vectors * D * F.vectors'
end

function (l::DCCGarchNormal{T,F,D})(x::Array{T,3}, y::Matrix{T}, 
                                   θnet::AbstractVector, θlike::AbstractVector) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)
    
    # Transform DCC parameters to valid range
    a, b = transform_ab(θlike[1], θlike[2])
    
    # Initialize network
    net = l.nc(θnet)
    
    # Dimensions
    N, Tsteps = l.N, size(x, 1)
    
    # Initialize arrays for means, standard deviations and standardized residuals
    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)  # standard deviation instead of variance
    z = similar(μ)
    
    # Extract parameters and compute standardized residuals for each time step
    @inbounds for t in 1:Tsteps
        # Use view for efficient slicing of x
        xt_view = @view x[t, :, :]
        # Get network output
        out = net(xt_view)     
        length(out) == 2N || error("Network output must be 2N scalars.")
        
        # Fixed: don't use @view on output vector, use indexing instead
        μ[:, t] .= out[1:N]       # Means
        σ[:, t] .= exp.(out[N+1:2N] ./ 2)  # Standard deviations (exp of half-log-variance)
        z[:, t] .= (y[:, t] .- μ[:, t]) ./ σ[:, t]  # Standardized residuals
    end
    
    # Compute unconditional correlation matrix
    Qbar = Tsteps > 1 ? (z * z') / Tsteps : I(N)
    
    # Ensure Qbar is correlation matrix
    if Tsteps > 1
        d_inv = 1 ./ sqrt.(diag(Qbar))
        Qbar = Symmetric(Diagonal(d_inv) * Qbar * Diagonal(d_inv))
    end
    
    # Initialize Q1 as unconditional correlation
    Q = copy(Qbar)
    
    # Initialize log-likelihood
    logl = zero(T)
    
    # Process first time step separately
    t = 1
    
    # Compute correlation matrix R_t from Q
    d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
    R = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
    
    # Construct covariance matrix H_t
    Dt = Diagonal(σ[:, t])  # Using standard deviation directly
    H_t = Dt * R * Dt
    
    # Always ensure H_t is positive definite - no try/catch for Zygote compatibility
    H_t = nearest_pd(H_t)
    Hchol = cholesky(H_t)
    
    # Compute log-likelihood contribution
    diff = y[:, t] .- μ[:, t]
    quad = sum(abs2, Hchol.L \ diff)
    
    logl -= 0.5 * (N * log(2π) + 2*sum(log, diag(Hchol.L)) + quad)
    
    # Update Q for second time step using first observation
    if Tsteps > 1
        Q_next = (1-a-b).*Qbar .+ a.*(z[:,1]*z[:,1]') .+ b.*Q
        Q = Q_next
    end
    
    # DCC recursion for remaining time steps
    for t in 2:Tsteps
        # Compute correlation matrix R_t from Q_t (which uses information up to t-1)
        d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
        R = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
        
        # Construct covariance matrix H_t
        Dt = Diagonal(σ[:, t])  # Using standard deviation directly
        H_t = Dt * R * Dt
        
        # Always ensure H_t is positive definite - no try/catch for Zygote compatibility
        H_t = nearest_pd(H_t)
        Hchol = cholesky(H_t)
        
        # Compute log-likelihood contribution
        diff = y[:, t] .- μ[:, t]
        quad = sum(abs2, Hchol.L \ diff)
        
        logl -= 0.5 * (N * log(2π) + 2*sum(log, diag(Hchol.L)) + quad)
        
        # Update Q for next time step using current observation
        if t < Tsteps
            Q_next = (1-a-b).*Qbar .+ a.*(z[:,t]*z[:,t]') .+ b.*Q
            Q = Q_next
        end
    end
    
    # Add prior contribution
    logl += sum(logpdf.(l.prior_μ, μ))
    
    return logl
end

function posterior_predict(l::DCCGarchNormal{T,F,D}, 
                          x::Array{T,3}, 
                          θnet::AbstractVector, 
                          θlike::AbstractVector,
                          y_hist::Matrix{T}) where {T,F,D}
    θnet = T.(θnet)
    θlike = T.(θlike)
    
    # Transform parameters to valid range
    a, b = transform_ab(θlike[1], θlike[2])
    
    # Initialize network
    net = l.nc(θnet)
    
    # Dimensions
    N, Tsteps = l.N, size(x, 1)
    
    # Process historical data for DCC recursion
    μ = Matrix{T}(undef, N, Tsteps)
    σ = similar(μ)  # standard deviation instead of variance
    z = similar(μ)
    
    @inbounds for t in 1:Tsteps
        # Use view for efficient slicing of x
        xt_view = @view x[t, :, :]
        # Get network output
        out = net(xt_view)
        length(out) == 2N || error("Network output must be 2N scalars.")
        
        # Fixed: don't use @view on output vector, use indexing instead
        μ[:, t] .= out[1:N]
        σ[:, t] .= exp.(out[N+1:2N] ./ 2)  # Standard deviations (exp of half-log-variance)
        z[:, t] .= (y_hist[:, t] .- μ[:, t]) ./ σ[:, t]  # Standardized residuals
    end
    
    # Compute unconditional correlation matrix
    Qbar = Tsteps > 1 ? (z * z') / Tsteps : I(N)
    
    # Ensure Qbar is correlation matrix
    if Tsteps > 1
        d_inv = 1 ./ sqrt.(diag(Qbar))
        Qbar = Symmetric(Diagonal(d_inv) * Qbar * Diagonal(d_inv))
    end
    
    # Initialize Q1 as unconditional correlation
    Q = copy(Qbar)
    
    # Proper DCC recursion with temporal ordering
    if Tsteps > 1
        # Process observations sequentially
        for t in 1:(Tsteps-1)
            Q_next = (1-a-b).*Qbar .+ a.*(z[:,t]*z[:,t]') .+ b.*Q
            Q = Q_next
        end
        
        # Final update with last observation
        Q = (1-a-b).*Qbar .+ a.*(z[:,Tsteps]*z[:,Tsteps]') .+ b.*Q
    end
    
    # Compute correlation matrix R_t
    d_inv = 1 ./ sqrt.(max.(diag(Q), 1e-10))
    R = Symmetric(Diagonal(d_inv) * Q * Diagonal(d_inv))
    
    # Means and variance for prediction (from last time step)
    μp = μ[:, end]
    Hp = Diagonal(σ[:, end]) * R * Diagonal(σ[:, end])  # Using standard deviation directly
    
    # Make sure Hp is positive definite - no try/catch for Zygote compatibility
    Hp = nearest_pd(Hp)
    
    # Generate prediction
    return rand(MvNormal(μp, Hp))
end

# Helper functions
function transform_ab(raw_a, raw_b)
    a = sigmoid(raw_a)
    b = sigmoid(raw_b) * (1 - a)
    return a, b
end

function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end

export NetConstructor, DCCGarchNormal, posterior_predict