using Distributions

#----------------------------------------------------------------------- 
# PURPOSE:  
# To perform Value-at-Risk (VaR) backtesting for long position using the
#unconditional coverage, independence, and conditional coverage family of tests.
#----------------------------------------------------------------------- 
# USAGE:  
# results = VaR(retuns, VaR, alphas) 
# 
# INPUTS: 
# returns:     ( m x 1 ) vector of the out-of-sample data, 
# VaR:         ( m x 1 ) vector of VaR estimates 
# alphas:     a%s vector
#----------------------------------------------------------------------- 
# OUTPUTS:  
# results:  PF:      Percentage of Failures 
#           TUFF:    Time Until First Failure 
#           LRTUFF:  Likelihood Ratio of Time Until First Failure 
#           LRUC:    Likelihood Ratio Unconditional Coverage   
#           LRIND:   Likelihood Ratio Independence Coverage 
#           LRCC:    Likelihood Ratio Conditional Coverage    
#           Basel:   Basel II Accord 
#----------------------------------------------------------------------- 

struct VaRResults
    PF::Float32
    TUFF::Int
    LRTUFF::Float32
    LRUC::Float32
    LRIND::Float32
    LRCC::Float32
    BASEL::Int
end

function VaR(returns::Array{Float32,1}, VaR::Array{Float32,1}, alphas::Array{Float32,1})
    VaR = reshape(VaR, :, 1)
    VaR(returns, VaR, alphas)
end

function VaR(returns::Array{Float32,1}, VaR::Array{Float32,2}, alphas::Array{Float32,1})
    # Ensure that VaR and alphas dimensions match
    if size(VaR, 2) != length(alphas)
        error("The number of VaR columns must match the length of alphas")
    end

    # Initializations
    results = Dict()

    for i in 1:length(alphas)
        hit = returns .< VaR[:, i]
        n1 = sum(hit)
        n0 = length(hit) - n1
        PF = n1/length(hit)
        if (PF == 0)
            println("Your estimation is overshooting, 0 mistake for alpha = ", alphas[i])
            continue
        end

        alpha = alphas[i]

        limits = cumsum(pdf.(Binomial(length(returns), alpha), 1:50))
        green = count(limits .< 0.90)
        yellow = green + count((limits .> 0.90) .& (limits .< 0.99))

        TUFF = NaN
        LRTUFF = NaN
        LRUC = NaN
        LRIND = NaN
        LRCC = NaN
        BASEL = NaN

        # Kupiec (1995) Time Until First Failure 
        if (n1 != 0 && PF != 0)
            TUFF = findfirst(hit)
            LRTUFF = -2*log((alpha*(1-alpha)^(TUFF-1))) + 2*log((1/TUFF)*(1-1/TUFF)^(TUFF-1)) 
        end

        # Christoffersen Tests 
        # Unconditional Coverage 
        if (n1 != 0 && PF != 0)
            #println("$alpha,$n0,$n1,$PF")
            LRUC = -2*(n1*log(alpha) + n0*log(1-alpha) - n1*log(PF) - n0*log(1-PF))
            #println(LRUC)
        end

        # Independence Coverage 
        if (n1 != 0 && PF != 0)
            n00=n01=n10=n11=0
            for i in 1:(length(returns)-1)
                n00 += (hit[i]==0 && hit[i+1]==0)
                n01 += (hit[i]==0 && hit[i+1]==1)
                n10 += (hit[i]==1 && hit[i+1]==0)
                n11 += (hit[i]==1 && hit[i+1]==1)
            end

            p01 = n01/(n00+n01)
            p00 = 1 - p01
            p11 = n11/(n10+n11)
            p2 = (n01+n11)/(n00+n01+n10+n11)

            # In case n11 = 0, then the test is estimated as ((1-p01)^n00)*(p01^n01)
            if (n11 == 0)
                LRIND = ((1-p01)^n00)*(p01^n01)
            else
                LRIND = -2*log((((1-p2)^(n00+n10))*(p2^(n01+n11)))/(((1-p01)^n00)*(p01^n01)*((1-p11)^n10)*(p11^n11)))
            end
        end

        # Conditional Coverage
        if isfinite(LRUC)
            LRCC = LRUC + LRIND
        else
            LRCC = LRIND
        end

        # BASEL II Accord
        if n1 >= yellow
            BASEL = -1
        elseif n1 <= yellow && n1 > green
            BASEL = 0
        else
            BASEL = 1
        end

        results[alpha] = VaRResults(PF, TUFF, LRTUFF, LRUC, LRIND, LRCC, BASEL)
    end

    return results
end
