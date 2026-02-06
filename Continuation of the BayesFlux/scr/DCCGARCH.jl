#######################################################################################
#   Classical DCC-GARCH(1,1) - Complete Multi-Asset [2, 5, 10, 15, 30]
#   with 30 Assets, DCC and GARCH(1,1) Estimation fails to converge all the time (perhaps due to numerical issue and high dimensionality)
#  I should keep  the Classical DCC-GARCH as Benchmark Model
# 
#######################################################################################

using Optim, Distributions
using LinearAlgebra, Statistics
using CSV, DataFrames
using Printf, Plots, StatsPlots

# ══════════════════════════════════════════════════════════════════════════════
#                      UTILITY & PORTFOLIO FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

const VAR_CONFIDENCE_LEVELS = [0.01, 0.05, 0.10]

struct VaRResults
    PF::Float32; TUFF::Int32; LRTUFF::Float32; LRUC::Float32
    LRIND::Float32; LRCC::Float32; BASEL::Int32
end

function VaR_backtest(returns::Array{Float32,1}, VaR::Array{Float32,2}, alphas::Array{Float32,1})
    results = Dict{Float32, VaRResults}()
    for i in 1:length(alphas)
        hit = returns .< VaR[:, i]
        n1, n0 = sum(hit), length(hit) - sum(hit)
        PF = Float32(n1/length(hit))
        PF == 0 && continue; alpha = alphas[i]
        limits = cumsum(pdf.(Binomial(length(returns), alpha), 1:50)); green = count(limits .< 0.90)
        yellow = green + count((limits .> 0.90) .& (limits .< 0.99))
        TUFF, LRTUFF, LRUC, LRIND, LRCC = Int32(0), Float32(NaN), Float32(NaN), Float32(NaN), Float32(NaN)
        if n1 != 0; first_hit = findfirst(hit)
            if !isnothing(first_hit); TUFF = Int32(first_hit)
                try; log_term1 = log(alpha * (1-alpha)^(TUFF-1)); log_term2 = log((1/TUFF) * (1-1/TUFF)^(TUFF-1))
                    isfinite(log_term1) && isfinite(log_term2) && (LRTUFF = Float32(-2 * log_term1 + 2 * log_term2))
                catch; end; end; end
        if n1 != 0; try; log_likelihood_unrestricted = n1*log(PF) + n0*log(1-PF)
                log_likelihood_restricted = n1*log(alpha) + n0*log(1-alpha)
                if isfinite(log_likelihood_unrestricted) && isfinite(log_likelihood_restricted)
                    LRUC = Float32(-2 * (log_likelihood_restricted - log_likelihood_unrestricted)); end
            catch; end; end
        if n1 != 0; n00=n01=n10=n11=0
            for j in 1:(length(returns)-1); n00 += (hit[j]==0 && hit[j+1]==0); n01 += (hit[j]==0 && hit[j+1]==1)
                n10 += (hit[j]==1 && hit[j+1]==0); n11 += (hit[j]==1 && hit[j+1]==1); end
            try; if (n00+n01) > 0 && (n10+n11) > 0
                    p01, p11, p2 = n01/(n00+n01), n11/(n10+n11), (n01+n11)/(n00+n01+n10+n11)
                    if p01 > 0 && p11 > 0 && p2 > 0 && (1-p2) > 0 && (1-p01)>0 && (1-p11)>0
                        log_unrestricted = n00*log(1-p01) + n01*log(p01) + n10*log(1-p11) + n11*log(p11)
                        log_restricted = (n00+n10)*log(1-p2) + (n01+n11)*log(p2)
                        isfinite(log_unrestricted) && isfinite(log_restricted) && (LRIND = Float32(-2 * (log_restricted - log_unrestricted)))
                    end; end; catch; end; end
        LRCC = isfinite(LRUC) && isfinite(LRIND) ? LRUC + LRIND : (isfinite(LRUC) ? LRUC : (isfinite(LRIND) ? LRIND : Float32(NaN)))
        BASEL = n1 >= yellow ? Int32(-1) : (n1 <= yellow && n1 > green ? Int32(0) : Int32(1))
        results[alpha] = VaRResults(PF, TUFF, LRTUFF, LRUC, LRIND, LRCC, BASEL)
    end; return results
end

function enhanced_var_backtesting(portfolio_returns, daily_portfolio_vars, confidence_levels = VAR_CONFIDENCE_LEVELS)
    n_obs, n_levels = length(portfolio_returns), length(confidence_levels)
    returns_f32, alphas_f32 = Float32.(portfolio_returns), Float32.(confidence_levels)
    var_matrix = Matrix{Float32}(undef, n_obs, n_levels)
    for t in 1:n_obs, (j, α) in enumerate(confidence_levels); var_matrix[t, j] = Float32(-daily_portfolio_vars[t][j]); end
    return VaR_backtest(returns_f32, var_matrix, alphas_f32)
end

nearest_pd(A) = (A + A')/2 + 1e-1*I


# ----------------------------------------------------------------------------------
function construct_minimum_variance_portfolio(cov_matrix)
    Σ = Matrix{Float64}(cov_matrix)
    N = size(Σ, 1)
    ones_vec = ones(Float64, N)
    
    F = cholesky(Symmetric(nearest_pd(Σ)))
    u = F \ (F' \ ones_vec)
    w = u / (ones_vec' * u)[1]
    
    return Float32.(w)
end

function calculate_portfolio_var(portfolio_weights, forecast_mean, cov_matrix; confidence_levels=(0.01, 0.05, 0.10))
    μp = portfolio_weights' * forecast_mean
    σp = sqrt(max(0.0, portfolio_weights' * cov_matrix * portfolio_weights))
    z = Dict(0.01=>2.326, 0.05=>1.645, 0.10=>1.282)
    var_results = Dict{Float64, Float64}()
    for α in confidence_levels; var_results[α] = -μp + z[α]*σp; end
    return var_results, μp, σp
end

# ══════════════════════════════════════════════════════════════════════════════
#                     CLASSICAL DCC-GARCH ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

function garch_log_likelihood(p, returns)
    μ, ω, α, β = p
    if ω <= 1e-8 || α < 0 || β < 0 || α + β >= 1.0; return Inf; end
    T = length(returns); σ² = zeros(T); unconditional_var = ω / (1 - α - β)
    if unconditional_var <= 0; return Inf; end; σ²[1] = unconditional_var
    for t in 2:T; σ²[t] = ω + α * (returns[t-1] - μ)^2 + β * σ²[t-1]; end
    if any(v -> v <= 0 || !isfinite(v), σ²); return Inf; end
    logL = -sum(logpdf.(Normal.(μ, sqrt.(σ²)), returns))
    return isfinite(logL) ? logL : Inf
end

# Estimate GARCH(1,1) parameters and standardized residuals
function estimate_garch(returns::Vector{Float64})
    T = length(returns)
    initial_params = [mean(returns), var(returns)*0.05, 0.1, 0.85]
    objective = p -> garch_log_likelihood(p, returns)
    result = optimize(objective, initial_params, NelderMead(), Optim.Options(iterations=1000, show_trace=false))
    
    if Optim.converged(result)
        μ, ω, α, β = Optim.minimizer(result)
        σ² = zeros(T)
        σ²[1] = ω / (1 - α - β)
        for t in 2:T
            σ²[t] = ω + α * (returns[t-1] - μ)^2 + β * σ²[t-1]
        end
        std_residuals = (returns .- μ) ./ sqrt.(σ²)
        return (μ, ω, α, β), std_residuals, σ²
    else

        @warn "GARCH estimation did not converge."
        return (NaN, NaN, NaN, NaN), fill(NaN, T), fill(NaN, T)
    end
end

function dcc_log_likelihood(p, Z::Matrix{Float64})
    a, b = p; if a < 0 || b < 0 || a + b >= 1.0; return Inf; end; N, T = size(Z)
    Q_bar = cov(Z, dims=2); Q = copy(Q_bar); logL = 0.0
    for t in 1:T; z_t = Z[:, t]; Q = (1 - a - b) * Q_bar + a * (z_t * z_t') + b * Q
        try; inv_sqrt_diag_Q = Diagonal(1.0 ./ sqrt.(diag(Q))); R = inv_sqrt_diag_Q * Q * inv_sqrt_diag_Q
            logL -= 0.5 * (logdet(R) + z_t' * inv(R) * z_t)
        catch; return Inf; end; end
    return isfinite(logL) ? -logL : Inf
end

function estimate_dcc(Z::Matrix{Float64})
    result = optimize(p -> dcc_log_likelihood(p, Z), [0.05, 0.9], NelderMead(), Optim.Options(iterations=500, show_trace=false))
    if Optim.converged(result); return Optim.minimizer(result); else; @warn "DCC estimation did not converge."; return [NaN, NaN]; end
end

struct DCCGarchFit
    garch_params::Matrix{Float64}; dcc_params::Vector{Float64}; Q_bar::Matrix{Float64}
    last_residuals::Vector{Float64}; last_variances::Vector{Float64}; last_Q::Matrix{Float64}
    training_rmse::Float64
end

function fit_dcc_garch(returns::Matrix{Float64})
    N, T = size(returns); garch_params = zeros(N, 4); std_residuals_matrix = zeros(N, T); variances_matrix = zeros(N, T)
    fitted_means = zeros(N, T)
    Threads.@threads for i in 1:N; params, residuals, variances = estimate_garch(returns[i, :])
        garch_params[i, :] = [params...]; std_residuals_matrix[i, :] = residuals
        variances_matrix[i, :] = variances; fitted_means[i, :] .= params[1]; end
    dcc_ab = estimate_dcc(std_residuals_matrix); a, b = dcc_ab; Q_bar = cov(std_residuals_matrix, dims=2); Q_T = copy(Q_bar)
    for t in 1:T; z_t = std_residuals_matrix[:, t]; Q_T = (1 - a - b) * Q_bar + a * (z_t * z_t') + b * Q_T; end
    training_rmse = sqrt(mean((returns - fitted_means).^2))
    return DCCGarchFit(garch_params, dcc_ab, Q_bar, std_residuals_matrix[:, end], variances_matrix[:, end], Q_T, training_rmse)
end

function forecast_dcc_garch(model::DCCGarchFit, last_returns::Vector{Float64})
    N = size(model.garch_params, 1); μ, ω, α, β = model.garch_params[:, 1], model.garch_params[:, 2], model.garch_params[:, 3], model.garch_params[:, 4]
    a, b = model.dcc_params; next_variances = ω + α .* (last_returns - μ).^2 + β .* model.last_variances
    D_next = Diagonal(sqrt.(max.(0, next_variances)))
    Q_next = (1 - a - b) * model.Q_bar + a * (model.last_residuals * model.last_residuals') + b * model.last_Q
    inv_sqrt_diag_Q = Diagonal(1.0 ./ sqrt.(diag(Q_next))); R_next = inv_sqrt_diag_Q * Q_next * inv_sqrt_diag_Q
    H_next = D_next * R_next * D_next
    return nearest_pd(H_next), μ
end

# ══════════════════════════════════════════════════════════════════════════════
#                      MAIN ANALYSIS & ROLLING WINDOW
# ══════════════════════════════════════════════════════════════════════════════

function classical_dcc_garch_analysis(data_path::String, n_assets::Int)
   
    println("CLASSICAL DCC-GARCH ANALYSIS FOR $n_assets ASSETS")

    df = CSV.read(data_path, DataFrame)
    etf_names_base = ["16383", "16386", "16388", "16397", "16403", "16412", "16414", "16418", "16421", "16423", "16424", "16426", "16433", "16437", "16452", "16460", "24697", "27635", "28272", "28273", "28274", "28275", "28276", "28277", "28278", "28279", "28280", "31372", "31466"]
    
    cols_to_scale = vcat(etf_names_base, "rf")
    for col in cols_to_scale
        if col in names(df)
            df[!, col] = df[!, col] .* 100
        end
    end
    println("Data loaded and multiplied by 100.")
    
    for col in vcat(etf_names_base, "rf"); if any(ismissing, df[!, col]); df[!, col] = coalesce.(df[!, col], mean(skipmissing(df[!, col]))); end; end

    local returns_selected, asset_names
    if n_assets >= 30
        etf_returns = Matrix{Float64}(df[!, etf_names_base])
        rf_returns = Vector{Float64}(df[!, "rf"])
        returns_selected = hcat(etf_returns, rf_returns)
        asset_names = vcat(etf_names_base, "rf")
        n_assets = 30
    else
        returns_selected = Matrix{Float64}(df[!, etf_names_base[1:n_assets]])
        asset_names = etf_names_base[1:n_assets]
    end

    train_window = 1000; test_start = 3701; test_end = 3949
    test_indices = test_start:test_end; n_test = length(test_indices)

    println("Data prepared: $(size(returns_selected, 1)) observations for $n_assets assets.")
    println("Rolling window forecast from day $test_start to $test_end ($n_test steps).")
    
    portfolio_returns = zeros(Float64, n_test); daily_portfolio_vars = Vector{Vector{Float64}}(undef, n_test)
    daily_portfolio_weights = zeros(Float32, n_assets, n_test); daily_portfolio_volatility = zeros(Float64, n_test)
    training_rmses = zeros(Float64, n_test); test_squared_errors = zeros(Float64, n_test, n_assets)

    for (i, t) in enumerate(test_indices)
        if i % 50 == 0; @printf "Processing forecast %d/%d (Day %d)...\n" i n_test t; end
        
        window_end = t - 1; window_start = window_end - train_window + 1
        training_returns = collect(returns_selected[window_start:window_end, :]')
        
        model = fit_dcc_garch(training_returns)
        
        if any(isnan, model.dcc_params) || any(isnan, model.garch_params)
            @warn "Model fitting failed at step $i. Skipping."; portfolio_returns[i] = NaN
            daily_portfolio_vars[i] = [NaN, NaN, NaN]; daily_portfolio_weights[:, i] .= NaN
            daily_portfolio_volatility[i] = NaN; training_rmses[i] = NaN; test_squared_errors[i, :] .= NaN
            continue; end
        
        training_rmses[i] = model.training_rmse
        last_day_returns = returns_selected[window_end, :]
        H_forecast, μ_forecast = forecast_dcc_garch(model, last_day_returns)
        
        w = construct_minimum_variance_portfolio(H_forecast)
        var_results, _, σp = calculate_portfolio_var(w, μ_forecast, H_forecast)
        
        actual_returns_t = returns_selected[t, :]
        portfolio_returns[i] = w' * actual_returns_t
        daily_portfolio_vars[i] = [var_results[α] for α in VAR_CONFIDENCE_LEVELS]
        daily_portfolio_weights[:, i] = w; daily_portfolio_volatility[i] = σp
        test_squared_errors[i, :] = (actual_returns_t .- μ_forecast).^2
    end

    valid_indices = .!isnan.(portfolio_returns)
    valid_returns = portfolio_returns[valid_indices]; valid_vars = daily_portfolio_vars[valid_indices]
    valid_weights = daily_portfolio_weights[:, valid_indices]; valid_vols = daily_portfolio_volatility[valid_indices]
    valid_train_rmses = training_rmses[valid_indices]; valid_test_sq_err = test_squared_errors[valid_indices, :]

    avg_training_rmse = mean(valid_train_rmses)
    test_rmse = sqrt(mean(valid_test_sq_err))
    
    print_classical_portfolio_summary(valid_returns, valid_vols, valid_vars, avg_training_rmse, test_rmse)
    
    var_backtest_results = enhanced_var_backtesting(valid_returns, valid_vars)
    
    if !isempty(var_backtest_results)
        println("\nVaR Backtesting Results (* = Rejects H0 at 5% significance level)")
        println("─────┬───────────┬──────────────────────┬──────────────────────┬───────────")
        println("  α  │ Fail Rate │ LRuc (crit = 3.84)   │ LRcc (crit = 5.99)   │ Basel Zone")
        println("─────┼───────────┼──────────────────────┼──────────────────────┼───────────")
        for (alpha, result) in sort(collect(var_backtest_results), by=x->x[1])
            level_pct = round(Int, 100*alpha)
            basel_status = result.BASEL == 1 ? "Green" : (result.BASEL == 0 ? "Yellow" : "Red")
            pf_str = @sprintf("%.1f%%", 100 * result.PF)

            lruc_str = isnan(result.LRUC) ? "NaN" : @sprintf("%.2f %s", result.LRUC, result.LRUC > 3.84 ? "*" : " ")
            lrcc_str = isnan(result.LRCC) ? "NaN" : @sprintf("%.2f %s", result.LRCC, result.LRCC > 5.99 ? "*" : " ")

            @printf " %-3d%% │ %-9s │ %-20s │ %-20s │ %s\n" level_pct pf_str lruc_str lrcc_str basel_status
        end
        println("─────┴───────────┴──────────────────────┴──────────────────────┴───────────")
    else
        println("\nVaR backtesting could not be performed.")
    end
    
    save_classical_results(valid_returns, valid_vars, valid_weights, var_backtest_results, n_assets)
    plot_and_save_visuals(valid_returns, valid_vars, var_backtest_results, valid_vols, asset_names)
end

# ══════════════════════════════════════════════════════════════════════════════
#                NEW: REPORTING, PLOTTING, AND SAVING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

function print_classical_portfolio_summary(returns, volatilities, vars, train_rmse, test_rmse)
    println("PERFORMANCE & RMSE SUMMARY (CLASSICAL DCC-GARCH)")
    
    avg_ret = mean(returns)
    avg_vol = mean(volatilities)
    avg_var1 = mean([v[1] for v in vars])
    avg_var5 = mean([v[2] for v in vars])
    avg_var10 = mean([v[3] for v in vars])

    println("\n┌────────────────────┬───────────────────┐")
    println("│ Metric             │ Value             │")
    println("├────────────────────┼───────────────────┤")
    @printf "│ Avg. Training RMSE │ %-17.6f │\n" train_rmse
    @printf "│ Testing RMSE       │ %-17.6f │\n" test_rmse
    println("├────────────────────┼───────────────────┤")
    @printf "│ Avg. Return        │ %-13.4f %% │\n" avg_ret
    @printf "│ Avg. Volatility    │ %-13.4f %% │\n" avg_vol
    @printf "│ Avg. VaR 1%%        │ %-13.4f %% │\n" avg_var1
    @printf "│ Avg. VaR 5%%        │ %-13.4f %% │\n" avg_var5
    @printf "│ Avg. VaR 10%%       │ %-13.4f %% │\n" avg_var10
    println("└────────────────────┴───────────────────┘")
end

function plot_enhanced_var_analysis(portfolio_returns::Vector{Float64}, daily_portfolio_vars::Vector{Vector{Float64}}, var_backtest_results::Dict{Float32, VaRResults}, asset_names::Vector{String})
    isempty(portfolio_returns) || isempty(daily_portfolio_vars) && return nothing
    try
        p1 = plot(title="Portfolio VaR vs Actual Returns (Classical DCC-GARCH)", xlabel="Day", ylabel="Return/VaR (%)", size=(1000, 400), legend=:bottomleft)
        plot!(p1, 1:length(portfolio_returns), portfolio_returns, label="Actual Returns", color=:black, linewidth=1.5)
        colors = [:red, :orange, :blue]
        for (i, level) in enumerate(VAR_CONFIDENCE_LEVELS)
            var_series = [vars[i] for vars in daily_portfolio_vars]
            plot!(p1, 1:length(var_series), -var_series, label="$(round(Int, 100*level))% VaR", color=colors[i], linewidth=2, linestyle=:dash)
            violations = portfolio_returns .< -var_series
            any(violations) && scatter!(p1, findall(violations), portfolio_returns[violations], color=colors[i], markersize=4, alpha=0.8, label=nothing)
        end
        p2 = bar(title="VaR Backtesting Results", xlabel="Confidence Level", ylabel="Test Statistic", size=(1000, 400))
        if !isempty(var_backtest_results)
            levels, lruc_values, lrind_values, lrcc_values = String[], Float64[], Float64[], Float64[]
            for (alpha, result) in sort(collect(var_backtest_results), by=x->x[1])
                push!(levels, "$(round(Int, 100*alpha))%"); push!(lruc_values, isfinite(result.LRUC) ? Float64(result.LRUC) : 0.0)
                push!(lrind_values, isfinite(result.LRIND) ? Float64(result.LRIND) : 0.0); push!(lrcc_values, isfinite(result.LRCC) ? Float64(result.LRCC) : 0.0)
            end
            x_pos = 1:length(levels)
            bar!(p2, x_pos .- 0.25, lruc_values, width=0.2, label="UC Test", alpha=0.8)
            bar!(p2, x_pos, lrind_values, width=0.2, label="IND Test", alpha=0.8)
            bar!(p2, x_pos .+ 0.25, lrcc_values, width=0.2, label="CC Test", alpha=0.8)
            hline!(p2, [3.84], label="Critical Value (5%)", linestyle=:dash, color=:red, linewidth=2); xticks!(p2, x_pos, levels)
        end
        p3 = histogram(portfolio_returns, bins=50, normalize=:pdf, alpha=0.7, title="Return Distribution with VaR Levels", xlabel="Portfolio Returns (%)", ylabel="Density", label="Return Distribution", color=:lightblue, size=(1000, 400))
        for (i, α) in enumerate(VAR_CONFIDENCE_LEVELS)
            if haskey(var_backtest_results, Float32(α))
                result = var_backtest_results[Float32(α)]; var_cutoff = quantile(portfolio_returns, α)
                line_color = result.BASEL == 1 ? :green : (result.BASEL == 0 ? :orange : :red)
                vline!(p3, [var_cutoff], label="$(round(Int,100*α))% VaR (PF: $(round(100*result.PF, digits=1))%)", linewidth=3, linestyle=:dash, color=line_color)
            end
        end
        return plot(p1, p2, p3, layout=(3,1), size=(1000, 1200))
    catch e; println("Warning: Plotting failed: $e"); return nothing; end
end

function plot_and_save_visuals(portfolio_returns, daily_vars, backtest_results, portfolio_vols, asset_names)
    n_assets = length(asset_names)
    var_plot = plot_enhanced_var_analysis(portfolio_returns, daily_vars, backtest_results, asset_names)
    if var_plot !== nothing; display(var_plot)
        savefig(var_plot, "classical_var_analysis_$(n_assets)_assets.png")
        println("\n VaR analysis plot saved as classical_var_analysis_$(n_assets)_assets.png"); end
        
    mvp_vol_plot = plot(1:length(portfolio_vols), portfolio_vols, title="MVP Volatility Evolution ($n_assets Assets)",
        xlabel="Day (Test Period)", ylabel="Portfolio Volatility (%)", linewidth=2, color=:purple, label="Predicted Volatility", size=(1000, 400))
    display(mvp_vol_plot)
    savefig(mvp_vol_plot, "classical_mvp_volatility_$(n_assets)_assets.png")
    println(" MVP volatility plot saved as classical_mvp_volatility_$(n_assets)_assets.png")
end

function save_classical_results(portfolio_returns, daily_vars, portfolio_weights, backtest_results, n_assets)
    println("\n Saving results to CSV files..."); results_dir = "classical_dcc_results"; !isdir(results_dir) && mkdir(results_dir)
    n_obs = length(portfolio_returns)
    var_df = DataFrame(Day = 1:n_obs, Actual_Return = portfolio_returns, VaR_1_percent = [v[1] for v in daily_vars],
        VaR_5_percent = [v[2] for v in daily_vars], VaR_10_percent = [v[3] for v in daily_vars])
    CSV.write(joinpath(results_dir, "portfolio_var_returns_$(n_assets)_assets.csv"), var_df); println("   ✓ Portfolio returns and VaR saved.")
    weights_df = DataFrame(portfolio_weights', :auto); rename!(weights_df, ["Asset_$(i)_Weight" for i in 1:n_assets]); weights_df.Day = 1:n_obs
    CSV.write(joinpath(results_dir, "mvp_weights_$(n_assets)_assets.csv"), weights_df); println("   ✓ Portfolio weights saved.")
    summary_df = DataFrame(Confidence_Level=Float32[], PF=Float32[], TUFF=Int32[], LRUC=Float32[], LRIND=Float32[], LRCC=Float32[], Basel_Zone=String[])
    for (alpha, result) in sort(collect(backtest_results), by=x->x[1])
        basel_str = result.BASEL == 1 ? "Green" : (result.BASEL == 0 ? "Yellow" : "Red")
        push!(summary_df, (alpha, result.PF, result.TUFF, result.LRUC, result.LRIND, result.LRCC, basel_str)); end
    CSV.write(joinpath(results_dir, "var_backtest_summary_$(n_assets)_assets.csv"), summary_df); println("   ✓ VaR backtest summary saved.")
    println("All results for $n_assets assets saved in '$results_dir'.")
end

# ══════════════════════════════════════════════════════════════════════════════
#                              RUN THE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# ❗ IMPORTANT
data_path = "/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/data/etfReturns.csv"

asset_counts_to_run = [2, 5, 10, 15, 29,30]

for n in asset_counts_to_run
    classical_dcc_garch_analysis(data_path, n)
end
