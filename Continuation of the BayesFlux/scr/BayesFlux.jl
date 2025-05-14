


# BayesFlux: A Julia module for Bayesian Neural Networks. 
#It includes various layers (dense, recurrent), model construction and 
#deconstruction utilities, a range of likelihood functions, network priors, 
#and initializers. The module also provides core functionalities for BNNs, 
#mode finding algorithms, tools for Bayesian inference (including MCMC and 
#Variational Inference methods), and additional utilities, particularly for RNNs.
#This module is ideal for applications requiring probabilistic modeling and 
#uncertainty quantification in neural networks.


module BayesFlux

include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/utils/gradient_utils.jl")

include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/layers/dense.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/layers/recurrent.jl")

include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/model/deconstruct.jl")
export destruct
export NetConstructor

include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/likelihoods/abstract.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/likelihoods/feedforward.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/likelihoods/seq_to_one.jl")
export BNNLikelihood, posterior_predict
export FeedforwardNormal, FeedforwardTDist
export SeqToOneNormal, SeqToOneTDist

#####Trial likelihoods
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/likelihoods/arch_seq_to_one.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/likelihoods/DCCGarchSeqToMultiNormal.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/likelihoods/DCCGarchSeqToMultiTDist.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/likelihoods/garch_seq_to_one.jl")
export ArchSeqToOneNormal
export ArchSeqToOneTDist
export GarchSeqToOneNormal
export GarchSeqToOneTDist
export DCCGarchSeqToMultiNormal
export DCCGarchSeqToMultiTDist
export transform_ab
export NetConstructor,DCCGarchNormal , posterior_predict


include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/netpriors/abstract.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/netpriors/gaussian.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/netpriors/mixturescale.jl")
export NetworkPrior, sample_prior
export GaussianPrior
export MixtureScalePrior

include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/initialisers/abstract.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/BayesFlux-2.jl-main/src/initialisers/basics.jl")
export BNNInitialiser
export InitialiseAllSame

include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/model/BNN.jl")
export BNN
export split_params
export loglikeprior, ∇loglikeprior
export sample_prior_predictive, get_posterior_networks, sample_posterior_predict

include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mode/abstract.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mode/flux.jl")
export BNNModeFinder, find_mode, step!
export FluxModeFinder

# Abstract MCMC
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/abstract.jl")
# Mass Adapters
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/adapters/mass/abstract_mass.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/adapters/mass/diagcovariancemassadapter.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/adapters/mass/fixedmassmatrix.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/adapters/mass/fullcovariancemassadapter.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/adapters/mass/rmspropmassadapter.jl")
export MassAdapter
export DiagCovMassAdapter, FixedMassAdapter, FullCovMassAdapter, RMSPropMassAdapter
# Stepsize Adapters
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/adapters/stepsize/abstract_stepsize.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/adapters/stepsize/constantstepsize.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/adapters/stepsize/dualaveragestepsize.jl")
export StepsizeAdapter
export ConstantStepsize, DualAveragingStepSize
# MCMC Methods
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/sgld.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/sgnht.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/sgnht-s.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/ggmc.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/amh.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/mcmc/hmc.jl")
export MCMCState, mcmc
export SGLD
export SGNHT
export SGNHTS
export GGMC
export AdaptiveMH
export HMC


# Variational Inference Methods
# include("./inference/vi/advi.jl")
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/inference/vi/bbb.jl")
# export advi
export bbb

# Utilities
include("/Users/kevin/Documents/University of Maastricht /Master Econometrics and Operations Research/Master Thesis /MasterThesis/Continuation of the BayesFlux/scr/utils/rnn_utils.jl")
export make_rnn_tensor
 
end # module
