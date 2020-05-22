#=
Generic helpers for k-armed bandit problems and algorithms.
=#

module KArmedBandits

using Random: randn
using Statistics: mean

export KArmedBandit, WhiteBoxModel, test_model

#--- DEFAULT SETTINGS

# problem
const K = 10 # bandits' number of arms
const VARIABILITY = 1.0 # the standard deviation of the reward distribution
const DRIFT = 0.01 # a rate of value drift for nonstationary problems

# learner algorithm (model)
const ALPHA = 0.1 # step size

# testbed
const SIMULATIONS = 1000 # number of simulations per model (learner)
const STEPS = 10000 # number of steps per simulation

#--- BANDIT PROBLEM

mutable struct KArmedBandit
    arms::Int64
    values::Array{Float64, 1}
    reward_variability::Float64
    drift::Float64
end

# Constructor for equal starting values
KArmedBandit(arms::Int64 = K, reward_variability::Float64 = VARIABILITY, drift::Float64 = DRIFT) =
    KArmedBandit(arms, zeros(arms), reward_variability, drift)

# Constructor for arbitrary starting values
KArmedBandit(
    values::Array{Float64, 1},
    reward_variability::Float64 = VARIABILITY,
    drift::Float64 = DRIFT,
) = KArmedBandit(length(values), values, reward_variability, drift)

function get_reward(bandit::KArmedBandit, action::Int64)::Float64
    # sample from the normal distribution with mean equal to the value of the action
    return bandit.values[action] + bandit.reward_variability * randn()
end

function update_values!(bandit::KArmedBandit)
    return for i in 1:length(bandit.values)
        # add a normally-distributed perturbation with mean 0 and standard deviation equal to the drift
        bandit.values[i] += bandit.drift * randn()
    end
end

#--- TESTBED

function run_simulation!(
    model,
    bandit::KArmedBandit = KArmedBandit(),
    steps::Int64 = STEPS,
)::Tuple{Array{Float64, 1}, Array{Bool, 1}}

    # track model performance
    rewards = zeros(steps)
    optimal = fill(false, steps)

    for step in 1:steps
        optimal_action = argmax(bandit.values)

        # run a single step, updating the model
        action, reward = run_step!(model, bandit)

        # track performance
        rewards[step] = reward
        optimal[step] = action == optimal_action

        # update values for nonstationary problems
        update_values!(bandit)
    end

    return rewards, optimal
end

function test_model(
    model,
    bandit::KArmedBandit = KArmedBandit(),
    simulations::Int64 = SIMULATIONS,
    steps::Int64 = STEPS,
)::Tuple{Array{Float64, 1}, Array{Float64, 1}}

    rewards = fill([], simulations)
    optimal = fill([], simulations)
    Threads.@threads for sim in 1:simulations
        rewards[sim], optimal[sim] = run_simulation!(deepcopy(model), deepcopy(bandit), steps)
    end

    avg_rewards = [mean([sim[step] for sim in rewards]) for step in 1:steps]
    percent_optimal = [mean([sim[step] for sim in optimal]) * 100.0 for step in 1:steps]
    return avg_rewards, percent_optimal
end

#--- GENERIC BANDIT ALGORITHM HELPERS

function update_estimated_values!(
    estimated_values::Array{Float64, 1},
    action::Int64,
    reward::Float64,
    step_size::Float64 = ALPHA,
)
    # standard incremental update of estimated values
    old_estimate = estimated_values[action]
    new_estimate = old_estimate + step_size * (reward - old_estimate)
    return estimated_values[action] = new_estimate
end

#--- CONTROL ALGORITHM (WHITE BOX MODEL)

struct WhiteBoxModel end

function run_step!(model::WhiteBoxModel, bandit::KArmedBandit)::Tuple{Int64, Float64}
    # get a reward from the best action
    action = argmax(bandit.values)
    reward = get_reward(bandit, action)
    return action, reward
end

#--- IMPORT OTHER ALGORITHMS

include("epsilon-greedy.jl")
include("gradient.jl")
include("upper-confidence-bound.jl")

end
