#=
Exercise 2.11
Make a figure analogous to Figure 2.6 for the nonstationary case outlined in Exercise 2.5.
Include the constant-step-size ε-greedy algorithm with α = 0.1. Use runs of 200,000 steps and,
as a performance measure for each algorithm and parameter setting, use the average reward over the
last 100,000 steps.

The algorithms I will include are:
- constant-step-size ε-greedy with α = 0.1 (varying ε)
- constant-step-size ε-greedy with optimistic initialization and α = 0.1 (varying Q0)
- gradient (varying α) - upper confidence bound (varying c)

I will vary each of these parameters from 1/128 to 4 by factors of 2.
=#

#---
using Plots
using Random
using Statistics
using StatsBase

#--- SETTINGS

const VERBOSE = true
println("Threads: $(Threads.nthreads())")

# simulation parameters
const K = 10
const SIMULATIONS = 2000
const STEPS = 200000
const ALPHA = 0.1 # for ε-greedy and UCB
const EPSILON = 0.1 # for optimistic ε-greedy

# hyperparameter values
const HYPERPARAM_VALUES = [2.0^n for n in -7:2]

#--- BANDIT SIMULATION HELPERS

function get_reward(values::Array{Float64, 1}, action::Int64)::Float64
    # sample from the normal distribution with variance 1 (i.e. stdev 1)
    # and mean equal to the value of the action
    values[action] + randn()
end

function update_estimated_values!(
    estimated_values::Array{Float64, 1},
    action::Int64,
    reward::Float64,
)
    # incremental update of estimated values
    old_estimate = estimated_values[action]
    new_estimate = old_estimate + ALPHA * (reward - old_estimate)
    estimated_values[action] = new_estimate
end

function perturb_values!(values::Array{Float64, 1})
    for i in 1:K
        # add a normally distributed increment with mean 0 and standard deviation 0.01
        values[i] += 0.01 * randn()
    end
end

function run_simulation!(model)::Float64
    # reality
    values = zeros(K)

    # track model performance
    rewards = zeros(STEPS)

    for step in 1:STEPS
        # run a single step, updating the model
        reward = run_step!(values, model)

        # track performance
        rewards[step] = reward

        # permute true values (nonstationary)
        perturb_values!(values)
    end

    # return average reward over second half of simulation
    mean(rewards[(fld(STEPS, 2) + 1):STEPS])
end

function test_model(model)::Float64
    performances = zeros(SIMULATIONS)
    Threads.@threads for i in 1:SIMULATIONS
        performances[i] = run_simulation!(deepcopy(model))
    end

    mean(performances)
end;

#--- PERFECT CHOICE CONTROL

struct WhiteBoxModel end

function run_step!(values::Array{Float64, 1}, model::WhiteBoxModel)
    # get a reward from the best action
    action = argmax(values)
    get_reward(values, action)
end;

#--- EPSILON GREEDY

mutable struct EpsilonGreedyModel
    epsilon::Float64
    estimated_values::Array{Float64, 1}
end

function choose_action(model::EpsilonGreedyModel)::Int64
    # ε-greedy algorithm
    if rand() < model.epsilon
        # explore
        rand(1:K)
    else
        # exploit
        argmax(model.estimated_values)
    end
end

function run_step!(values::Array{Float64, 1}, model::EpsilonGreedyModel)
    # pick action, get reward, update model
    action = choose_action(model)
    reward = get_reward(values, action)
    update_estimated_values!(model.estimated_values, action, reward)

    # track performance
    reward
end;

#--- UPPER CONFIDENCE BOUND

mutable struct UcbModel
    c::Float64
    estimated_values::Array{Float64, 1}
    choices::Array{Float64, 1}
    step::Int64
end

function choose_action(model::UcbModel)::Int64
    upper_bounds = [
        model.estimated_values[action] + model.c * sqrt(log(model.step) / model.choices[action]) for action in 1:K
    ]
    action = argmax(upper_bounds)
    model.choices[action] += 1
    action
end

function run_step!(values::Array{Float64, 1}, model::UcbModel)::Float64
    # pick action, get reward, update model
    model.step += 1
    action = choose_action(model)
    reward = get_reward(values, action)
    update_estimated_values!(model.estimated_values, action, reward)

    # track performance
    reward
end;

#--- STOCHASTIC GRADIENT ASCENT

mutable struct GradientModel
    alpha::Float64
    preferences::Array{Float64, 1}
    baseline::Float64
end

function get_probabilities(preferences::Array{Float64, 1})::Array{Float64, 1}
    # non-normalized probabilities are e^H, where H is the preference of an action
    probabilities = exp.(preferences)
    probabilities ./ sum(probabilities)
end

function choose_action(probabilities::Array{Float64, 1})::Int64
    sample(Weights(probabilities))
end

function update_baseline!(model::GradientModel, reward::Float64)
    # incremental update with constant step size (since nonstationary)
    model.baseline += model.alpha * (reward - model.baseline)
end

function update_preferences!(
    model::GradientModel,
    probabilities::Array{Float64, 1},
    chosen_action::Int64,
    reward::Float64,
)
    step = model.alpha * (reward - model.baseline)
    for action in 1:K
        if action == chosen_action
            model.preferences[action] += step * (1 - probabilities[action])
        else
            model.preferences[action] -= step * probabilities[action]
        end
    end
end

function run_step!(values::Array{Float64, 1}, model::GradientModel)::Float64
    # pick action, get reward, update model
    probabilities = get_probabilities(model.preferences)
    action = choose_action(probabilities)

    reward = get_reward(values, action)
    if model.baseline != model.baseline # NaN, uninitialized
        model.baseline = reward
    end

    update_preferences!(model, probabilities, action, reward)
    update_baseline!(model, reward)

    # track performance
    reward
end;

#--- GET WHITEBOX BASELINE

println("Getting white box baseline...")
white_box_baseline = VERBOSE ? (@time test_model(WhiteBoxModel())) : test_model(WhiteBoxModel())

#--- RUN PARAMETER STUDIES

function run_parameter_studies()
    epsilon_greedy_perf = []
    epsilon_greedy_optimistic_perf = []
    ucb_perf = []
    gradient_perf = []

    println("Running parameter studies...")

    for val in HYPERPARAM_VALUES
        # intitialize models with the particular hyperparameter value
        # then run simulations and record performance

        # ε-greedy, varying ε (exploration)
        if val <= 1 # epsilon is a probability
            epsilon_greedy_model = EpsilonGreedyModel(val, zeros(K))
            VERBOSE && println("\nTesting ε-greedy algorithm (ε = $val)...")
            perf = VERBOSE ? (@time test_model(epsilon_greedy_model)) : test_model(
                epsilon_greedy_model,
            )
            push!(epsilon_greedy_perf, perf)
        else
            push!(epsilon_greedy_perf, 0.0)
        end

        # optimistic ε-greedy, varying Q0 (initialization)
        epsilon_greedy_optimistic_model = EpsilonGreedyModel(EPSILON, fill(val, K))
        VERBOSE &&
        println("\nTesting ε-greedy algorithm with optimistic initializations (Q0 = $val)...")
        perf = VERBOSE ? (@time test_model(epsilon_greedy_optimistic_model)) :
            test_model(epsilon_greedy_optimistic_model)
        push!(epsilon_greedy_optimistic_perf, perf)

        # upper confidence bound, varying c (confidence)
        ucb_model = UcbModel(val, zeros(K), zeros(K), 0)
        VERBOSE && println("\nTesting upper confidence bound (UCB) algorithm (c = $val)...")
        perf = VERBOSE ? (@time test_model(ucb_model)) : test_model(ucb_model)
        push!(ucb_perf, perf)

        # stochastic gradient ascent, varying α (step size)
        gradient_model = GradientModel(val, zeros(K), NaN)
        VERBOSE && println("\nTesting stochastic gradient ascent algorithm (α = $val)...")
        perf = VERBOSE ? (@time test_model(gradient_model)) : test_model(gradient_model)
        push!(gradient_perf, perf)
    end

    println("\nParameter studies complete!")
    epsilon_greedy_perf, epsilon_greedy_optimistic_perf, ucb_perf, gradient_perf
end

epsilon_greedy_perf, epsilon_greedy_optimistic_perf, ucb_perf, gradient_perf =
    @timev run_parameter_studies();

#--- PLOT RESULTS

println("\nPlotting results of studies...")
plotly()

plot(HYPERPARAM_VALUES, epsilon_greedy_perf, label = "ε-greedy (x=ε)")
plot!(
    HYPERPARAM_VALUES,
    epsilon_greedy_optimistic_perf,
    label = "ε-greedy with optimistic initialization (x=Q0)",
)
plot!(HYPERPARAM_VALUES, ucb_perf, label = "upper confidence bound (x=c)")
plot!(HYPERPARAM_VALUES, gradient_perf, label = "gradient ascent (x=α)")

plot!(
    HYPERPARAM_VALUES,
    fill(white_box_baseline, length(HYPERPARAM_VALUES)),
    label = "max reward (white box)",
)
plot!(
    title = "Parameter Study for Nonstationary Problem",
    xaxis = ("hyperparameter (x) value", :log),
    yaxis = ("average reward over steps $(fld(STEPS, 2) + 1)-$STEPS"),
    legend = :outertopright,
    size = (1200, 800),
)
