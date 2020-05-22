#=
Exercise 2.11
Make a figure analogous to Figure 2.6 for the nonstationary case outlined in Exercise 2.5.
Include the constant-step-size ε-greedy algorithm with α = 0.1. Use runs of 200,000 steps and,
as a performance measure for each algorithm and parameter setting, use the average reward over the
last 100,000 steps.
_____

The algorithms I will include are:
- constant-step-size ε-greedy with α = 0.1 (varying ε)
- constant-step-size ε-greedy with optimistic initialization and α = 0.1 (varying Q0)
- gradient (varying α) - upper confidence bound (varying c)

I will vary each of these parameters from 1/128 to 4 by factors of 2.
=#

include("..\\tools\\k-armed-bandits\\k-armed-bandits.jl")

using .KArmedBandits
using Plots
using Statistics: mean

#--- SETTINGS

println("Threads: $(Threads.nthreads())")

# simulation parameters
const K = 10
const SIMULATIONS = 50#2000
const STEPS = 100#200000

# hyperparameter values
const HYPERPARAM_VALUES = [2.0^n for n in -7:2]

#--- MEASURE PERFORMANCE

function get_model_perf(model)::Float64
    rewards, optimality = @time test_model(model)
    # average reward over second half of simulation
    return mean(rewards[(fld(STEPS, 2) + 1):STEPS])
end

#--- GET WHITEBOX BASELINE

println("Getting white box baseline...")
white_box_baseline = get_model_perf(WhiteBoxModel())

#--- RUN PARAMETER STUDIES

function run_parameter_studies()
    epsilon_greedy_perf = []
    epsilon_greedy_optimistic_perf = []
    ucb_perf = []
    gradient_perf = []

    println("Running parameter studies...")

    for test_val in HYPERPARAM_VALUES
        # intitialize models with the particular hyperparameter value
        # then run simulations and record performance

        # ε-greedy, varying ε (exploration)
        if test_val <= 1 # epsilon is a probability
            println("\nTesting ε-greedy algorithm (ε = $test_val)...")
            model = EpsilonGreedyModel(
                epsilon = test_val,
                alpha = 0.1,
                actions = K,
                initial_estimate = 0.0,
            )
            perf = get_model_perf(model)
            push!(epsilon_greedy_perf, perf)
        else
            push!(epsilon_greedy_perf, missing)
        end

        # optimistic ε-greedy, varying Q0 (initialization)
        println("\nTesting ε-greedy algorithm with optimistic initializations (Q0 = $test_val)...")
        model =
            EpsilonGreedyModel(epsilon = 0.1, alpha = 0.1, actions = K, initial_estimate = test_val)
        perf = get_model_perf(model)
        push!(epsilon_greedy_optimistic_perf, perf)

        # upper confidence bound, varying c (confidence)
        println("\nTesting upper confidence bound (UCB) algorithm (c = $test_val)...")
        model = UcbModel(c = test_val, alpha = 0.1, actions = K)
        perf = get_model_perf(model)
        push!(ucb_perf, perf)

        # stochastic gradient ascent, varying α (step size)
        println("\nTesting stochastic gradient ascent algorithm (α = $test_val)...")
        model = GradientModel(alpha = test_val, actions = K)
        perf = get_model_perf(model)
        push!(gradient_perf, perf)
    end

    println("\nParameter studies complete!")
    return epsilon_greedy_perf, epsilon_greedy_optimistic_perf, ucb_perf, gradient_perf
end

epsilon_greedy_perf, epsilon_greedy_optimistic_perf, ucb_perf, gradient_perf =
    @timev run_parameter_studies();

#--- PLOT RESULTS

println("\nPlotting results of studies...")
plotly()

plot(
    title = "Parameter Study for Nonstationary K-Armed Bandit Problem",
    xaxis = ("Hyperparameter (x) Value", :log),
    yaxis = ("Average Reward over Steps $(fld(STEPS, 2) + 1)-$STEPS"),
    legend = :outertopright,
    size = (900, 600),
)

plot!(
    HYPERPARAM_VALUES,
    fill(white_box_baseline, length(HYPERPARAM_VALUES)),
    label = "max reward (white box model)",
)

plot!(HYPERPARAM_VALUES, epsilon_greedy_perf, label = "ε-greedy (x=ε)")
plot!(HYPERPARAM_VALUES, epsilon_greedy_optimistic_perf, label = "optimistic ε-greedy (x=Q0)")
plot!(HYPERPARAM_VALUES, ucb_perf, label = "upper confidence bound (x=c)")
plot!(HYPERPARAM_VALUES, gradient_perf, label = "gradient ascent (x=α)")

gui()
