using Random: rand

export EpsilonGreedyModel

#--- DEFAULT SETTINGS

const EPSILON = 0.1 # probability of exploration
const INITIAL_ESTIMATE = 0.0 # starting value estimate

#--- ε-GREEDY ALGORITHM

mutable struct EpsilonGreedyModel
    epsilon::Float64
    alpha::Float64
    actions::Int64
    estimated_values::Array{Float64, 1}
end

EpsilonGreedyModel(;
    epsilon::Float64 = EPSILON,
    alpha::Float64 = ALPHA,
    actions::Int64 = K,
    initial_estimate::Float64 = INITIAL_ESTIMATE,
) = EpsilonGreedyModel(epsilon, alpha, actions, fill(initial_estimate, actions))

function choose_action(model::EpsilonGreedyModel)::Int64
    # ε-greedy algorithm
    return if rand() < model.epsilon
        # explore
        rand(1:(model.actions))
    else
        # exploit
        argmax(model.estimated_values)
    end
end

function run_step!(model::EpsilonGreedyModel, bandit::KArmedBandit)::Tuple{Int64, Float64}
    # pick action, get reward, update model
    action = choose_action(model)
    reward = get_reward(bandit, action)
    update_estimated_values!(model.estimated_values, action, reward, model.alpha)

    # track performance
    return action, reward
end;
