using StatsBase: sample, Weights

export GradientModel

#--- GRADIENT ALGORITHM

mutable struct GradientModel
    alpha::Float64
    actions::Int64
    preferences::Array{Float64, 1}
    baseline::Float64
end

GradientModel(; alpha::Float64 = ALPHA, actions::Int64 = K) =
    GradientModel(alpha, actions, zeros(actions), NaN)

function get_probabilities(preferences::Array{Float64, 1})::Array{Float64, 1}
    # non-normalized probabilities are e^H, where H is the preference of an action
    probabilities = exp.(preferences)
    return probabilities ./ sum(probabilities)
end

function choose_action(probabilities::Array{Float64, 1})::Int64
    return sample(Weights(probabilities))
end

function update_baseline!(model::GradientModel, reward::Float64)
    # incremental update with constant step size (since nonstationary)
    return model.baseline += model.alpha * (reward - model.baseline)
end

function update_preferences!(
    model::GradientModel,
    probabilities::Array{Float64, 1},
    chosen_action::Int64,
    reward::Float64,
)
    step = model.alpha * (reward - model.baseline)
    return for action in 1:(model.actions)
        if action == chosen_action
            model.preferences[action] += step * (1 - probabilities[action])
        else
            model.preferences[action] -= step * probabilities[action]
        end
    end
end

function run_step!(model::GradientModel, bandit::KArmedBandit)::Tuple{Int64, Float64}
    # pick action, get reward, update model
    probabilities = get_probabilities(model.preferences)
    action = choose_action(probabilities)

    reward = get_reward(bandit, action)
    if model.baseline != model.baseline # NaN, uninitialized
        model.baseline = reward
    end

    update_preferences!(model, probabilities, action, reward)
    update_baseline!(model, reward)

    # track performance
    return action, reward
end;
