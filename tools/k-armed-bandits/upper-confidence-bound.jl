export UcbModel

#--- DEFAULT SETTINGS

const C = 1.0 # how much confidence you want

#--- UPPER CONFIDENCE BOUND ALGORITHM

mutable struct UcbModel
    c::Float64
    alpha::Float64
    actions::Int64
    estimated_values::Array{Float64, 1}
    frequencies::Array{Int64, 1}
    step::Int64
end

UcbModel(; c::Float64 = C, alpha::Float64 = ALPHA, actions::Int64 = K) =
    UcbModel(c, alpha, actions, zeros(actions), zeros(actions), 0)

function choose_action(model::UcbModel)::Int64
    upper_bounds = [
        model.estimated_values[action] +
        model.c * sqrt(log(model.step) / model.frequencies[action])
        for action in 1:(model.actions)
    ]
    action = argmax(upper_bounds)
    model.frequencies[action] += 1
    return action
end

function run_step!(model::UcbModel, bandit::KArmedBandit)::Tuple{Int64, Float64}
    model.step += 1

    # pick action, get reward, update model
    action = choose_action(model)
    reward = get_reward(bandit, action)
    update_estimated_values!(model.estimated_values, action, reward, model.alpha)

    # track performance
    return action, reward
end;
