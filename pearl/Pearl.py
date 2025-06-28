import enum
from typing import Callable

import numpy as np

from pearl.agent import RLAgent
from pearl.method import ExplainabilityMethod
from pearl.env import RLEnvironment


class ScorePolicy(enum.Enum):
    PEARL  = 0  # use pearl for evaluation
    REWARD = 1  # use reward for evaluation
    MIXED  = 2  # use both, with weights


class StepPolicy(enum.Enum):
    INDEPENDENT = 1 # each agent steps using its own decision-making
    BEST_AGENT  = 2 # all agents forward their observations to the best agent and choose his choice
    WORST_AGENT = 3 # all agents forward their observations to the worst agent and choose his choice
    RANDOM      = 4 # choice is selected randomly


class AgentState:
    steps_ep     = 0
    steps_total  = 0
    reward_total = 0
    curr_ep      = 0
    done = False


def eval(
        agents: list[RLAgent],
        env_factory: Callable[[RLAgent], RLEnvironment],
        methods_factory: list[Callable[[RLAgent], ExplainabilityMethod]],
        methods_weights: list[float],
        scorePolicy: ScorePolicy = ScorePolicy.PEARL,
        stepPolicy: StepPolicy   = StepPolicy.INDEPENDENT,
        stepPolicy_weights: list[float] = None,
        max_steps_total: int = -1,
        max_steps_ep   : int = -1,
        max_episodes   : int = 2,
) -> list[float]:
    assert env_factory is not None, "Environment factory cannot be null."
    assert methods_factory is not None, "Methods factory cannot be null."

    assert methods_weights is not None, "Methods weights cannot be null."
    assert len(methods_weights) == len(methods_factory) is not None, "The list of methods weights should be of the same size as the list of methods."

    assert stepPolicy is not None, "StepPolicy cannot be null."
    assert scorePolicy is not None, "ScorePolicy cannot be null."
    assert scorePolicy != ScorePolicy.MIXED or stepPolicy_weights is not None, "StepPolicy weights cannot be null."
    assert scorePolicy != ScorePolicy.MIXED or len(stepPolicy_weights) == 2, "StepPolicy weights should be of size 2"

    assert max_steps_total != -1 or max_episodes != -1, "Termination condition is not defined."

    envs = [env_factory(agent) for agent in agents]
    states = [AgentState() for _ in agents]
    methods = [[fact(agent) for fact in methods_factory] for agent in agents]

    for agent_idx in range(len(agents)):
        env = envs[agent_idx]
        env.reset()
        for method in methods[agent_idx]:
            method.set(env)
            method.prepare(agents[agent_idx])

    scores = np.zeros(len(agents))

    terminate = False

    for state in states:
        state.done = False
        state.reward_total = 0
        state.steps_ep = 0
        state.steps_total = 0
        state.curr_ep = 1

    while not terminate:
        scores_delta = np.zeros(len(agents))
        for agent_idx in range(len(agents)):
            agent   = agents[agent_idx]
            env     = envs[agent_idx]
            state   = states[agent_idx]

            # (state.steps_ep >= max_steps_ep >= 0)
            if state.done or (state.steps_total >= max_steps_total >= 0):
                continue

            # select an action
            obs = env.get_observations()
            actions = env.get_available_actions()
            action = 0

            if stepPolicy == StepPolicy.RANDOM:
                action = np.random.choice(len(actions))
            elif stepPolicy == StepPolicy.BEST_AGENT:
                normalized_scores = [scores[i] / states[i].steps_total if states[i].steps_total else 0 for i in range(len(agents))]
                best_agent = np.argmax(normalized_scores)
                target_agent = agents[best_agent]
                props = target_agent.predict(obs)
                action = np.argmax(props)
            elif stepPolicy == StepPolicy.WORST_AGENT:
                normalized_scores = [scores[i] / states[i].steps_total if states[i].steps_total else 0 for i in range(len(agents))]
                best_agent = np.argmin(normalized_scores)
                target_agent = agents[best_agent]
                props = target_agent.predict(obs)
                action = np.argmax(props)
            elif stepPolicy == StepPolicy.INDEPENDENT:
                props = agent.predict(obs)
                action = np.argmax(props)

            # step into env with action
            for method in methods[agent_idx]:
                method.onStep(action)

            _, reward_dict, terminated, truncated, info = env.step(action)
            state.reward_total += reward_dict["reward"]
            state.steps_total += 1
            state.steps_ep    += 1
            state.done         = terminated or truncated or state.steps_ep >= max_steps_ep >= 0

            for method in methods[agent_idx]:
                method.onStepAfter(action, reward_dict, terminated or truncated, info)


            move_score = 0
            if scorePolicy == ScorePolicy.PEARL:
                for method_id in range(len(methods[agent_idx])):
                    score = methods[agent_idx][method_id].value(obs)
                    move_score += score * methods_weights[method_id]
                    # print(f"move_score[{agent_idx}, {method_id}]: {move_score}")
            elif scorePolicy == ScorePolicy.REWARD:
                move_score += reward_dict["reward"][0]
            elif scorePolicy == ScorePolicy.MIXED:
                for method_id in range(len(methods[agent_idx])):
                    score = methods[agent_idx][method_id].value(obs)
                    move_score += score * methods_weights[method_id]
                move_score = move_score * stepPolicy_weights[0] + reward_dict["reward"][0] * stepPolicy_weights[1]

            scores_delta[agent_idx] = move_score

        scores += scores_delta

        terminate = True
        for agent_idx in range(len(agents)):
            state = states[agent_idx]
            if state.done:
                if state.steps_total >= max_steps_total >= 0 or state.curr_ep >= max_episodes >= 0:
                    continue

                state.curr_ep += 1
                state.steps_ep = 0
                state.done = False
                envs[agent_idx].reset()
                terminate = False
            else:
                if state.steps_total >= max_steps_total >= 0:
                    continue
                terminate = False

    normalized_scores = [scores[i] / states[i].steps_total if states[i].steps_total else 0 for i in range(len(agents))]
    return normalized_scores