import lime
import lime.lime_tabular
import numpy as np
import torch
import json

def explain_action(agent, observation, num_features=8, file_name="lime_explanation.html", return_html=False):
    """
    Uses LIME to explain the decision-making process of the agent for a given observation.
    """
    def action_probability_fn(states):
        """
        Function that returns the action probabilities given a batch of states.
        Used by LIME to sample and train an interpretable model.
        """
        states = torch.tensor(states, dtype=torch.float32).to(agent.device)
        with torch.no_grad():
            action_probs = agent.policy(states)
        return action_probs.cpu().numpy()  # Return numpy array of shape (num_samples, num_actions)

    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array([observation]),  
        class_names = ["NOOP", "FIRE_LEFT", "FIRE_MAIN", "FIRE_RIGHT"],
        feature_names=["x_position", "y_position", "x_velocity", "y_velocity",
                    "angle", "angular_velocity", "left_leg_contact", "right_leg_contact"],  
        mode="classification",  
        discretize_continuous=False,
        sample_around_instance=True,
    )

    # Explain the action
    exp = explainer.explain_instance(
        observation, action_probability_fn, num_features=num_features, top_labels=1
    )

    # Convert explanation to JSON
    explanation_json = {
        "observation": observation.tolist(),
        #"feature_names": exp.domain_mapper.feature_names,
        "local_exp": exp.local_exp,
        "predict_proba": exp.predict_proba.tolist(),
        "score": exp.score,
    }

    if return_html:
        return exp.as_html(), explanation_json
    else:
        exp.save_to_file(file_name)
        print(f"Explanation saved to {file_name}")
        return explanation_json