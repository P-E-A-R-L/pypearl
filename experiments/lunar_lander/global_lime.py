import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_json_explanations(file_path):
    """
    Load JSON explanations from a file.
    """
    with open(file_path, "r") as f:
        explanations = json.load(f)
    return explanations

def aggregate_feature_importance(explanations):
    """
    Aggregate feature importance scores across all explanations.
    """
    feature_importance = {}  # Dictionary to store aggregated importance scores
    feature_counts = {}  # Dictionary to count occurrences of each feature
    feature_names = explanations[0]["feature_names"]
    
    # start from 1 to skip the feature names
    for exp in explanations[1:]:
        local_exp = exp["local_exp"]
        for class_id, features in local_exp.items():
            for feature_idx, importance in features:
                print(feature_idx)
                print(importance)
                print(feature_names[feature_idx])
                print("----")
                feature_name = feature_names[feature_idx]
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = 0.0
                    feature_counts[feature_name] = 0
                feature_importance[feature_name] += abs(importance)  # Use absolute importance
                feature_counts[feature_name] += 1

    # Calculate average importance for each feature
    for feature_name in feature_importance:
        feature_importance[feature_name] /= feature_counts[feature_name]

    return feature_importance

def plot_global_feature_importance(feature_importance, save_path=None):
    """
    Plot a bar chart of global feature importance.
    """
    features = list(feature_importance.keys())
    importance_scores = list(feature_importance.values())

    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    features = [features[i] for i in sorted_indices]
    importance_scores = [importance_scores[i] for i in sorted_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(features, importance_scores, color="skyblue")
    plt.xlabel("Feature")
    plt.ylabel("Average Importance Score")
    plt.title("Global Feature Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")
    else:
        plt.show()

def globalize_and_visualize(json_file, output_folder="global_explanation"):
    """
    Globalize and visualize feature importance from JSON explanations.
    """
    # Load JSON explanations
    explanations = load_json_explanations(json_file)

    # Aggregate feature importance
    feature_importance = aggregate_feature_importance(explanations)

    # Plot and save global feature importance
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "global_feature_importance.png")
    plot_global_feature_importance(feature_importance, save_path=output_file)
    
def aggregate_feature_importance_per_class(explanations):
    """
    Aggregate feature importance scores per class (action) and return normalized vectors.
    
    Args:
        explanations: List of LIME explanations loaded from JSON
        
    Returns:
        dict: {
            class_id: {
                'feature_names': list of feature names,
                'importance_vector': numpy array of normalized importance values
            }
        }
    """
    # Get feature names from first explanation (assuming all have same features)
    feature_names = explanations[0]["feature_names"]
    num_features = len(feature_names)
    
    # Initialize storage: class_id -> (sum_importance, count)
    class_importance = {}
    
    # Process each explanation (skip first item if it's metadata)
    for exp in explanations[1:]:
        local_exp = exp["local_exp"]
        
        for class_id, features in local_exp.items():
            # Initialize for new class_ids
            if class_id not in class_importance:
                class_importance[class_id] = {
                    'sum_importance': np.zeros(num_features),
                    'count': np.zeros(num_features)
                }
            
            # Accumulate importance values
            for feature_idx, importance in features:
                abs_importance = abs(importance)
                class_importance[class_id]['sum_importance'][feature_idx] += abs_importance
                class_importance[class_id]['count'][feature_idx] += 1
    
    # Calculate averages and normalize
    result = {}
    for class_id, data in class_importance.items():
        # Calculate average importance per feature
        avg_importance = np.where(data['count'] > 0,
                                data['sum_importance'] / data['count'],
                                0)
        
        # Normalize to sum to 1
        total = avg_importance.sum()
        normalized = avg_importance / total if total > 0 else avg_importance
        
        result[class_id] = {
            'feature_names': feature_names,
            'importance_vector': normalized
        }
    
    return result


def plot_feature_importance_per_class(class_importance, save_dir="class_importance_plots"):
    """
    Plot feature importance for each class (action) separately.
    
    Args:
        class_importance: Output from aggregate_feature_importance_per_class
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for class_id, data in class_importance.items():
        features = data['feature_names']
        importance = data['importance_vector']
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        features_sorted = [features[i] for i in sorted_idx]
        importance_sorted = importance[sorted_idx]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(features_sorted, importance_sorted, color='skyblue')
        plt.xlabel("Feature")
        plt.ylabel("Normalized Importance")
        plt.title(f"Feature Importance for Action {class_id}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(save_dir, f"action_{class_id}_importance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot for action {class_id} to {save_path}")


def globalize_and_visualize_per_class(json_file, output_folder="class_explanations"):
    """
    Process explanations to show feature importance per class/action.
    """
    # Load JSON explanations
    explanations = load_json_explanations(json_file)
    
    # Aggregate per-class importance
    class_importance = aggregate_feature_importance_per_class(explanations)
    
    # Plot and save
    plot_feature_importance_per_class(class_importance, save_dir=output_folder)
    
    # Also return the data for further analysis
    return class_importance

# Example usage
if __name__ == "__main__":
    json_file = "./explanations/lime_explanation_episode_900.json"
    globalize_and_visualize(json_file)
    
    class_importance = globalize_and_visualize_per_class(json_file)
    
    # Print summary
    print("\nFeature importance per action:")
    for class_id, data in class_importance.items():
        print(f"\nAction {class_id}:")
        for feat, imp in zip(data['feature_names'], data['importance_vector']):
            print(f"  {feat}: {imp:.4f}")
