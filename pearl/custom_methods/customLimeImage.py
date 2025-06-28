import numpy as np
from skimage.segmentation import slic
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
from skimage.color import gray2rgb
from typing import Callable


class CustomImageExplanation:
    def __init__(self, local_exp, segments):
        self.local_exp = local_exp
        self.segments = segments

    def get_image_and_mask(self, label, positive_only=True, num_features=5, hide_rest=False):
        mask = np.zeros(self.segments.shape, dtype=bool)
        importance = np.zeros(self.segments.shape, dtype=np.float32)

        for segment_id, weight in self.local_exp[label][:num_features]:
            if not positive_only or weight > 0:
                mask[self.segments == segment_id] = True
                importance[self.segments == segment_id] = weight

        return mask.astype(np.uint8), mask


class CustomLimeImageExplainer:
    def __init__(self, num_samples: int = 1000, num_segments: int = 50):
        self.num_samples = num_samples
        self.num_segments = num_segments

    def explain_instance(self, image: np.ndarray, classifier_fn: Callable[[np.ndarray], np.ndarray],
                         top_labels: int = 1, hide_color=0, num_features: int = 10, num_samples: int = 100):

        if image.ndim != 3 or image.shape[2] not in [1, 3]:
            raise ValueError("Input image must be HxWx3 or HxWx1")

        image = gray2rgb(image) if image.shape[2] == 1 else image
        segments = slic(image, n_segments=self.num_segments, compactness=10, sigma=1)

        N = self.num_samples
        K = np.max(segments) + 1
        samples = np.random.randint(0, 2, size=(N, K))
        samples[0, :] = 1  # original image

        perturbed_images = np.zeros((N, *image.shape), dtype=np.float32)
        for i, sample in enumerate(samples):
            mask = np.isin(segments, np.where(sample == 1)[0])
            perturbed = image.copy()
            perturbed[~mask] = hide_color
            perturbed_images[i] = perturbed

        predictions = classifier_fn(perturbed_images)
        distances = cosine_distances(samples, samples[:1]).ravel()
        weights = np.exp(-(distances ** 2) / 0.25)

        local_exp = {}
        for class_idx in range(top_labels):
            y = predictions[:, class_idx]
            model = Ridge(alpha=1.0, fit_intercept=True)
            model.fit(samples, y, sample_weight=weights)
            explanation = list(enumerate(model.coef_))
            explanation.sort(key=lambda x: abs(x[1]), reverse=True)
            local_exp[class_idx] = explanation[:num_features]

        return CustomImageExplanation(local_exp, segments)
