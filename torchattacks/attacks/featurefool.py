# adv_sample_crafter/torchattacks/attacks/featurefool.py
import torch
import torch.nn as nn
from ..attack import Attack

class FeatureFool(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, lambda_reg=1e-5):
        super().__init__("FeatureFool", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.lambda_reg = lambda_reg
        self.supported_mode = ["default"] # Untargeted

    def _get_features(self, images):
        """
        Helper to get features from the model.
        Assumes the model's forward method supports `return_features=True`
        and returns a tuple (logits, features).
        """
        # This check is important for ensuring the model is compatible.
        if not (hasattr(self.model, 'forward') and 'return_features' in self.model.forward.__code__.co_varnames):
            raise ValueError("The model for FeatureFool must support `return_features=True` in its forward method.")
        
        _, features = self.model(images, return_features=True)
        return features

    def forward(self, images, labels):
        # images: input batch (x_batch in cloudleck.py)
        # labels: s_pred (student's prediction) - not directly used in this feature-based attack

        adv_images = images.clone().detach()
        
        # Use the internal, potentially updated, model to get features
        original_features = self._get_features(images).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            current_features = self._get_features(adv_images)

            # Loss: Maximize distance between current features and original features
            feature_distance = nn.MSELoss()(current_features, original_features)
            perturbation_distance = nn.MSELoss()(adv_images, images)
            # Gradient Ascent to maximize this loss
            loss = feature_distance - self.lambda_reg * perturbation_distance 

            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
