# adv_sample_crafter/torchattacks/attacks/featurefool.py
import torch
import torch.nn as nn
from ..attack import Attack

class FeatureFool(Attack):
    def __init__(self, model, feature_fn, eps=8/255, alpha=2/255, steps=10, lambda_reg=1e-5):
        super().__init__("FeatureFool", model)
        self.feature_fn = feature_fn # Function to extract features
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.lambda_reg = lambda_reg
        self.supported_mode = ["default"] # Untargeted

    def forward(self, images, labels):
        # images: input batch (x_batch in cloudleck.py)
        # labels: s_pred (student's prediction) - not directly used in this feature-based attack, but kept for compatibility with Attack.forward signature.

        adv_images = images.clone().detach()
        original_features = self.feature_fn(images).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            current_features = self.feature_fn(adv_images)

            # Loss: Maximize distance between current features and original features
            # while minimizing perturbation.
            # Loss = - MSE(current_features, original_features) + lambda_reg * MSE(adv_images, images)
            # The paper mentions "minimal perturbation" and "pushing towards decision boundary".
            # This loss pushes features away from original, and regularizes perturbation.

            # 목표: max (feature_distance - lambda * perturbation_distance)
            feature_distance = nn.MSELoss()(current_features, original_features)
            perturbation_distance = nn.MSELoss()(adv_images, images)
            # Gradient Ascent로 이 loss를 최대화하면 원하는 목표를 달성
            loss = feature_distance - self.lambda_reg * perturbation_distance 

            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images