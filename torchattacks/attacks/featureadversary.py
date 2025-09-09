import torch
import torch.nn as nn
from ..attack import Attack # torchattacks 라이브러리의 Attack 클래스를 상속한다고 가정

class FeatureAdversary(Attack):
    """
    Creates adversarial examples by minimizing the feature distance 
    to a target guide image.
    """
    def __init__(self, model, feature_fn, eps=8/255, alpha=2/255, steps=10):
        super().__init__("FeatureAdversary", model)
        self.feature_fn = feature_fn # Function to extract features
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["targeted"] # This is a targeted attack

    def forward(self, images, guides):
        """
        Generates adversarial examples.
        
        :param images: The original images to be perturbed.
        :param guides: The target images whose features we want to mimic.
        """
        # Ensure model is in evaluation mode
        self.model.eval()

        adv_images = images.clone().detach()

        # <<< MODIFIED: Target feature calculation moved outside the loop for efficiency >>>
        # The target features are constant throughout the iterative process.
        with torch.no_grad():
            target_features = self.feature_fn(guides).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            current_features = self.feature_fn(adv_images)

            # Loss: Minimize the L2 distance (using MSE) between current and target features
            loss = nn.MSELoss()(current_features, target_features)

            # Calculate the gradient of the loss with respect to the adversarial images
            grad = torch.autograd.grad(loss, adv_images, 
                                       retain_graph=False, create_graph=False)[0]

            # Update the adversarial images via gradient descent
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            
            # Project the perturbation back into the epsilon-ball
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            
            # Ensure the final image is in the valid [0, 1] range
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images