import torch
import torch.nn as nn

from ..attack import Attack


class JBDA(Attack):
    """
    Jacobian-based Data Augmentation (JBDA)
    This is a targeted iterative method that takes steps in the direction of the sign of the gradient of the target class logit.
    This implementation is based on the method described in Papernot et al.

    Arguments:
        model (nn.Module): model to attack.
        alpha (float): step size. (Default: 0.1)
        steps (int): number of steps. (Default: 10)
    """

    def __init__(self, model, alpha=0.1, steps=10):
        super().__init__("JBDA", model)
        self.alpha = alpha
        self.steps = steps
        self.targeted = True  # JBDA is inherently a targeted attack
        self._supported_mode = ['targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        # Iterating over each image in the batch as the early exit condition is per-sample
        for i in range(len(images)):
            img = images[i:i + 1].clone().detach()
            target_label = labels[i:i + 1]

            for _ in range(self.steps):
                img.requires_grad = True
                logits = self.model(img)

                # Early exit: if the model's prediction already matches the target, stop.
                if logits.argmax(dim=1) == target_label:
                    break

                # Use CrossEntropyLoss to get the gradient towards the target label
                loss = nn.CrossEntropyLoss()(logits, target_label)

                # Get the gradient of the loss with respect to the input image
                grad = torch.autograd.grad(loss, img,
                                           retain_graph=False, create_graph=False)[0]

                # Take a step in the direction of the sign of the gradient
                adv_img_step = img + self.alpha * grad.sign()
                # Clamp the image to the valid [0, 1] range
                img = torch.clamp(adv_img_step, min=0, max=1).detach()

            adv_images[i] = img

        return adv_images
