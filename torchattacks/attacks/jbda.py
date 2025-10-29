import torch
import torch.nn as nn

from ..attack import Attack

class JBDA(Attack):
    r"""
    Implements Jacobian-based Dataset Augmentation (JBDA) from Papernot et al. (2017).
    Generates new data points by moving input samples in the direction indicated
    by the sign of the Jacobian of the substitute model, corresponding to the
    oracle's predicted label for that sample.

    Args:
        substitute_model (nn.Module): The substitute model (F) used to compute the Jacobian.
        lambda_val (float): The step size (lambda in the paper) for augmentation. (Default: 0.1)
        micro_batch_size (int or None): Micro-batch size for memory efficiency during gradient calculation.
                                        If None, uses the full batch size. (Default: 64)
    """

    def __init__(self, model, lambda_val=0.1, micro_batch_size=64):
        # Note: We pass substitute_model, but the parent class saves it as self.model
        super().__init__("JacobianAugmentation", model)
        self.lambda_val = float(lambda_val)
        self.micro_batch_size = None if micro_batch_size is None else int(micro_batch_size)

    def forward(self, images, oracle_labels):
        device = self.device
        images = images.to(device)
        oracle_labels = oracle_labels.to(device).long()

        N = images.size(0)
        mb = self.micro_batch_size or N
        augmented_images = torch.empty_like(images)

        was_training = self.model.training
        self.model.eval()  # BN/Dropout 고정; grad는 별개

        for s in range(0, N, mb):
            e = min(s + mb, N)
            if e <= s:
                continue

            # 1) 입력: 그래디언트 확실히 켬 (detach 쓰지 않음)
            chunk_imgs = images[s:e].clone().requires_grad_(True)
            chunk_oracle_labels = oracle_labels[s:e]

            # 2) 그래드 활성 컨텍스트에서 모델 호출
            with torch.enable_grad():
                outputs = self.model(chunk_imgs)  # [B,C] 혹은 [B,1]

            # ---- 진단용 점검 (실전에서는 assert→if+raise로 바꿔도 됨) ----
            assert torch.is_grad_enabled(), "Grad is disabled (no_grad/inference_mode?)"
            assert chunk_imgs.requires_grad, "chunk_imgs.requires_grad is False"
            assert isinstance(outputs, torch.Tensor), "model output is not a Tensor"
            assert outputs.requires_grad, "model outputs detached in forward (check for .detach() or no_grad)"
            # ------------------------------------------------------------

            B, C = outputs.shape[0], outputs.shape[1]
            if B == 0:
                continue

            # 3) 로짓 합에 대한 grad = 배치 개별 grad와 동일
            if C >= 2:
                # 멀티클래스: 선택 클래스 로짓들을 모아서 한 번에 grad
                relevant_logits = outputs.gather(1, chunk_oracle_labels.view(-1,1)).squeeze(1)  # [B]
                # 텐서 상태 유지(.sum()), .item() 절대 금지
                grad = torch.autograd.grad(relevant_logits.sum(), chunk_imgs, create_graph=False, retain_graph=False)[0]
            else:
                # 이진(단일 로짓): 라벨로 부호만 다르게
                single_logit = outputs.squeeze(1)  # [B]
                g = torch.autograd.grad(single_logit.sum(), chunk_imgs, create_graph=False, retain_graph=False)[0]
                # y∈{0,1} → 0:-1, 1:+1
                sign_mult = (2 * chunk_oracle_labels.float() - 1.0).view(-1,1,1,1)
                grad = g * sign_mult

            # 4) 증강
            with torch.no_grad():
                delta = self.lambda_val * torch.sign(grad)
                chunk_aug = chunk_imgs + delta
                chunk_aug.clamp_(0.0, 1.0)
                augmented_images[s:e].copy_(chunk_aug)

        if was_training:
            self.model.train()

        return augmented_images
