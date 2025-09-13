import torch
import torch.nn as nn

from ..attack import Attack


class JBDA(Attack):
    r"""
    JBDA (Jacobian-based Data Augmentation) - Targeted iterative version with memory-optimized micro-batching.

    - Distance Measure : Linf
    - 타깃형 업데이트: cost = -CE(target) (또는 target logit 최대화)
    - L_inf 프로젝션 + [0,1] 클램프
    - random_start 지원
    - **micro_batch_size**로 그래디언트/활성값을 chunk 단위로 계산하여 GPU 메모리 절감

    Args:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): random init in [-eps, eps]. (Default: False)
        use_target_logit (bool): CE 대신 target logit 최대화. (Default: False)
        micro_batch_size (int or None): 메모리 절감을 위한 마이크로배치 크기. None이면 전체 배치. (Default: 64)
    """

    def __init__(self, model, eps=1, alpha=2/255, steps=10,
                 random_start=False, use_target_logit=False, micro_batch_size=64):
        super().__init__("JBDA", model)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.random_start = bool(random_start)
        self.use_target_logit = bool(use_target_logit)
        self.micro_batch_size = None if micro_batch_size is None else int(micro_batch_size)

        self.supported_mode = ["targeted"]

    def forward(self, images, labels, eps=None):
        r"""
        Overridden.

        Args:
            images: (N,C,H,W) in [0,1]
            labels: (N,) long
            eps: float or per-sample tensor (N,1,1,1). None이면 self.eps
        """
        device = self.device
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device).long()

        # Target labels (PGD 스타일)
        if not self.targeted:
            target_labels = labels
        else:
            target_labels = self.get_target_label(images, labels)

        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        # 동적 eps 처리
        if eps is None:
            eps_val = self.eps
        else:
            eps_val = eps.to(device) if torch.is_tensor(eps) else float(eps)

        # Random start
        if self.random_start:
            if torch.is_tensor(eps_val):
                noise = torch.empty_like(adv_images).uniform_(-1.0, 1.0)
                adv_images = adv_images + noise * eps_val
            else:
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps_val, eps_val)
            adv_images = adv_images.clamp(0.0, 1.0).detach()

        N = images.size(0)
        mb = self.micro_batch_size or N  # None이면 전체 배치
        was_training = self.model.training
        self.model.eval()  # 불필요한 상태 변화(BN/Dropout) 방지

        # 전체 step 루프
        for _ in range(self.steps):
            # 선택 사항: 완료된 샘플 스킵을 위한 마스크 (메모리 절감 + 약간의 속도 향상)
            # 타깃 달성 여부를 확인하려면 한 번의 라운드가 필요하므로 chunk마다 체크
            done_mask = torch.zeros(N, dtype=torch.bool, device=device)

            # 1) 타깃 달성 여부 갱신 (마이크로배치로 예측만)
            with torch.no_grad():
                for s in range(0, N, mb):
                    e = min(s + mb, N)
                    preds = self.get_logits(adv_images[s:e]).argmax(dim=1)
                    done_mask[s:e] = (preds == target_labels[s:e])

            if done_mask.all():
                break

            # 2) 그라디언트 계산 및 업데이트 (마이크로배치)
            for s in range(0, N, mb):
                e = min(s + mb, N)

                # 이미 타깃 달성한 샘플은 스킵
                if done_mask[s:e].all():
                    continue

                # 슬라이스 복사본에만 requires_grad 걸기 (leaf tensor)
                chunk_adv = adv_images[s:e].detach().requires_grad_(True)
                chunk_imgs = images[s:e]
                chunk_targets = target_labels[s:e]

                outputs = self.get_logits(chunk_adv)

                if self.use_target_logit:
                    tgt_logit = outputs.gather(1, chunk_targets.view(-1, 1)).squeeze(1)
                    cost = -tgt_logit.mean()
                else:
                    cost = -loss_fn(outputs, chunk_targets)

                grad = torch.autograd.grad(cost, chunk_adv, retain_graph=False, create_graph=False)[0]

                # 업데이트
                with torch.no_grad():
                    chunk_adv = chunk_adv + self.alpha * grad.sign()

                    # L_inf 프로젝션
                    if torch.is_tensor(eps_val):
                        eps_slice = eps_val[s:e]
                        chunk_adv = torch.max(torch.min(chunk_adv, chunk_imgs + eps_slice), chunk_imgs - eps_slice)
                    else:
                        delta = torch.clamp(chunk_adv - chunk_imgs, min=-self.eps if eps is None else -eps_val,
                                            max=self.eps if eps is None else eps_val)
                        chunk_adv = chunk_imgs + delta

                    # 픽셀 범위
                    chunk_adv.clamp_(0.0, 1.0)

                    # 원본 텐서에 반영 (in-place copy to slice)
                    adv_images[s:e].copy_(chunk_adv)

                # 즉시 그래프/중간값 해제
                del chunk_adv, outputs, grad, cost

            # (선택) 메모리 여유 없으면 캐시 정리
            # torch.cuda.empty_cache()

        if was_training:
            self.model.train()

        return adv_images
