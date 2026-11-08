import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from fsop.config import load_config
from fsop.train import losses
from fsop.train.device import autocast_ctx
from fsop.train.trainer import TrainCfg, Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/train/spark_skidpad_centering.yaml')
    parser.add_argument('--dataset', default='data/sim/spark_skidpad_centering')
    parser.add_argument('--checkpoint', default='runs/spark_skidpad/final.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.batches < 1:
        parser.error('--batches must be positive')
    cfg = load_config(TrainCfg, args.cfg)
    cfg.sim_root = args.dataset
    cfg.init_ckpt = args.checkpoint
    cfg.w_steer = 1.0
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.name = Path(args.output).stem + '_calibration'
    cfg.out_dir = str(Path(args.output).parent)
    trainer = Trainer(cfg)
    try:
        trainer.model.eval()
        parameters = [p for p in trainer.model.base.policy.parameters() if p.requires_grad]
        plan_sq = steer_sq = dot = 0.0
        error_sum = expert_plan_error_sum = 0.0
        count = samples = labeled_batches = 0
        iterator = iter(trainer._loader(trainer.train_ds, True))
        for _ in range(args.batches):
            try:
                batch = trainer._to(next(iterator))
            except StopIteration:
                iterator = iter(trainer._loader(trainer.train_ds, True))
                batch = trainer._to(next(iterator))
            mask = batch['steer_valid'].bool()
            n = int(mask.sum().item())
            samples += len(mask)
            if n == 0:
                continue
            labeled_batches += 1
            count += n
            target = batch['steer_target']
            geometry = (batch['speed'], batch['vehicle_lr'], batch['vehicle_wheelbase'])
            with autocast_ctx(trainer.device):
                out = trainer.forward(batch)
                pred = losses.pp_steer_from_op_plan(out['plan_mu'], *geometry)
                expert_plan = losses.pp_steer_from_op_plan(batch['plan'], *geometry)
                plan_loss = cfg.w_plan * losses.plan_nll(out['plan_mu'], out['plan_logstd'], batch['plan'])
                steer_loss = losses.steering_loss(out['plan_mu'], batch)
            error_sum += float((pred[mask] - target[mask]).abs().sum().detach())
            expert_plan_error_sum += float((expert_plan[mask] - target[mask]).abs().sum().detach())
            gp = torch.autograd.grad(plan_loss, parameters, retain_graph=True, allow_unused=True)
            gs = torch.autograd.grad(steer_loss, parameters, allow_unused=True)
            for p, s in zip(gp, gs, strict=True):
                if p is not None:
                    plan_sq += float(p.float().square().sum())
                if s is not None:
                    steer_sq += float(s.float().square().sum())
                if p is not None and s is not None:
                    dot += float((p.float() * s.float()).sum())
        if not count or plan_sq <= 0 or steer_sq <= 0 or not all(map(math.isfinite, (plan_sq, steer_sq, dot))):
            raise ValueError('calibration needs finite nonzero plan and steering gradients on labeled samples')
        checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        report = {
            'checkpoint': args.checkpoint, 'checkpoint_sha256': checkpoint_hash,
            'dataset': args.dataset, 'config': args.cfg, 'seed': args.seed,
            'batches': args.batches, 'labeled_batches': labeled_batches,
            'samples': samples, 'labeled_samples': count,
            'labeled_episodes': sum(ep.steering is not None for ep in trainer.train_ds.eps),
            'training_episodes': len(trainer.train_ds.eps),
            'plan_gradient_norm_pooled': math.sqrt(plan_sq),
            'steer_gradient_norm_pooled_unweighted': math.sqrt(steer_sq),
            'gradient_cosine_pooled': dot / math.sqrt(plan_sq * steer_sq),
            'recommended_w_steer_for_20_percent': 0.2 * math.sqrt(plan_sq / steer_sq),
            'model_steer_mae_rad': error_sum / count,
            'expert_plan_steer_mae_rad': expert_plan_error_sum / count,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        trainer.tb.close()


if __name__ == '__main__':
    main()
