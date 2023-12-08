import os
import copy
import functools
import logging
import torch
from torch.optim import AdamW
import numpy as np
import blobfile as bf
import time
from datetime import datetime
from tqdm import tqdm


from utils.fp16 import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from utils.basic import update_ema
from utils.metric import PredictionMetrics


class Trainer:
    def __init__(self, args, device, model, diffusion,
                 train_loader=None, test_loader=None, schedule_sampler=None):
        self.args = args
        self.device = device
        self.model = model
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.schedule_sampler = schedule_sampler
        
        self.save_interval = args.save_interval
        self.test_interval = args.test_interval

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.gradient_clipping = args.gradient_clipping
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.optimizer = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        self.step = 0
        self.learning_steps = 0
        self.ema_rate = (
            [args.ema_rate]
            if isinstance(args.ema_rate, float)
            else [float(x) for x in args.ema_rate.split(",")]
        )
        self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]  
        self.use_fp16 = args.use_fp16
        self.fp16_scale_growth = args.fp16_scale_growth
        self.lg_loss_scale = 20.0
        if self.use_fp16:
            self._setup_fp16()

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def train(self):
        for epoch in range(0, self.args.num_epoch):
            for step, batch in enumerate(tqdm(self.train_loader)):
                self.step += 1
                batch = batch.to(self.device)
                self.train_step(batch, step, epoch)

            if self.test_loader is not None:
                for step, batch_eval in enumerate(tqdm(self.test_loader)):
                    batch_eval = batch_eval.to(self.device)
                    self.forward_only(batch_eval, step, epoch)
                print('eval on validation set')
                self.test()

            # Save the last checkpoint
            if epoch % self.save_interval == 0:
                current = datetime.now()
                logging.info("Current save time: {}".format(current))
                logging.info('It is the {}-th epoch, saved'.format(epoch))
                self.save(epoch)

    def train_step(self, batch, step, epoch):
        self.forward_backward(batch, step, epoch)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()

    def forward_backward(self, batch, step, epoch):
        zero_grad(self.model_params)
        t, weights = self.schedule_sampler.sample(batch.num_graphs, self.device)
        compute_losses = functools.partial(
            self.diffusion.training_losses, 
            self.model, batch, t, weights)

        losses = compute_losses()
        if step % 50 == 0:
            logging.info("train epoch:{}, step: {}, loss: {}".format(epoch, step, losses.item()))
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (losses * loss_scale).backward()
        else:
            losses.backward()

    def forward_only(self, batch, step, epoch):
        with torch.no_grad():
            zero_grad(self.model_params)
            t, weights = self.schedule_sampler.sample(batch.num_graphs, self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses, 
                self.model, batch, t, weights)

            losses = compute_losses()
            if step % 50 == 0:
                logging.info("eval epoch:{}, step: {}, loss: {}".format(epoch, step, losses.item()))

    def test(self):
        met = PredictionMetrics()
        results = {}
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = batch.to(self.device)
                t, weights = self.schedule_sampler.sample(batch.num_graphs, self.device)
                t.fill_(1)
                input = self.model.token_embed(batch.embed_type, batch.x)

                batch_size, seq_len = batch.num_graphs, batch.num_nodes_per_graph
                input = input.view(batch_size, seq_len, -1)
                output, _ = self.model(
                    input, t, 
                    batch.attn_bias_type, 
                    batch.pred_type, 
                    batch.embed_type, 
                    batch.x_d,
                    isdiffu=True
                ) 
                feat = output.view(batch_size*seq_len, -1)
                x_pred, _, _= self.model.answer_queries(feat, batch)

                met.digest(x_pred, batch.x_ans,
                        weight=batch.x_pred_weight if hasattr(batch, 'x_pred_weight') else None)
         
            results['mrr'] = met.MRR()
            results['hit1'] = met.hits_at(1)
            results['hit3'] = met.hits_at(3)
            results['hit10'] = met.hits_at(10)
            logging.info('[{}]: MRR: {:.5}, hits@1: {:.5}, hits@3: {:.5}, hits@10: {:.5}'.format(
                self.args.split, results['mrr'], results['hit1'],results['hit3'],results['hit10']))
            current = datetime.now()
            logging.info("Current test time: {}".format(current))
            return met.MRR()

    def test_whole(self):        
        time_start = time.time()
        total = 0 

        met = PredictionMetrics()
        results = {}

        for batch in tqdm(self.test_loader):
            total += batch.num_graphs
            batch = batch.to(self.device)
            batch_size, seq_len = batch.num_graphs, batch.num_nodes_per_graph
            x_start = self.model.token_embed(batch.embed_type, batch.x) 
            x_start = x_start.view(batch_size, seq_len, -1)

            noise = torch.randn_like(x_start)
            x_mask = batch.pred_type.view(batch_size, seq_len)
            x_mask = torch.broadcast_to(x_mask.unsqueeze(dim=-1), x_start.shape)
            x_noised = torch.where(x_mask == 0, x_start, noise)
            model_kwargs = { 
                "attn_bias_type": batch.attn_bias_type,
                "pred_type": batch.pred_type,
                "node_type": batch.embed_type, 
                "node_degree":batch.x_d,
                "isdiffu": True                    
            }

            step_gap = 1
            sample_fn = self.diffusion.p_sample_loop
            sample_shape = (batch_size, seq_len, self.args.hidden_dim)
            samples = sample_fn(
                self.model,
                sample_shape,
                noise=x_noised,
                clip_denoised=self.args.clip_denoised,
                # denoised_fn=partial(denoised_fn_round, args, model_emb),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                top_p=self.args.top_p,
                clamp_step=self.args.clamp_step,
                clamp_first=True,
                mask=x_mask,
                x_start=x_start,
                gap=step_gap
            )
            sample = samples[-1]
            feat = sample.view(batch_size*seq_len, -1) 
            x_pred, _, _= self.model.answer_queries(feat, batch)

            met.digest(x_pred, batch.x_ans,
                    weight=batch.x_pred_weight if hasattr(batch, 'x_pred_weight') else None)

        time_end = time.time()
        record_time = time_end - time_start
        average_time = record_time / total

        logging.info('overall time: {:.5}, average time: {:.5}'.format(record_time, average_time))

        results['mrr'] = met.MRR()
        results['hit1'] = met.hits_at(1)
        results['hit3'] = met.hits_at(3)
        results['hit10'] = met.hits_at(10)
        logging.info("\n")
        logging.info('[{}]: MRR: {:.5}, hits@1: {:.5}, hits@3: {:.5}, hits@10: {:.5}'.format(
            self.args.split, results['mrr'], results['hit1'],results['hit3'],results['hit10']))
        current = datetime.now()
        logging.info("Current test time: {}".format(current))

    def optimize_fp16(self):
        if any(not torch.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logging.info("Found NaN, decreased lg_loss_scale to {}".format(self.lg_loss_scale))
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.optimizer.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def grad_clip(self):
        max_grad_norm=self.gradient_clipping
        if hasattr(self.optimizer, "clip_grad_norm"):
            self.optimizer.clip_grad_norm(max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm,
            )

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.grad_clip()
        self._log_grad_norm()
        self._anneal_lr()
        self.optimizer.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            if p.grad != None:
                sqsum += (p.grad ** 2).sum().item()

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        frac_done = (self.step) / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


    def save(self, epoch):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            logging.info("saving model {}...".format(epoch))
            if not rate:
                filename = "model{:03d}.pt".format(epoch)
            else:
                filename = "model_{:03d}.pt".format(epoch)

            print('writing to', bf.join(self.args.checkpoint_dir, filename))
            with bf.BlobFile(bf.join(self.args.checkpoint_dir, filename), "wb") as f: # DEBUG **
                torch.save(state_dict, f) # save locally
                # pass # save empty

        # save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params # DEBUG **
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params



