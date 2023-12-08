import os
import time
import torch
import argparse
import logging
from easydict import EasyDict

from train import Trainer
from task.query2graph import Query2Graph
from task.triple2graph import Triple2Graph
from models.kgreasoning_model import KGReasoning
from models.gaussian_diffusion import SpacedDiffusion
from models.gaussian_diffusion import space_timesteps, get_named_beta_schedule
from utils.basic import create_named_schedule_sampler
from utils.basic import set_seed, set_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def put_default_config(args):
    def set_default(key, value):
        if key not in args:
            args[key] = value
    set_default("type", "pretrain")
    set_default("names", "fb15k237")
    set_default("root_dir", "DiffCLR")
    set_default("data_dir", f"{args['root_dir']}/data/FB15k-237-betae")
    set_default("output_dir", f"{args['root_dir']}/output")
    set_default("checkpoint_dir", f"{args['root_dir']}/checkpoints/")
    set_default("train_modes", ["1p", "2p", "3p", "2i", "3i"])
    set_default("test_modes", ["1p", "2p", "3p", "2i", "3i", "ip", "pi"])
    set_default("do_train", False)
    set_default("do_valid", False)
    set_default("do_test", False)
    set_default("load_up_model", False)
    set_default("load_down_model", False)
    set_default("up_model_path", None)
    set_default("down_model_path", None),
    set_default("save_interval", 4)
    set_default("test_interval", 2)
    set_default("seed", 100)
    set_default("seed2", 101)
    set_default("num_workers", 10)
    set_default("num_epoch", 400)
    set_default("batch_size", 128)
    set_default("lr", 1e-4)
    set_default('scheduler', 'exp')
    set_default('exponential_lr_rate', 0.997)
    set_default("schedule_sampler", "lossaware")
    set_default("diffusion_steps",2000)
    set_default("noise_schedule", "sqrt")
    set_default("num_heads", 8)
    set_default("num_layers", 8)
    set_default("dim_feedforward",2048)
    set_default("hidden_t_dim", 128)
    set_default("hidden_dim", 1024)
    set_default("dropout", 0.1)
    set_default("gradient_clipping", -1.0)
    set_default("weight_decay", 0.0)
    set_default("loss", "CE")
    set_default("smoothing", 0.1)
    set_default("predict_xstart",True)
    set_default("model_mean_type","xstart")
    set_default("rescale_timesteps",True)
    set_default("clamp_step", 0)
    set_default("clip_denoised",False)
    set_default("top_p", -1)
    set_default("mask_ratio", 0.8)
    set_default('p_mask_ratio', 1.0)
    set_default("pretrain_mask_ratio", [0.2, 0.4]) 
    set_default("pretrain_mask_type_ratio", [1, 0]) 
    set_default("pretrain_dataset_source", "relation") 
    set_default("edge_drop_out_rate", 0)
    set_default("sample_retries", 5)
    set_default("ladies_size", 8)
    set_default("pretrain_sampler_ratio",{
        '1p': 0, '2p': 0, '3p': 0, '2i': 0, '3i': 0, 'meta_tree': 5, 'ladies': 5,})
    set_default("induced_edge_prob", 0.8)
    set_default("max_position_embeddings", 40)
    set_default("alpha", 1),
    set_default("ema_rate", 0.9999),
    set_default("resume_checkpoint", None),
    set_default("use_fp16", False),
    set_default("fp16_scale_growth", 0.0001),
    set_default("split", "test"),
    set_default("hop", 5),
    set_default("best_model", True)
    return args


def dfs_parsing(args_list, parse_status, task):
    stat = parse_status.get(task)
    if stat == 'Done':
        return
    if stat == 'Parsing':
        assert False, f'Loop detected in config.'
    parse_status[task] = 'Parsing'
    if task not in args_list:
        assert False, f'Task {task} not found'
    args = args_list[task]
    if 'base' in args:
        dfs_parsing(args_list, parse_status, args['base'])
        args_base = args_list[args['base']]
        del args['base']
        for k in args_base:
            if k not in args:
                args[k] = args_base[k]
    put_default_config(args)
    parse_status[task] = 'Done'


def args_to_config(config):
    import json
    args_list = json.load(config.config)
    assert isinstance(args_list, dict), "Config should be an dict of tasks."
    parse_status = dict()
    for task in args_list:
        dfs_parsing(args_list, parse_status, task)
    return args_list


class Runner(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def load_data(self):
        if self.args.type == "pretrain":
            t2g = Triple2Graph(self.args)
            self.num_nodes = t2g.num_nodes
            self.relation_cnt = t2g.relation_cnt
            train_loader = t2g.dataloader_train()
            test_loader = t2g.dataloader_test()
        elif self.args.type == "reasoning":
            q2g = Query2Graph(self.args)
            self.num_nodes = q2g.num_nodes
            self.relation_cnt = q2g.relation_cnt
            train_loader = q2g.dataloader_train()
            # valid_loader = q2g.dataloader_valid()
            test_loader = q2g.dataloader_test()
        return train_loader, test_loader

    def add_model(self):
        model = KGReasoning(self.num_nodes, self.relation_cnt, self.args)
        
        betas = get_named_beta_schedule(self.args.noise_schedule, self.args.diffusion_steps)
        use_timesteps = space_timesteps(self.args.diffusion_steps, [self.args.diffusion_steps])
        diffusion = SpacedDiffusion(self.args, betas=betas, use_timesteps=use_timesteps)
        return model, diffusion
  
    def train(self):
        logging.info('load data!')
        train_loader, test_loader = self.load_data()

        logging.info('add model!')
        model, diffusion = self.add_model()
        model = model.to(self.device)
        schedule_sampler = create_named_schedule_sampler(self.args.schedule_sampler, diffusion)

        n_gpu = torch.cuda.device_count()
        if n_gpu>1:
            model = torch.nn.DataParallel(model)

        model_total_params = sum(p.numel() for p in model.parameters())
        logging.info('### The parameter count is {}'.format(model_total_params)) # 11111

        if self.args.load_up_model and self.args.best_model:
            best_result, best_model = 0, self.args.up_model_path
            for i in range(0,10000,100):
                model_name = f"model_{(i):03d}.pt"
                load_path = os.path.join(self.args.up_model_path, model_name)
                if not os.path.exists(load_path):
                    break

                logging.info('load model {}'.format(model_name))
                state = torch.load(load_path)
                model.load_state_dict(state)

                model.eval()
                cur_result = Trainer(
                    args=self.args,
                    device = self.device,
                    model=model,
                    diffusion=diffusion,
                    test_loader=test_loader,
                    schedule_sampler=schedule_sampler
                ).test()
                          
                if cur_result > best_result:
                    best_model = load_path
                    best_result = cur_result
            
            self.args.up_model_path = best_model
            torch.cuda.empty_cache()

        if self.args.load_up_model:
            logging.info('load best model {}'.format(self.args.up_model_path))
            state = torch.load(self.args.up_model_path)
            model.load_state_dict(state)

        model.train() 
        Trainer(
            args=self.args,
            device = self.device,
            model=model,
            diffusion=diffusion,
            train_loader=train_loader,
            test_loader=test_loader,
            schedule_sampler=schedule_sampler
        ).train()

    @torch.no_grad() 
    def test(self):
        logging.info('load data!')
        train_loader, test_loader = self.load_data()

        logging.info('add model!')
        model, diffusion = self.add_model() 
        model = model.to(self.device)
        schedule_sampler = create_named_schedule_sampler(self.args.schedule_sampler, diffusion) 

        n_gpu = torch.cuda.device_count()
        if n_gpu>1:
            model = torch.nn.DataParallel(model)

        if self.args.load_down_model and self.args.best_model:
            best_result, best_model = 0, self.args.load_down_model
            for i in range(0,10000):
                model_name = f"model_{(i):03d}.pt"
                load_path = os.path.join(self.args.down_model_path, model_name)
                if not os.path.exists(load_path):
                    break

                logging.info('load model {}'.format(model_name))
                state = torch.load(load_path)
                model.load_state_dict(state)

                model.eval()
                cur_result = Trainer(
                    args=self.args,
                    device = self.device,
                    model=model,
                    diffusion=diffusion,
                    test_loader=test_loader,
                    schedule_sampler=schedule_sampler
                ).test()
                          
                if cur_result > best_result:
                    best_model = load_path
                    best_result = cur_result
            
            self.args.down_model_path = best_model
            torch.cuda.empty_cache()

        if self.args.load_down_model:
            logging.info('load best model {}'.format(self.args.down_model_path))
            state = torch.load(self.args.down_model_path)
            model.load_state_dict(state)

        model.eval().requires_grad_(False)
        Trainer(
            args=self.args,
            device = self.device,
            model=model,
            diffusion=diffusion,
            test_loader=test_loader,
            schedule_sampler=schedule_sampler
        ).test()

    @torch.no_grad() 
    def test_item(self):
        logging.info('load data!')
        train_loader, test_loader = self.load_data()

        logging.info('add model!')
        model, diffusion = self.add_model() 
        model = model.to(self.device)
        schedule_sampler = create_named_schedule_sampler(self.args.schedule_sampler, diffusion)
    
        n_gpu = torch.cuda.device_count()
        if n_gpu>1:
            model = torch.nn.DataParallel(model)

        if self.args.load_down_model and self.args.best_model:
            best_result, best_model = 0, self.args.load_down_model
            for i in range(0,10000,1):
                model_name = f"model_{(i):03d}.pt"
                load_path = os.path.join(self.args.down_model_path, model_name)
                if not os.path.exists(load_path):
                    break

                logging.info('load model {}'.format(model_name))
                state = torch.load(load_path)
                model.load_state_dict(state)

                model.eval()
                cur_result = Trainer(
                    args=self.args,
                    device = self.device,
                    model=model,
                    diffusion=diffusion,
                    test_loader=test_loader,
                    schedule_sampler=schedule_sampler
                ).test()
                          
                if cur_result > best_result:
                    best_model = load_path
                    best_result = cur_result
            
            self.args.down_model_path = best_model
            torch.cuda.empty_cache()

        if self.args.load_down_model:
            logging.info('load best model {}'.format(self.args.down_model_path))
            state = torch.load(self.args.down_model_path)
            model.load_state_dict(state)

        model.eval().requires_grad_(False)

        # test model by a single mode
        test_mode_list = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]
        for test_mode in test_mode_list:
            logging.info('load data! {}'.format(test_mode))
            self.args.test_modes = [test_mode]
            q2g = Query2Graph(self.args)
            test_loader = q2g.dataloader_test()
            Trainer(
                args=self.args,
                device = self.device,
                model=model,
                diffusion=diffusion,
                test_loader=test_loader,
                schedule_sampler=schedule_sampler
            ).test()


def run_pretrain(args, device):
    runner = Runner(args, device)
    if args.do_train:
        logging.info('pre training time!')
        runner.train()
    if args.do_test:
        logging.info('pre testing time!')
        runner.test()


def run_reasoning(args, device):
    runner = Runner(args, device)
    if args.do_train:
        logging.info('finetune training time!')
        runner.train()
    if args.do_test:
        logging.info('finetune testing time!')
        # runner.test()
        runner.test_item()


def get_argparser():
    parser = argparse.ArgumentParser(prog='python main.py', description='DiffCLR')
    # pretrain 
    parser.add_argument('--config', default="configs/nell995.json", type=argparse.FileType('r'), help='config file')
    parser.add_argument('--task', default="pretrain1", type=str, help='task to run')
    
    # fineturn
    # parser.add_argument('--config', default="configs/fb15k237.json", type=argparse.FileType('r'), help='config file')
    # parser.add_argument('--tasks', default=["reasoning_multi"], type=list, help='task to run')

    return parser


def main():
    config = get_argparser().parse_args()
    args_list = args_to_config(config)

    if config.task not in args_list:
        assert False, f'Task {config.task} not found in config' 
 

    args = args_list[config.task]
    args = EasyDict(args)

    checkpoint_dir = args.checkpoint_dir
    if args.do_train:
        args.up_model_path = os.path.join(checkpoint_dir, args.up_model_path)
        model_dir = f"{args.type}/{args.names}_{config.task}_" + time.strftime("%Y%m%d-%H:%M:%S")
        model_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        args.checkpoint_dir = model_dir
        logger_file = os.path.join(model_dir,'log.txt')
        set_logger(logger_file)

    if args.do_test:
        args.down_model_path = os.path.join(checkpoint_dir, args.down_model_path)
        model_dir = args.down_model_path.split('/')[-2:]
        model_dir = "_".join(model_dir)
        model_dir = os.path.join(args.output_dir, model_dir)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        args.output_dir = model_dir
        logger_file = os.path.join(model_dir,'log.txt')
        set_logger(logger_file)
    
    logging.info('Running dataset: {}'.format(args.names))
    logging.info('Running task: {}'.format(config.task))
    logging.info('Definitive args:')
    logging.info(args)

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.empty_cache()

    if args.type == 'pretrain':
        run_pretrain(args, device)
    elif args.type == 'reasoning':
        run_reasoning(args, device)
    else:
        assert False, f'This is not runnable.'


if __name__ == "__main__":
    
    main()
