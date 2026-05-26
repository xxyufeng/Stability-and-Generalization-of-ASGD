import os
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if k in os.environ:
        del os.environ[k]

os.environ["RAY_raylet_ip_address"] = "127.0.0.1"
os.environ["RAY_node_ip_address"] = "127.0.0.1"

os.environ["RAY_raylet_start_wait_time_s"] = "120" 
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, Subset
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt 
import time
# from tqdm import tqdm
import pandas as pd
import copy
import shutil

from dataset_and_model import load_data, load_model

############################
# Utils & Logger 
############################

#distribute data to workers, making sure all changed samples go to one worker
def distribute_data(n_workers, dataset, heter=True, seed=None):
    if seed is not None:
        np.random.seed(42+seed)
    print("----Distributing data by labels----")
    
    # Extract labels
    labels = []
    # optimize this by directly accessing dataset.targets or dataset.labels if available,
    # otherwise we have to iterate (slow).
    # Assuming MNIST/CIFAR standard structure where dataset has targets/labels or [1] is label.
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'): # Cifar10 sometimes
        labels = np.array(dataset.labels)
    else:
        # Fallback to iteration
        loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        all_labels = []
        for _, y in loader:
            all_labels.extend(y.numpy())
        labels = np.array(all_labels)

    unique_labels = np.unique(labels)
    indices_by_label = {lbl: np.where(labels == lbl)[0] for lbl in unique_labels}
    
    distributed = {}
    
    worker_indices = {w: [] for w in range(n_workers)}
    
    sorted_labels = sorted(unique_labels)
    n_classes = len(sorted_labels)

    # Improved distribution: Allows n_workers > n_classes (e.g. RCV1 binary classification)
    # Each worker is assigned to exactly one label class, but one label can be split across multiple workers.
    for w in range(n_workers):
        # 1. Determine which label this worker is responsible for
        label_idx = w % n_classes
        lbl = sorted_labels[label_idx]
        
        # 2. Find all indices for this label
        all_indices = indices_by_label[lbl]
        
        # 3. Determine which slice of this label's data this worker gets
        # Identify all workers sharing this label
        sharers = [i for i in range(n_workers) if i % n_classes == label_idx]
        num_sharers = len(sharers)
        rank_in_group = sharers.index(w)
        
        # Split indices equally among sharers
        chunk_size = len(all_indices) // num_sharers
        start = rank_in_group * chunk_size
        # The last sharer gets the remainder to ensure no data loss
        end = start + chunk_size if rank_in_group < num_sharers - 1 else len(all_indices)
        
        worker_indices[w].extend(all_indices[start:end])
        
    for w in range(n_workers):
        np.random.shuffle(worker_indices[w])
        # Only model 0 exists significantly now, but keeping dictionary structure for compatibility
        # We only have model 0 (main model).
        distributed[w] = worker_indices[w]

    # Return standard format. 
    # The caller expects distributed[key][worker_id], but here we only have one 'model' concept basically.
    
    # Reformat to match expect structure: distributed[model_idx][worker_id]
    distributed_formatted = {0: distributed}
    
    return distributed_formatted, None 


class Logger:
    def __init__(self, log_dir, log_filename):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, log_filename)

    def update(self, iteration, delay, NMT, t_loss, t_acc, te_loss, te_acc, gap, delay_accum):
        with open(self.file_path, 'a') as f:
            f.write(f"Iter: {iteration}, Delay: {delay}, NMT: {NMT}, "
                    f"TrLoss: {t_loss:.4f}, TrAcc: {t_acc:.4f}, "
                    f"TeLoss: {te_loss:.4f}, TeAcc: {te_acc:.4f}, "
                    f"Gap: {gap:.4f}, DelayAccum: {delay_accum:.4f}",
                    )
            f.write("\n")

class DataRecorder:
    def __init__(self, rec_dir, filename):
        self.rec_dir = rec_dir
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir, exist_ok=True)
        self.data = []
        self.save_path = os.path.join(rec_dir, filename)

    def update(self, rec_dict):
        self.data.append(rec_dict)

    def save(self):
        if not self.data: return
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path, index=False)
        print(f"Data saved to {self.save_path}")

class ModelCheckpoint:
    def __init__(self, checkpoint_dir, context_str):
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.context = context_str

    def save(self, model, iteration):
        path = os.path.join(self.checkpoint_dir, f"{self.context}_iter{iteration}.pt")
        torch.save(model.state_dict(), path)

class IndexSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        np.random.shuffle(self.indices)
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

############################
# Ray Actors
############################

@ray.remote
class RayServer:
    def __init__(self, args, distributed_info, log_paths):
        # init params
        self.args = args
        self.distributed_datasets = distributed_info
        
        self.iteration = 1
        self.start_time = time.time()
        
        # Logging
        self.logger = Logger(log_paths['log_dir'], log_paths['log_name'])
        self.recorder = DataRecorder(log_paths['rec_dir'], log_paths['csv_name'])
        self.checkpoint = ModelCheckpoint(log_paths['ckpt_dir'], log_paths['context'])
        
        # Models
        self.device = 'cpu' # Server parameter integration usually on CPU to save GPU for workers, or use GPU if available
        if args.device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
            
        self.num_models = 1 # Only one model for random SGD (no stability pairs)
        self.models = {i: load_model(args.model, args.loss, q=args.q).to(self.device) for i in range(self.num_models)}
        self.optimizers = {i: optim.SGD(self.models[i].parameters(), lr=args.lr) for i in range(self.num_models)}
        
        # Datasets for evaluation
        self.train_dataset = load_data(args.dataset, args.dataset_path, 'train')
        self.test_dataset = load_data(args.dataset, args.dataset_path, 'test')
        
        # Create Train Subset (All used samples)
        used_indices = []
        for wid in range(args.num_workers):
            # Key 0 is the main dataset
            used_indices.extend(self.distributed_datasets[0][wid])
        used_indices = list(dict.fromkeys(used_indices))
        self.train_sub_dataset = Subset(self.train_dataset, used_indices)
        
        # Metrics tracking
        self.worker_contributions = {i: 0 for i in range(args.num_workers)}
        self.delay_accum = 0

        self.history = {"iterations":[]}
        
        # Buffer for Waiting ASGD
        self.grad_buffer = {i: [] for i in range(self.num_models)}
        self.worker_buffer = []  # Stores worker IDs that have submitted gradients in this batch
        self.delay_buffer = []
        
        # For Random/Shuffle SGD
        self.worker_handles = []
        self.shuffle_order = []
        self.shuffle_idx = 0

    def set_workers(self, worker_handles):
        """Set worker handles for Random SGD communication"""
        self.worker_handles = worker_handles
        if self.args.ASGD_type in ['shuffle', 'shuffle_waiting']:
            self.shuffle_order = list(np.random.permutation(len(self.worker_handles)))
            self.shuffle_idx = 0


    def get_weights(self):
        """Worker pull weights from Server"""
        # Transfer to CPU to reduce serialization overhead and GPU memory usage
        return {k: v.cpu().state_dict() for k, v in self.models.items()}, self.iteration

    def get_current_iter(self):
        return self.iteration

    def push_weights(self, num_pushes=1):
        """
        For Random/Shuffle SGD: Select workers and push current weights to them.
        This triggers the workers to perform a computation step.
        """
        if not self.worker_handles:
            return
        
        # Get current weights
        weights_dict, iteration = self.get_weights()
        
        for _ in range(num_pushes):
            if self.args.ASGD_type in ['shuffle', 'shuffle_waiting']:
                if self.shuffle_idx >= len(self.shuffle_order):
                    self.shuffle_order = list(np.random.permutation(len(self.worker_handles)))
                    self.shuffle_idx = 0
                worker_idx = self.shuffle_order[self.shuffle_idx]
                self.shuffle_idx += 1
            else:
                # Randomly select a worker
                worker_idx = np.random.randint(len(self.worker_handles))

            target_worker = self.worker_handles[worker_idx]
            # Command the worker to update and compute
            target_worker.accept_weights_and_compute.remote(weights_dict, iteration)

    def trigger_all_workers(self):
        """
        For Random SGD: Trigger all workers to start computing.
        """
        if not self.worker_handles:
            return

        weights_dict, iteration = self.get_weights()
        
        for target_worker in self.worker_handles:
             target_worker.accept_weights_and_compute.remote(weights_dict, iteration)

    def apply_gradients(self, worker_id, pull_iter, grads_dict, loss_mt):
        """Worker submit gradients, Server update models"""
        
        # 1. Delay Calculation & Buffer storing
        delay = self.iteration - pull_iter
        self.worker_buffer.append(worker_id)
        self.delay_buffer.append(delay)
        
        for i in range(self.num_models):
            if i in grads_dict:
                self.grad_buffer[i].append(grads_dict[i])
                
        # Determine batch wait size (b)
        if self.args.ASGD_type in ['waiting', 'random_waiting', 'shuffle_waiting']:
            b = getattr(self.args, 'wait_b', 1) 
        else:
            b = 1
        
        # If we reached the target b, we proceed to update
        if len(self.worker_buffer) >= b:
            avg_delay = sum(self.delay_buffer) / b
            
            # Record contributions and delay sum
            for wid in self.worker_buffer:
                self.worker_contributions[wid] += 1
            self.delay_accum += sum(self.delay_buffer)
            
            # 2. Update Models
            scale_factor = 1.0
            if self.args.lr_type == 'adaptive':
                # Adaptive learning rate based on delay: step_size * (1/(delay + 1))
                scale_factor = (1/self.args.lr) / (10*avg_delay+1/self.args.lr)

            for i in range(self.num_models):
                if not self.grad_buffer[i]: continue
                
                self.optimizers[i].zero_grad()
                
                # Average the buffered gradients
                avg_grads = []
                for param_idx in range(len(self.grad_buffer[i][0])):
                    param_grads = [b_grads[param_idx] for b_grads in self.grad_buffer[i] if b_grads[param_idx] is not None]
                    if param_grads:
                        # Stack to average
                        avg_grads.append(torch.stack(param_grads).mean(dim=0))
                    else:
                        avg_grads.append(None)
                
                # Apply averaged gradients
                for param, grad in zip(self.models[i].parameters(), avg_grads):
                    if grad is not None:
                        # Apply scaling to the gradient (equivalent to scaling LR)
                        param.grad = grad.to(self.device) * scale_factor
                self.optimizers[i].step()

            # 3. Evaluation Check
            current_iter = self.iteration
            
            check_points = [100, 200, 300, 400, 500, 600, 700, 800, 900]
            if current_iter % self.args.eval_interval == 0 or current_iter == 1 or current_iter in check_points:
                self._evaluate_routine(current_iter, avg_delay)
                
            self.iteration += 1
            
            finished = self.iteration > self.args.iterations
            
            # Follow-up actions based on mode
            if not finished:
                if self.args.ASGD_type in ['random', 'shuffle']:
                    self.push_weights(num_pushes=1) 
                elif self.args.ASGD_type in ['random_waiting', 'shuffle_waiting']:
                    self.push_weights(num_pushes=b)
                elif self.args.ASGD_type in ['pure', 'waiting']:
                    # Triggers exactly the b workers that just participated
                    weights_dict, new_iter = self.get_weights()
                    for wid in self.worker_buffer:
                        target_worker = self.worker_handles[wid]
                        target_worker.accept_weights_and_compute.remote(weights_dict, new_iter)
            
            # Clear buffers for the next wait batch
            self.grad_buffer = {i: [] for i in range(self.num_models)}
            self.worker_buffer = []  
            self.delay_buffer = []
            
        else:
            finished = self.iteration > self.args.iterations
            
        return finished

    def _evaluate_routine(self, current_iter, delay):
        # Calculate Loss/Acc
        tr_loss, tr_acc = self._run_inference(self.models[0], self.train_sub_dataset)
        te_loss, te_acc = self._run_inference(self.models[0], self.test_dataset)

        # Bounds
        NMT = max(self.worker_contributions.values())
        delay_accum = self.delay_accum

        gen_gap = te_loss - tr_loss

        print(f"Iter:{current_iter}| Delay:{delay}| NMT:{NMT}| Train Loss:{tr_loss:.6f}| Train Acc:{tr_acc:.6f}| Test Loss:{te_loss:.6f}| Test Acc:{te_acc:.6f}| Gap:{gen_gap:.6f}| DelayAccum:{delay_accum:.4f}")

        # Logging
        self.logger.update(current_iter, delay, NMT, tr_loss, tr_acc, te_loss, te_acc, gen_gap, delay_accum)
        
        rec_data = {
            'iteration': current_iter, 'delay': delay, 'NMT': NMT, 'delay_accum': delay_accum,
            'train_loss': tr_loss, 'train_acc': tr_acc,
            'test_loss': te_loss, 'test_acc': te_acc,
            'generalization_gap': gen_gap
        }
        self.recorder.update(rec_data)

        if current_iter >= self.args.iterations:
            self.recorder.save()
            self.checkpoint.save(self.models[0], current_iter)

    def _run_inference(self, model, dataset):
        model.eval()
        # Create temp loader
        loader = DataLoader(dataset, batch_size=2000, shuffle=False) # Large batch for faster inference
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, loss = model(x, y)
                total_loss += loss.item() * x.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        model.train()
        return total_loss / total, correct / total



@ray.remote
class RayWorker:
    def __init__(self, worker_id, args, indices_dict, server):
        self.worker_id = worker_id
        self.args = args
        self.server = server
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        
        # Load local dataset copies
        self.full_train = load_data(args.dataset, args.dataset_path, 'train')
        
        self.num_models = 1 # Only one model for random SGD
        self.models = {i: load_model(args.model, args.loss, q=args.q).to(self.device) for i in range(self.num_models)}
        
        self.indices_dict = {k: list(v) for k, v in indices_dict.items()}
        self.dataset_len = len(self.indices_dict[0])
        
        self.perm_indices = np.arange(self.dataset_len)
        self.curr_pos = 0
        
        # Buffer for received weights (Random SGD) - Use a queue to prevent overwriting
        # Although Ray actors are single-threaded by default, explicit queueing 
        # ensures logic correctness if concurrency changes or for mental model.
        self.model_queue = []
            
        # seed
        torch.manual_seed(args.seed_base + worker_id)
        np.random.seed(args.seed_base + worker_id)
        
        np.random.shuffle(self.perm_indices)

    def _get_next_batch_indices(self):
        """Get next batch indices (Position Indices)"""
        if self.curr_pos >= self.dataset_len:
            np.random.shuffle(self.perm_indices)
            self.curr_pos = 0
            
        end_pos = min(self.curr_pos + self.args.batch_size, self.dataset_len)
        batch_positions = self.perm_indices[self.curr_pos : end_pos]
        self.curr_pos = end_pos
        return batch_positions

    def _fetch_data(self, model_idx, batch_positions):
        """Get real data for a specific model based on unified position indices"""
        # 1. Map to the real dataset indices for the specific model
        real_indices = [self.indices_dict[model_idx][p] for p in batch_positions]
        
        # 2. Extract data from Dataset and manually collate
        # (Replacing the functionality of DataLoader)
        batch_x = []
        batch_y = []
        for idx in real_indices:
            x, y = self.full_train[idx]
            batch_x.append(x)
            batch_y.append(y)
            
        if len(batch_x) > 0:
            inputs = torch.stack(batch_x)
            labels = torch.tensor(batch_y)
        else:
            inputs = torch.empty(0)
            labels = torch.empty(0)
            
        return inputs, labels

    def _compute_and_report(self, pull_iter):
        """Core computation logic: calculate gradients and reporting to server"""
        # --- 2. Speed Control (Simulate Delay) ---
        b = getattr(self.args, 'wait_b', 1) 
            
        if self.worker_id >= b:
            if self.args.slow_delay > 0:
                time.sleep(self.args.slow_delay)
        
        # --- 3. Compute Gradients ---
        grads_dict = {}
        loss_model0 = 0.0
        
        batch_positions = self._get_next_batch_indices()
        
        for i in range(self.num_models):
            self.models[i].train()
            self.models[i].zero_grad()
            
            inputs, labels = self._fetch_data(i, batch_positions)
            
            if inputs.size(0) == 0: continue

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, loss = self.models[i](inputs, labels)
            loss.backward()
            
            # Collect Gradients
            grads = [p.grad.detach().cpu() if p.grad is not None else None for p in self.models[i].parameters()]
            grads_dict[i] = grads
            
            if i == 0:
                loss_model0 = loss.item()

        # --- 4. Push Gradients ---
        # Note: In Random Mode, apply_gradients might trigger the next random worker
        finished = ray.get(self.server.apply_gradients.remote(self.worker_id, pull_iter, grads_dict, loss_model0))
        return finished

    def accept_weights_and_compute(self, weights_dict, iteration):
        """
        For Random SGD: Receive weights from server, update local buffer, 
        and perform computation.
        Uses a queue to ensure no weights are overwritten if multiple requests arrive.
        """
        # Enqueue the received task
        self.model_queue.append((weights_dict, iteration))
        
        # Process the queue
        # Note: In standard Ray actors, this method blocks, so the queue is processed synchronously.
        # This prevents the actor from accepting new 'accept_weights_and_compute' calls until this loop finishes,
        # ensuring sequential processing without overwrites.
        while self.model_queue:
            # Dequeue (FIFO)
            current_weights, current_iter = self.model_queue.pop(0)
            
            # Load weights into local models
            for i in range(self.num_models):
                self.models[i].load_state_dict(current_weights[i])
                
            # Compute and Report
            self._compute_and_report(current_iter)

############################
# Main Driver
############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Original Args
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "rcv1"])
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--model", type=str, default="fcnet_mnist")
    parser.add_argument("--ASGD-type", type=str, default="pure", choices=["pure", "random", "waiting", "random_waiting", "shuffle", "shuffle_waiting"])
    parser.add_argument("--wait-b", type=int, default=1, help="Number of gradients to wait for before updating (Algorithm 3)")
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-type", type=str, default="constant", choices=["constant", "adaptive"])
    parser.add_argument("--q", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers-list", type=str, default="1",
                        help="comma-separated list of worker counts, e.g. 1,11,21")
    parser.add_argument("--num-samples", type=int, default=1000)
    # parser.add_argument("--n-pairs", type=int, default=1) # Deprecated
    parser.add_argument("--device", type=str, default="cpu") 
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--log-root", type=str, default="Exp_Ray_ASGD")
    parser.add_argument("--save-prefix", type=str, default="")
    
    # New Args for Speed Control
    parser.add_argument("--fast-worker-id", type=int, default=0, help="ID of the worker that runs fastest")
    parser.add_argument("--slow-delay", type=float, default=0, help="Seconds to sleep for slow workers (simulating delay)")
    parser.add_argument("--slow-delay-list", type=str, default="0",
                        help="comma-separated list of slow-delay, e.g. 1,11,21")
    
    args = parser.parse_args()

    num_workers_list = [int(x) for x in args.num_workers_list.split(",") if x.strip()]
    slow_delay_list = [float(x) for x in args.slow_delay_list.split(",") if x.strip()]
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    if args.device == 'cuda':
        ray.init(num_gpus=1, ignore_reinit_error=True, include_dashboard=False)
    else:
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    print(f"Ray initialized. Resources: {ray.cluster_resources()}")

    for r in range(1, args.repeats + 1):
        print(f"\n=== Repeat {r} ===")
        current_seed = args.seed_base + (r - 1) * 10
        args.seed_base = current_seed # Update for this run

        # Set paths
        sub = args.save_prefix
        base_dir = os.path.join(args.log_root, f"{sub}{args.dataset}", f"r{r}")
        
        for nworkers in num_workers_list:
            for slow_delay in slow_delay_list:
                args.slow_delay = slow_delay
                args.num_workers = nworkers
                log_paths = {
                'log_dir': os.path.join(base_dir, "logs"),
                'rec_dir': os.path.join(base_dir, "records"),
                'ckpt_dir': os.path.join(base_dir, "checkpoints"),
                'log_name': f"log_ray_{args.ASGD_type}.txt",
                'csv_name': f"record_ray_{args.ASGD_type}_{args.num_workers}workers_{args.slow_delay}delay_{args.lr}{args.lr_type}lr.csv",
                'context': f"{args.ASGD_type}_m{args.num_workers}"
                }
            # for lr in lr_list:
                print(f"[Repeat {r}/{args.repeats}] Seed={current_seed}")
                # 1. Prepare Data Indices (Run on Driver)
                # Load dataset
                temp_data = load_data(args.dataset, args.dataset_path, 'train')
                
                # Distribute data by label (no stability pairs)
                distributed_datasets, _ = distribute_data(
                    nworkers, temp_data, heter=True, seed=current_seed
                )
                
                del temp_data # free memory on driver

                # 2. Start Server Actor
                # num_cpus=1 Ensures Server occupies one logical thread
                server_actor = RayServer.options(num_cpus=1).remote(
                    args, distributed_datasets, log_paths
                )

                # 3. Start Worker Actors
                worker_actors = []
                for i in range(args.num_workers):
                    # Only one "pair" (index 0) exists now
                    w_indices = {
                        0: distributed_datasets[0][i] 
                    }
                
                    res_opts = {"num_cpus": 0.2} 
                    
                    if args.device == 'cuda':
                        res_opts["num_gpus"] = 1.0 / (args.num_workers + 1)
                    elif args.device == 'cpu':
                        res_opts["num_cpus"] = max(0.2, 1.0 * ray.cluster_resources().get("CPU", 1) / (args.num_workers + 1))
                    
                    w = RayWorker.options(**res_opts).remote(i, args, w_indices, server_actor)
                    worker_actors.append(w)
                    
                print("Actors started. Training begins...")
                
                # 4. Trigger Training
                # In all modes (pure, random, waiting), we pass worker handles to server first
                ray.get(server_actor.set_workers.remote(worker_actors))
                
                # Trigger the process by triggering ALL workers to start
                print(f"Mode: {args.ASGD_type.capitalize()} SGD. Triggering all workers...")
                server_actor.trigger_all_workers.remote()
                
                futures = [] # No futures from train_loop because they return immediately
                
                # Wait until server reaches iterations, then force kill
                while True:
                    # Check server status
                    current_iter = ray.get(server_actor.get_current_iter.remote())
                    if current_iter > args.iterations:
                        print("Max iterations reached. Force stopping workers...")
                        break
                    
                    # Check if all workers finished normally (optional, but good practice)
                    if futures:
                        done, not_done = ray.wait(futures, num_returns=len(futures), timeout=1.0)
                        if len(not_done) == 0:
                            print("All workers finished normally.")
                            break
                        
                    time.sleep(1) 
                
                print(f"Repeat {r} finished.")
                
                # Kill actors to free resources for next repeat
                for w in worker_actors:
                    ray.kill(w)
                ray.kill(server_actor)
        
    ray.shutdown()
           
        