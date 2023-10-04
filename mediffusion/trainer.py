import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import bkh_pytorch_utils as bpu

class Trainer:
    def __init__(
            self, 
            max_steps,
            val_check_interval=5000,
            root_directory="./outputs", 
            precision="16-mixed",
            devices=-1,
            nodes=1,
            wandb_project=None,
            logger_instance='mediffusion_experimental_run',
        ):
        
        # Check if Wandb API key is present
        if 'WANDB_API_KEY' not in os.environ:
            raise EnvironmentError("Wandb API key not found. Please set the WANDB_API_KEY environment variable.")

        # Check if project name or instance is None
        if logger_instance is None or wandb_project is None:
            raise ValueError("Project name or logger instance cannot be None.")

        self.max_steps = max_steps
        self.val_check_interval = val_check_interval
        self.root_directory = root_directory
        self.precision = precision
        self.devices = devices
        self.nodes = nodes
        self.logger_instance = logger_instance
        self.wandb_project = wandb_project

        os.makedirs(f"{self.root_directory}/pl", exist_ok=True)
        os.makedirs(f"{self.root_directory}/wandb", exist_ok=True)

        self.trainer = self.setup_trainer()

    def setup_trainer(self):
        wandb_logger = WandbLogger(
            save_dir=self.root_directory + "/wandb",
            name=self.logger_instance,
            project=self.wandb_project,
            offline=False,
            log_model=False,
        )
        
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callback1= ModelCheckpoint(
            dirpath=self.root_directory + "/pl",
            filename=f'{{epoch}}-{{step}}-{{train_loss:0.4F}}',
            monitor="train_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        )

        ema = bpu.EMA(decay=0.9999, ema_interval_steps=1, ema_device="cpu", use_ema_for_validation=True)

        # Setup Trainer
        trainer = pl.Trainer(
            gradient_clip_val=1.0,
            deterministic=True,
            callbacks=[checkpoint_callback1, lr_monitor, ema],
            profiler='simple',
            logger=wandb_logger,
            precision=self.precision,
            accelerator="gpu",
            devices=1 if bpu.is_notebook_running() else self.devices,
            num_nodes=self.nodes,
            strategy="auto" if bpu.is_notebook_running() else DDPStrategy(find_unused_parameters=False),
            log_every_n_steps=10,
            default_root_dir=self.root_directory,
            num_sanity_val_steps=0,
            fast_dev_run=False,
            max_epochs=-1,
            max_steps=self.max_steps,
            use_distributed_sampler=bpu.is_notebook_running() is False and self.devices != 1,
            val_check_interval=self.val_check_interval,
        )
        return trainer

    def fit(self, model):
        self.trainer.fit(model)