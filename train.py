import numpy as np
from pathlib import Path 
from typing import Callable

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, ProgressBarCallback

from utils.CustomGym import SqlCurriculumGym

from utils.callbacks import AsymptoticConvergence, verbose_evaluate

import torch


"""Stage
1: The Breaker - Escape and Bypass
2: The Architect - Column alignment
3: The Aligner - Null alignment and verification
4: The Mapper - Table Discovery
5: The Extractor - Targeted dump (per table)
"""

#MOVE TO EVIRONS:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


STAGE_CONFIG = {
    1: {"n_steps":512,
                "batch_size":256,
                "learning_rate":2.0e-4,
                "gamma":0.93,
                "gae_lambda":0.92,
                "ent_coef":0.05,
                "max_grad_norm":0.5,
                "clip_range":0.15,
                "verbose":0,
                "device":DEVICE,
                "tensorboard_log":"./tb_logs/"},
    2: {"n_steps":512,
                "batch_size":256,
                "learning_rate":2.5e-4,
                "gamma":0.94,
                "gae_lambda":0.93,
                "ent_coef":0.04,
                "max_grad_norm":0.5,
                "clip_range":0.15,
                "verbose":0,
                "device":DEVICE,
                "tensorboard_log":"./tb_logs/",
                "n_epochs":10},
    3: {"n_steps":512,
                "batch_size":256,
                "learning_rate":2.5e-4,
                "gamma":0.96,
                "gae_lambda":0.94,
                "ent_coef":0.075,
                "max_grad_norm":0.5,
                "clip_range":0.2,
                "verbose":0,
                "device":DEVICE,
                "tensorboard_log":"./tb_logs/",
                "n_epochs":15},
    4: {"n_steps":1024,
                "batch_size":512,
                "learning_rate":1.7e-4,
                "gamma":0.99,
                "gae_lambda":0.94,
                "ent_coef":0.025,
                "max_grad_norm":0.32,
                "clip_range":0.15,
                "verbose":0,
                "device":DEVICE,
                "tensorboard_log":"./tb_logs/",
                "n_epochs":15},
    5: {"n_steps":2048,
                "batch_size":2048,
                "learning_rate":5.0e-5,
                "gamma":0.995,
                "gae_lambda":0.95,
                "ent_coef":0.05,
                "max_grad_norm":0.5,
                "clip_range":0.2,
                "verbose":0,
                "device":DEVICE,
                "tensorboard_log":"./tb_logs/",
                "n_epochs":20}
                }

def linear_schedule(initial_value: float, floor_scale: float = 0.5) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    min_lr = initial_value * floor_scale
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return min_lr + (initial_value - min_lr) * progress_remaining

    return func


def make_env(stage, episode_length, base_seed):
    def _init():
        env = SqlCurriculumGym(
            stage=stage,
            database_path="database/backend.db",
            episode_length=episode_length
        )
        #env.reset(seed=base_seed)
        return Monitor(env)
    return _init


def load_model(env, stage:int, stage_params:dict, floor_scale:float = 0.5) -> RecurrentPPO:
    print(f"[*] Loading model...")
    current_model_path = Path(f"model/ppo_sql_{version_number}_stage{stage}_parallel.zip")
    prev_model_path = Path(f"model/ppo_sql_{version_number}_stage{stage-1}_parallel.zip")

    custom_objects = {
        "n_steps": stage_params["n_steps"],
        "batch_size": stage_params["batch_size"],
        "gamma": stage_params["gamma"],
        "gae_lambda": stage_params["gae_lambda"],
        "ent_coef": stage_params["ent_coef"],
        "clip_range": stage_params['clip_range'],
        "verbose": stage_params["verbose"]
    }

    try:
        if current_model_path.exists():
            print(f"[+] Found existing model for current Stage... Loading")
            model = RecurrentPPO.load(current_model_path, env, custom_objects=custom_objects)
            model.lr_schedule = linear_schedule(stage_params['learning_rate'], floor_scale=LR_FLOOR_RATE_EP[stage])
            return model

        elif prev_model_path.exists():
            print(f"[+] Found existing model for previous Stage... Loading")
            model = RecurrentPPO.load(prev_model_path, env, custom_objects=custom_objects)
            model.lr_schedule = linear_schedule(stage_params['learning_rate'], floor_scale=LR_FLOOR_RATE_EP[stage])
            print(f"[+] Loaded existing stage model: \n{prev_model_path!r}")
            return model

        else:
            print(f"[!] No existing model found for any stage. Creating a new one..")
            model = RecurrentPPO(
                "MultiInputLstmPolicy",
                env,
                **stage_params,)
            model.lr_schedule = linear_schedule(stage_params['learning_rate'], floor_scale=LR_FLOOR_RATE_EP[stage])
            return model
    except Exception as exc:
        print(f"[!] Exception occured while loading past model: \n{exc!r}")
        

if __name__ == "__main__":
    UPDATES_TARGET = 250
    EP_LENGTH = [16, 64, 128, 128, 512]
    LR_FLOOR_RATE_EP =[0, 0.01, 0.2, 0.09, 0.09, 0.15]
    NUM_ENVS = 8
    parallellism = "single"  #single or multiple
    version_number = "v0.9"

    for s in range(3,6):
        stage = s

        stage_params = STAGE_CONFIG[stage]
        
        updates_target = UPDATES_TARGET
        steps_per_update = stage_params['n_steps'] * NUM_ENVS 
        total_steps = updates_target * steps_per_update
  
        asymp_callback = AsymptoticConvergence(window_size=1200, epsilon=0.001, min_episodes=60, verbose=1)
        
        
        if parallellism == "single":
            env = DummyVecEnv([
                make_env(stage=stage, episode_length=EP_LENGTH[stage-1], base_seed=42*(1000*i))
                for i in range(NUM_ENVS)
            ])
        elif parallellism == "multiple":
            env = SubprocVecEnv([
                make_env(stage=stage, episode_length=EP_LENGTH[stage-1], base_seed=42*(1000*i))
                for i in range(NUM_ENVS)
            ])
        else:
            raise ValueError("Only single or multiple is accepted values for parallellism!")

        eval_env = Monitor(SqlCurriculumGym(
                            stage=stage, 
                            database_path="database/backend.db",                 
                            episode_length=EP_LENGTH[stage-1]))

        eval_callback = EvalCallback(eval_env, 
                                    best_model_save_path="model/best_models/",
                                    log_path="model/logs/",
                                    eval_freq = steps_per_update // 2, 
                                    n_eval_episodes=25,
                                    deterministic=True,
                                    render=False,
                                    warn=True,
                                    verbose=1)

        model = load_model(env, stage=stage, stage_params=stage_params, floor_scale=0.3)
        
        callback_list = CallbackList([
            ProgressBarCallback(),
            eval_callback ])

        #Name of the log_file
        run_name = f"Stage_{stage}_env_{NUM_ENVS}_{version_number}"


        model.learn(total_timesteps=total_steps, callback=callback_list, tb_log_name=run_name)
        
        f_name = f"model/ppo_sql_{version_number}_stage{stage}_parallel.zip" 
    

        model.save(f_name)
        previous_model = f_name

        print("\n[+] Starting Final Multi-Seed Evaluation...")
        final_eval_env = Monitor(SqlCurriculumGym(stage=stage, episode_length=EP_LENGTH[stage-1]))
        
        # verbose_evaluate should internally loop through different seeds
        mean_reward, std_reward = verbose_evaluate(model, final_eval_env, version_number, episodes=25, stage=stage)