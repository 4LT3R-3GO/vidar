#load torch
import torch

#import numpy
import numpy as np

#load model related lib
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

#load custom env
from utils.EvalGym import SqlEvaluationGym




def run_evaluation():
    #Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "model/final_model.zip"

    #Target configurations -> implement command line params later
    TARGET_URL = "http://192.168.201.132/bWAPP/sqli_11.php?title=1&action=search"
    TARGET_PARAM = "title"
    COOKIES = {'security_level':'0',
           'PHPSESSID':'8k6kkuqua14d4qpjfg7t6gpu5i'}

    #Environment Settings
    EPISODE_LENGTH = 512

    #Environment Configurations
    print("[+] INITIALIZING EVALUATION ENVIRONMENT....")
    env = SqlEvaluationGym(
        target_url=TARGET_URL,
        target_param=TARGET_PARAM,
        cookies=COOKIES,
        episode_length=EPISODE_LENGTH,
        dbms="sqlite",
        info_index=2,
    )

    env = Monitor(env)


    #Loading the model
    print(f"[+] Loading Trained Model from {MODEL_PATH}...")

    try:
        model = RecurrentPPO.load(MODEL_PATH, env=env, device=DEVICE)
    except Exception as exc:
        print(f"[-] Failed to load model: {exc!r}")
        return
    
    #Run the episode
    print(f"[+] Starting live evaluation against target....")
    obs, info = env.reset()

    # RecurrentPPO requires maintaining the LSTM hidden states
    lstm_states = None

    episode_starts = np.ones((1,), dtype=bool)

    done = False
    truncated = False
    total_reward = 0.0
    step_count = 0

    while not (done or truncated):
        #predict the next action deterministically
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )

        #take the step and gather obs. 
        obs, reward, done, truncated, info = env.step(action=action)

        total_reward += reward
        step_count += 1
        
        #print the agents action for monitoring and evaluation
        op_name = env.unwrapped.op_name
        payload = env.unwrapped.last_payload
        response = env.unwrapped.last_response
        #print(f"[*] Step: {step_count:03d} | Action: {op_name} | Reward: {reward:+.3f} | Total Accumulated Reward: {total_reward:.3f}")

        episode_starts = np.zeros((1,), dtype=bool)

    #print the final evaluation report at episode end. 
    if done:
        print(f"[+] Episode completed after {step_count} steps...")
    else:
        print(f"[-] Episode truncated after {step_count} steps...")
    print(f"[*] Total accumulated reward; {total_reward:.3f}")

if __name__ == "__main__":
    run_evaluation()