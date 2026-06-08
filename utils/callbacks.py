import numpy as np

from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback, BaseCallback
from .actions import atomSqliteDict, operationsDict

import os
from datetime import datetime
import csv


class EpisodeLimitCallback(BaseCallback):
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None and np.any(self.locals["dones"]):
            self.episode += 1
            # if self.verbose:
            #     print(f"[*] Episode {self.episode} complete!")
        return self.episode < self.max_episodes
    
class AsymptoticConvergence(BaseCallback):
    def __init__(self, 
                 window_size=20,
                 epsilon=0.2,
                 min_episodes=50,
                 verbose=0):
    
        super().__init__(verbose)
        self.window_size = window_size
        self.epsilon = epsilon
        self.min_episodes = min_episodes

        self.episode_rewards = []
        self.running_rewards = {}

        #self.current_episode_reward = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals.get("infos", None)

        n_env = len(dones)

        for env_i in range(n_env):
            if env_i not in self.running_rewards:
                self.running_rewards[env_i] = 0.0
            
            self.running_rewards[env_i] += float(rewards[env_i])

            if dones[env_i]:
                ep_reward = self.running_rewards[env_i]
                self.episode_rewards.append(ep_reward)

                if self.verbose:
                    info = infos[env_i]
                    num_unique_payloads = info.get("num_unique_payloads")
                    sql = info.get("sql")
                    response_str = info.get("response_str")

                    # print(f"\n[*] Env {env_i} finished episode")
                    # print(f"[*] Training: Number of unique Payloads: {num_unique_payloads}")
                    # print(f"[*] Training: Last sent Payload / Response: {sql} / {response_str}")
                    # print(f"[*] Episode {len(self.episode_rewards)}, Reward: {self.running_rewards[env_i]}")
            
            self.running_rewards[env_i] = 0.0

            #ensure minimum episodes
            if len(self.episode_rewards) >= self.min_episodes:
            
                if len(self.episode_rewards) >= self.window_size *2:
                    #mean reward of the recent window
                    recent = np.mean(self.episode_rewards[-self.window_size:])
                    #mean reward of the previous window
                    previous = np.mean(self.episode_rewards[-2*self.window_size:-self.window_size])

                    diff = abs(recent - previous)

                    if self.verbose:
                        #print(f"[*] Check asymptotic: previous={previous:.3f}, recent={recent:.3f}, difference={diff:.3f}")

                        if diff < self.epsilon:
                            print(f"[+] Asymptotic Convergence found!")
                            return False
        return True
    
def verbose_evaluate(model, env, version_number, episodes=15, deterministic=True, stage=None, ):
    rewards = []

    log_dir = "model/eval_logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{timestamp}_{version_number}_eval_stage_{stage}.txt")
    csv_path = os.path.join(log_dir, f"{timestamp}_{version_number}_eval_stage_{stage}.csv")

    with open(csv_path, "w", newline="", encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "columns", "tables", "executes", "steps", "total_reward"])

    with open(log_path, "w", encoding='utf-8') as log:
        log.write(f"==== Evaluation Log Stage {stage} ====\n")
        log.write(f"Timestamp: {timestamp}\n\n")

        for ep in range(episodes):
            execs = 0
            obs, _ = env.reset()
            
            # --- CRITICAL RECURRENT UPDATES ---
            # Reset LSTM hidden states for a new episode
            lstm_states = None
            # Set mask to True to indicate the start of a sequence
            episode_starts = np.ones((1,), dtype=bool)
            # ----------------------------------

            terminated = False
            truncated = False
            ep_reward = 0
            step = 0
            
            print("-" * 60)
            print(f"Episode {ep+1}")

            while not (terminated or truncated):
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts, 
                    deterministic=deterministic
                )
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_starts = np.zeros((1,), dtype=bool)

                ep_reward += reward
                step += 1

                

                try:
                    op_choice = int(action)
                    op_name = operationsDict[op_choice]
                    if op_name == "EXECUTE":
                        execs += 1
                    line = f"step={step:03d}, agent_action={action}, tr_action=Operation: {op_name}, reward={reward:.3f},\n'sql': {info['sql']}, \n'response_str': {info['response_str']}, 'num_unique_payloads': {info['num_unique_payloads']},\n"
                except Exception as exc:
                    line = f"step={step:03d}, agent_action=[{action}], reward={reward:.3f}, info={info}\n"

                log.write(line)

                if terminated or truncated:
                    break
            

            rewards.append(ep_reward)
            print(f"Episode {ep+1} total reward: {ep_reward:.3f}")
            with open(csv_path, "a", newline="", encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([ep + 1, env.unwrapped.target_column_count, len(env.unwrapped.EXTRACTED_TABLES), execs, env.unwrapped.current_step, ep_reward])

            log.write(f"Episode {ep+1} total reward: {ep_reward:.3f}\n")
            log.write(f"[+] Running Mean: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}\n\n")

        return np.mean(rewards), np.std(rewards)