import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DelayedMABEnv(gym.Env):
    """
    Delayed Multi-Armed Bandit Environment
    
    Tests temporal task inference through a 6-phase structure with intermediate decision:
    
    1. SIGNAL PHASE 1: Agent receives first signal (can infer intermediate optimal arm)
    2. DISTRACTOR PHASE 1: No information, zero rewards regardless of actions
    3. INTERMEDIATE DECISION: Agent can act on signal 1 only (suboptimal but rewarding)
    4. SIGNAL PHASE 2: Agent receives second signal (needed for final optimal arm)
    5. DISTRACTOR PHASE 2: No information, zero rewards regardless of actions
    6. FINAL DECISION: Agent chooses arms, optimal choice requires both signals
    
    Key design principles:
    - Intermediate decision allows reward from signal 1 alone (intermediate optimum)
    - Final optimal arm ONLY determined by combining info from BOTH signal phases
    - Decision phases are short to force memory usage over exploration
    - Distractor phases prevent temporal correlation between signals and decisions
    - Total episode length: exactly 100 timesteps
    """
    
    def __init__(self, n_bandits=5, signal1_length=25, distractor1_length=15,
                 intermediate_decision_length=10, signal2_length=25, distractor2_length=15, 
                 final_decision_length=10, max_episode_length=100):
        super().__init__()
        
        self.n_bandits = n_bandits
        self.signal1_length = signal1_length
        self.distractor1_length = distractor1_length
        self.intermediate_decision_length = intermediate_decision_length
        self.signal2_length = signal2_length
        self.distractor2_length = distractor2_length
        self.final_decision_length = final_decision_length
        self.max_episode_length = max_episode_length
        self.act_dim = n_bandits
        
        # Verify total length equals 100
        total_length = (signal1_length + distractor1_length + intermediate_decision_length + 
                       signal2_length + distractor2_length + final_decision_length)
        assert total_length == 100, f"Phase lengths must sum to 100, got {total_length}"
        
        # Phase boundaries
        self.signal1_end = signal1_length
        self.distractor1_end = self.signal1_end + distractor1_length
        self.intermediate_decision_end = self.distractor1_end + intermediate_decision_length
        self.signal2_end = self.intermediate_decision_end + signal2_length
        self.distractor2_end = self.signal2_end + distractor2_length
        self.final_decision_end = self.distractor2_end + final_decision_length
        
        # Action space: choose between bandits or "wait" 
        self.action_space = spaces.Discrete(self.act_dim)
        
        # Observation space: [arm_values(n_bandits), phase_one_hot(3)]
        # arm_values: Signal1=base_payoffs, Signal2=multipliers, Others=zeros
        # Phase one-hot: [is_signal, is_distractor, is_decision]
        # Total dimensions: n_bandits + 3
        self.observation_space = spaces.Box(
            shape=(n_bandits + 3,),
            low=np.array([-10.0] * n_bandits + [0.0, 0.0, 0.0]),
            high=np.array([10.0] * n_bandits + [1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.phase = 0  # 0=signal1, 1=distractor1, 2=intermediate_decision, 3=signal2, 4=distractor2, 5=final_decision
        self.last_reward = 0.0
        self.cumulative_signal = 0.0
        self.signal1_cumulative = 0.0  # Track signal 1 separately for intermediate decision
        
        # Task parameters (set by BAMDP wrapper)
        self.base_payoffs = np.zeros(n_bandits)     # Signal 1: base payoffs for each arm
        self.multipliers = np.ones(n_bandits)       # Signal 2: multipliers for each arm
        self.final_payoffs = np.zeros(n_bandits)    # Final payoffs: base_payoffs * multipliers
        self.intermediate_optimal_arm = 0           # Best arm based on base payoffs only
        self.final_optimal_arm = 0                  # Best arm based on final payoffs
        self.arm_std = 0.1                          # Standard deviation for all arms
        
        self.reset()
    
    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.phase = 0
        self.last_reward = 0.0
        
        # Default task parameters (overridden by BAMDP wrapper)
        # Generate random base payoffs and multipliers
        self.base_payoffs = np.random.uniform(1, 5, self.n_bandits)
        self.multipliers = np.random.choice([-1, 1], self.n_bandits)  # Random +1 or -1 multipliers
        self._update_optimal_arms()
        self._set_arm_rewards()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        # Update phase based on timestep
        old_phase = self.phase
        self.phase = self._get_current_phase()
        
        # Phase-specific logic
        if self.phase == 0:  # Signal Phase 1
            reward = self._get_signal1_reward()
            
        elif self.phase == 1:  # Distractor Phase 1  
            reward = 0.0  # No information, no reward
            
        elif self.phase == 2:  # Intermediate Decision Phase
            reward = self._get_intermediate_decision_reward(action)
            
        elif self.phase == 3:  # Signal Phase 2
            reward = self._get_signal2_reward()
            
        elif self.phase == 4:  # Distractor Phase 2
            reward = 0.0  # No information, no reward
            
        elif self.phase == 5:  # Final Decision Phase
            reward = self._get_final_decision_reward(action)
        
        # No need to track cumulative signal - information is now in observations
            
        self.last_reward = reward
        
        # Episode termination
        if self.current_step >= self.final_decision_end or self.current_step >= self.max_episode_length:
            terminated = True
            
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_current_phase(self):
        """Determine current phase based on timestep"""
        if self.current_step <= self.signal1_end:
            return 0  # Signal Phase 1
        elif self.current_step <= self.distractor1_end:
            return 1  # Distractor Phase 1
        elif self.current_step <= self.intermediate_decision_end:
            return 2  # Intermediate Decision Phase
        elif self.current_step <= self.signal2_end:
            return 3  # Signal Phase 2
        elif self.current_step <= self.distractor2_end:
            return 4  # Distractor Phase 2
        else:
            return 5  # Final Decision Phase
    
    def _get_signal1_reward(self):
        """No reward during signal phases - information is in observation"""
        return 0.0
    
    def _get_signal2_reward(self):
        """No reward during signal phases - information is in observation"""
        return 0.0
    
    def _get_intermediate_decision_reward(self, action):
        """Provide reward based on base payoffs only (intermediate decision)"""
        # Use base payoffs for intermediate decision
        base_reward = self.base_payoffs[action]
        return np.random.normal(base_reward, self.arm_std)
    
    def _get_final_decision_reward(self, action):
        """Provide reward based on final payoffs (base_payoffs * multipliers)"""
        # Use final payoffs for final decision
        final_reward = self.final_payoffs[action]
        return np.random.normal(final_reward, self.arm_std)
    
    def _update_optimal_arms(self):
        """Determine intermediate and final optimal arms"""
        # Intermediate optimal arm: highest base payoff
        self.intermediate_optimal_arm = np.argmax(self.base_payoffs)
        
        # Calculate final payoffs and determine final optimal arm
        self.final_payoffs = self.base_payoffs * self.multipliers
        self.final_optimal_arm = np.argmax(self.final_payoffs)
    
    def _set_arm_rewards(self):
        """Set reward distributions for each arm (now handled in reward methods)"""
        # Reward distributions are now handled directly in the decision reward methods
        # to differentiate between intermediate and final optimal arms
        pass
    
    def _get_observation(self):
        """Get current observation with arm values and phase encoding"""
        
        # Create arm values based on current phase
        arm_values = np.zeros(self.n_bandits, dtype=np.float32)
        
        if self.phase == 0:  # Signal Phase 1: Show base payoffs
            arm_values = self.base_payoffs.copy()
        elif self.phase == 3:  # Signal Phase 2: Show multipliers
            arm_values = self.multipliers.copy()
        # Other phases: arm_values remain zeros
        
        # Create one-hot phase encoding: [is_signal, is_distractor, is_decision]
        phase_one_hot = np.zeros(3, dtype=np.float32)
        if self.phase in [0, 3]:  # Signal phases (signal1=0, signal2=3)
            phase_one_hot[0] = 1.0  # is_signal
        elif self.phase in [1, 4]:  # Distractor phases (distractor1=1, distractor2=4)
            phase_one_hot[1] = 1.0  # is_distractor
        elif self.phase in [2, 5]:  # Decision phases (intermediate=2, final=5)
            phase_one_hot[2] = 1.0  # is_decision
        
        # Combine arm values and phase encoding
        obs = np.concatenate([arm_values, phase_one_hot], dtype=np.float32)
        
        return obs
    
    def _get_step_in_current_phase(self):
        """Get current step within the current phase"""
        if self.phase == 0:  # Signal Phase 1
            return self.current_step
        elif self.phase == 1:  # Distractor Phase 1
            return self.current_step - self.signal1_end
        elif self.phase == 2:  # Intermediate Decision Phase
            return self.current_step - self.distractor1_end
        elif self.phase == 3:  # Signal Phase 2
            return self.current_step - self.intermediate_decision_end
        elif self.phase == 4:  # Distractor Phase 2
            return self.current_step - self.signal2_end
        else:  # phase == 5: Final Decision Phase
            return self.current_step - self.distractor2_end
    
    def _get_info(self):
        """Get environment info"""
        return {
            'phase': self.phase,
            'intermediate_optimal_arm': self.intermediate_optimal_arm,
            'final_optimal_arm': self.final_optimal_arm,
            'base_payoffs': self.base_payoffs.copy(),
            'multipliers': self.multipliers.copy(),
            'final_payoffs': self.final_payoffs.copy(),
            'step_in_phase': self._get_step_in_current_phase(),
            'total_steps': self.current_step
        }
    
    def set_task_params(self, base_payoffs, multipliers, arm_reward_scale=1.0):
        """Set task parameters - called by BAMDP wrapper"""
        self.base_payoffs = np.array(base_payoffs, dtype=np.float32)
        self.multipliers = np.array(multipliers, dtype=np.float32)
        self._update_optimal_arms()
        self._set_arm_rewards()


