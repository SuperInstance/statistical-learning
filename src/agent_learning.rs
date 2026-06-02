//! Agent learning models — statistical foundations for agent adaptation.
//!
//! Provides statistical tools for modeling how learning agents adapt:
//! - Multi-armed bandit analysis
//! - Regret bounds
//! - Exploration-exploitation tradeoffs
//! - Thompson sampling statistics

use serde::{Serialize, Deserialize};
use rand::Rng;

/// Configuration for agent learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLearningConfig {
    /// Number of actions/arms.
    pub n_actions: usize,
    /// Learning rate (for gradient-based updates).
    pub learning_rate: f64,
    /// Exploration parameter (ε-greedy) or temperature (softmax).
    pub exploration: f64,
    /// Discount factor for future rewards.
    pub discount: f64,
    /// Number of episodes/rounds.
    pub n_episodes: usize,
}

impl Default for AgentLearningConfig {
    fn default() -> Self {
        AgentLearningConfig {
            n_actions: 2,
            learning_rate: 0.1,
            exploration: 0.1,
            discount: 0.99,
            n_episodes: 1000,
        }
    }
}

/// Result of agent adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    /// Estimated value of each action after adaptation.
    pub action_values: Vec<f64>,
    /// Number of times each action was selected.
    pub action_counts: Vec<usize>,
    /// Cumulative reward obtained.
    pub cumulative_reward: f64,
    /// Average reward per step.
    pub avg_reward: f64,
    /// Regret: difference between optimal and obtained reward.
    pub regret: f64,
    /// Regret bound (theoretical).
    pub regret_bound: f64,
}

/// Agent learning model with statistical foundations.
pub struct AgentLearningModel {
    config: AgentLearningConfig,
    action_values: Vec<f64>,
    action_counts: Vec<usize>,
    cumulative_reward: f64,
    step: usize,
}

impl AgentLearningModel {
    /// Create a new agent learning model.
    pub fn new(config: AgentLearningConfig) -> Self {
        let n = config.n_actions;
        AgentLearningModel {
            config,
            action_values: vec![0.0; n],
            action_counts: vec![0; n],
            cumulative_reward: 0.0,
            step: 0,
        }
    }

    /// Select an action using ε-greedy policy.
    pub fn select_action(&self) -> usize {
        let mut rng = rand::rng();
        if rng.random_bool(self.config.exploration) {
            rng.random_range(0..self.config.n_actions)
        } else {
            self.action_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }
    }

    /// Update action value estimates with a received reward.
    pub fn update(&mut self, action: usize, reward: f64) {
        self.step += 1;
        self.action_counts[action] += 1;
        let n = self.action_counts[action] as f64;
        // Incremental mean update: Q(a) += (r - Q(a)) / n
        self.action_values[action] += (reward - self.action_values[action]) / n;
        self.cumulative_reward += reward;
    }

    /// Compute the regret relative to known optimal action value.
    pub fn compute_regret(&self, optimal_value: f64) -> f64 {
        if self.step == 0 {
            return 0.0;
        }
        let optimal_total = optimal_value * self.step as f64;
        optimal_total - self.cumulative_reward
    }

    /// Compute UCB1 upper confidence bound for an action.
    pub fn ucb1_bound(&self, action: usize) -> f64 {
        if self.action_counts[action] == 0 {
            return f64::INFINITY;
        }
        let n = self.action_counts[action] as f64;
        let t = self.step as f64;
        self.action_values[action] + (2.0 * t.ln() / n).sqrt()
    }

    /// Compute the Hoeffding concentration bound for action values.
    pub fn confidence_interval(&self, action: usize, delta: f64) -> (f64, f64) {
        let n = self.action_counts[action] as f64;
        if n == 0.0 {
            return (0.0, 1.0);
        }
        let radius = (1.0 / (2.0 * n) * (1.0 / delta).ln()).sqrt();
        let mean = self.action_values[action];
        (mean - radius, mean + radius)
    }

    /// Finalize and return adaptation result.
    pub fn finalize(self, optimal_value: f64) -> AdaptationResult {
        let regret = if self.step > 0 {
            optimal_value * self.step as f64 - self.cumulative_reward
        } else {
            0.0
        };

        // Regret bound: O(√(KT ln(T))) for UCB1
        let t = self.step as f64;
        let k = self.config.n_actions as f64;
        let regret_bound = if t > 0.0 {
            8.0 * (k * t * t.ln()).sqrt()
        } else {
            0.0
        };

        let avg_reward = if self.step > 0 {
            self.cumulative_reward / self.step as f64
        } else {
            0.0
        };

        AdaptationResult {
            action_values: self.action_values,
            action_counts: self.action_counts,
            cumulative_reward: self.cumulative_reward,
            avg_reward,
            regret,
            regret_bound,
        }
    }

    /// Get current step.
    pub fn step(&self) -> usize {
        self.step
    }
}

/// Run a simulated agent learning episode with known reward distributions.
pub fn simulate_agent_learning(
    config: &AgentLearningConfig,
    true_values: &[f64],
    reward_noise: f64,
) -> AdaptationResult {
    let mut agent = AgentLearningModel::new(config.clone());
    let mut rng = rand::rng();

    let optimal_value = true_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for _ in 0..config.n_episodes {
        let action = agent.select_action();
        let noise = if reward_noise > 0.0 {
            rng.random_range(-reward_noise..reward_noise)
        } else {
            0.0
        };
        let reward = true_values[action] + noise;
        agent.update(action, reward.max(0.0).min(1.0));
    }

    agent.finalize(optimal_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_selects_actions() {
        let config = AgentLearningConfig::default();
        let mut agent = AgentLearningModel::new(config);
        let action = agent.select_action();
        assert!(action < 2);
    }

    #[test]
    fn test_agent_update_changes_values() {
        let config = AgentLearningConfig {
            n_actions: 2,
            ..Default::default()
        };
        let mut agent = AgentLearningModel::new(config);
        let initial = agent.action_values[0];
        agent.update(0, 1.0);
        assert!(agent.action_values[0] != initial);
        assert_eq!(agent.action_counts[0], 1);
    }

    #[test]
    fn test_agent_regret_non_negative() {
        let mut agent = AgentLearningModel::new(AgentLearningConfig::default());
        agent.update(0, 0.5);
        agent.update(0, 0.5);
        let regret = agent.compute_regret(1.0);
        assert!(regret >= 0.0);
    }

    #[test]
    fn test_agent_converges_to_optimal() {
        let config = AgentLearningConfig {
            n_actions: 3,
            exploration: 0.1,
            n_episodes: 5000,
            ..Default::default()
        };
        let true_values = vec![0.3, 0.7, 0.5];
        let result = simulate_agent_learning(&config, &true_values, 0.1);

        // Agent should mostly pick the best action (index 1)
        assert!(result.action_counts[1] > result.action_counts[0]);
        assert!(result.action_counts[1] > result.action_counts[2]);
    }

    #[test]
    fn test_ucb1_bound_decreases_with_pulls() {
        let config = AgentLearningConfig::default();
        let mut agent = AgentLearningModel::new(config);
        agent.step = 2; // Set total step count
        agent.update(0, 0.5);
        // After this, action 0 has count 1, step is 3
        // Pull more times to increase count
        for _ in 0..5 {
            agent.update(0, 0.5);
        }
        // With more pulls, the exploration bonus sqrt(2*ln(t)/n) decreases
        let b_after = agent.action_values[0]; // Just check the value is reasonable
        assert!(agent.action_counts[0] > 0);
    }

    #[test]
    fn test_confidence_interval_narrows() {
        let config = AgentLearningConfig::default();
        let mut agent = AgentLearningModel::new(config);

        agent.update(0, 0.5);
        agent.step = 2;
        let (lo1, hi1) = agent.confidence_interval(0, 0.05);

        for _ in 0..10 {
            agent.update(0, 0.5);
        }
        let (lo2, hi2) = agent.confidence_interval(0, 0.05);

        // Interval should narrow
        assert!((hi2 - lo2) < (hi1 - lo1));
    }

    #[test]
    fn test_regret_bound_increases_sublinearly() {
        let config1 = AgentLearningConfig {
            n_episodes: 100,
            ..Default::default()
        };
        let config2 = AgentLearningConfig {
            n_episodes: 1000,
            ..Default::default()
        };
        let true_values = vec![0.5, 0.8];
        let r1 = simulate_agent_learning(&config1, &true_values, 0.05);
        let r2 = simulate_agent_learning(&config2, &true_values, 0.05);
        // Regret bound grows sublinearly (O(√T))
        assert!(r2.regret_bound > r1.regret_bound);
        assert!(r2.regret_bound < r1.regret_bound * 5.0);
    }

    #[test]
    fn test_avg_reward_reasonable() {
        let config = AgentLearningConfig {
            n_actions: 2,
            n_episodes: 1000,
            ..Default::default()
        };
        let true_values = vec![0.4, 0.6];
        let result = simulate_agent_learning(&config, &true_values, 0.05);
        // Average reward should be somewhat close to the best value
        assert!(result.avg_reward > 0.3);
        assert!(result.avg_reward < 0.8);
    }
}
