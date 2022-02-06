# Discussion

We believe that using a MainAgent which fingerprints the adversary and assigns a defending agent profile, is likely the best approach (assuming the agents it selects are “optimal”) due to the red agent’s behaviour not changing mid-episode and that the first few steps (used for fingerprinting) set as “Sleep” for the blue agent does not downgrade the performance. We considered informing the MainAgent once an episode ends (with agent.end_episode()), however we felt this would not accurately represent the challenge (as it was not in the original evaluation.py file) and as a result the fingerprinting does not use this information.

We also speculate that after fingerprinting the red agent, the best strategy for MainAgent may be to call an ensemble of different B_line and Meander blue agent models. However, given that we implemented a single approach, this was not possible.

To fingerprint the red agent, we sum the past two 52-bit observations and hard-coded them into the MainAgent. We also added some memory to the MainAgent so that in the rare case where it fingerprints Sleep when it shouldn't, it remembers the previous agent it assigned before the sleep (if the sum over the observation vector is greater than 0, i.e activity is visible) and reverts to it if this occurred in the past 3 steps.

        sleep_fingerprinted = [0] * 52
        meander_fingerprinted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bline_fingerprinted = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bline_fingerprinted_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
Finally, for the B_line blue agent, we reduced the possible random actions in our epsilon greedy exploration to ones which may be useful. This signaficantly improved our training.

        def get_action(self, observation, action_space=None):
                if np.random.random() > self.epsilon:
                        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                        actions = self.q_eval.forward(state)
                        action = T.argmax(actions).item()
                else:
                        #action = np.random.choice(self.action_space)
                        possibly_useful_actions_bline = [0,1,4,5,9,10,11,17,18,22,23,24,30,31,35,36,37]
                        action = random.choice(possibly_useful_actions_bline)


Overall, we feel that version 1 served as a good introduction to the challenge, and we look forward to version 2.



# Results

Our results outperform BlueReactRestoreAgent and BlueReactRemoveAgent. This was checked in benchmarks.py

### evaluation.py on 10 episodes, with random.seed(1) for reproducibility:

The full output of the evaluation is in MainAgent_10episodes.txt.

*30 length episodes*
1. steps: 30, adversary: B_lineAgent, mean: -7.799999999999994, standard deviation 1.4142135623730951
2. steps: 30, adversary: RedMeanderAgent, mean: -5.000000000000001, standard deviation 7.554248252914826
3. steps: 30, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

*50 length episodes*
1. steps: 50, adversary: B_lineAgent, mean: -11.999999999999988, standard deviation 3.675746333890725
2. steps: 50, adversary: RedMeanderAgent, mean: -15.899999999999999, standard deviation 19.994721525664946
3. steps: 50, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

*100 length episodes*
1. steps: 100, adversary: B_lineAgent, mean: -19.500000000000018, standard deviation 3.945461527713438
2. steps: 100, adversary: RedMeanderAgent, mean: -69.99999999999996, standard deviation 35.567775677805095
3. steps: 100, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

### evaluation.py on 100 episodes, with random.seed(1) for reproducibility:

The full output of the evaluation is in MainAgent_100episodes.txt.

*30 length episodes*
1. steps: 30, adversary: B_lineAgent, mean: -8.809999999999995, standard deviation 4.678545487519375
2. steps: 30, adversary: RedMeanderAgent, mean: -5.6899999999999995, standard deviation 5.591290124159652
3. steps: 30, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

*50 length episodes*
1. steps: 50, adversary: B_lineAgent, mean: -14.559999999999992, standard deviation 12.409771321117265
2. steps: 50, adversary: RedMeanderAgent, mean: -11.92, standard deviation 11.845026233176801
3. steps: 50, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

*100 length episodes*
1. steps: 100, adversary: B_lineAgent, mean: -23.92, standard deviation 20.69269133649544
2. steps: 100, adversary: RedMeanderAgent, mean: -45.449999999999974, standard deviation 42.93391092214526
3. steps: 100, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

### evaluation.py on 1000 episodes, with random.seed(1) for reproducibility:

The full output of the evaluation is in MainAgent_1000episodes.txt.

*30 length episodes*
1. steps: 30, adversary: B_lineAgent, mean: -8.989999999999995, standard deviation 5.08623335777762
2. steps: 30, adversary: RedMeanderAgent, mean: -5.034, standard deviation 5.7042951346317095
3. steps: 30, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

*50 length episodes*
1. steps: 50, adversary: B_lineAgent, mean: -13.63899999999999, standard deviation 10.273783608273527
2. steps: 50, adversary: RedMeanderAgent, mean: -12.956999999999999, standard deviation 15.101457049073282
3. steps: 50, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

*100 length episodes*
1. steps: 100, adversary: B_lineAgent, mean: -27.204999999999995, standard deviation 28.246714000956803
2. steps: 100, adversary: RedMeanderAgent, mean: -55.54979999999996, standard deviation 49.9558970654277
3. steps: 100, adversary: SleepAgent, mean: 0.0, standard deviation 0.0

# Cheat check

We implemented cheat_check.py to shuffle the order in which the red agents will appear, otherwise the MainAgent would be trivial. Cheat_check.py then sums over all the scores (therefore it would need to be divided by the number of episodes for it to be comparable to evaluation.py). We noticed no statistical difference between cheat_check.py and evaluation.py outputs.

### cheat_check.py on 10 episodes, with random.seed(1) for reproducibility:

*30 length episodes*
{'Sleep': 0.0, 'B_line': -75.99999999999996, 'Meander': -38.0}

*50 length episodes*
{'Sleep': 0.0, 'B_line': -109.99999999999984, 'Meander': -76.99999999999999}

*100 length episodes*
{'Sleep': 0.0, 'B_line': -218.00000000000017, 'Meander': -572.9999999999998}

### cheat_check.py on 100 episodes, with random.seed(1) for reproducibility:

*30 length episodes*
{'Sleep': 0.0, 'B_line': -919.9999999999982, 'Meander': -590.9999999999997}

*50 length episodes*
{'Sleep': 0.0, 'B_line': -1391.9999999999973, 'Meander': -1373.0000000000027}

*100 length episodes*
{'Sleep': 0.0, 'B_line': -2630.0000000000014, 'Meander': -5278.999999999988}
