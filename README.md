# agar-py

Implementing agar.io to run locally as a Python process. This will optimize for our model being able to hook into the game state and learn in rapid iterations. We are developing a set of models to learn to play this game via Reinforcement Learning.

---

# TODO

### Done

- [x] Split via space bar
- [x] Shoot via `q` as we are currently using `w` for motion
- [x] Viruses splitting agents which are signficantly larger than the virus
- [x] Recombining agent cell parts
- [x] Eventually merge split cells back together
- [x] Have some intertia to bring cells together
- [x] Finish acceleration (Cam)
- [x] Make it such that an agent cannot split after a certain number of clock ticks not ms
- [x] Checks to see if AgentCells are overlapping
- [x] Eating mass
- [x] Split into many pieces when eats virus
- [x] Case where cells totally overlap (pick random angle)
- [x] Bug with eating parts of agents
- [x] Fix bug with movement of heuristic agent at edge of map (Sam)
- [x] Get bigger at edge of map -> should get scooched back into map boundary (Sam)
- [x] When you die you keep getting negative reward -> this should only happen once (Mak)
- [x] Create IAM users for each group member on ccabo@seas AWS account
- [x] Fix food bug (all the boys 😤)
- [x] Add GUI functionality to the training loop (Sam)
- [x] Mass decay (cam)
- [x] Infrastructure for saving model net params (maybe to a folder in github?) (cam)
- [x] Update rewards function to penalize being eaten (Mak)
- [x] Privacy.com on AWS account & billing alerts (Cam)
- [x] Setup infra on AWS EC2 (Salib)
- [x] Try running it locally and then via sagemaker in a notebook (Salib)
- [x] Refactor train file to not run on import (Mak)
- [x] Optimize state encoding runtime (Cam)
  - [x] Look into encoding state via pytorch -> distributed
- [x] Store encoded state in buffer; make sure this is not by reference (mak)

### Up next

- [ ] Test state encoding, make sure it is seemingly correct (Cam)
- [ ] Better parameterization for epsilon decay (and other hyperparams) (Cam)
- [ ] Look at more examples online (Cam)
- [ ] Write logs (or a subset?) to a file from train loop (Cam)
- [ ] Random actions until replay buffer is full (Mak)
  - [ ] Make sure to print at useful points (like when buffer is full and we start training)
  - [ ] Don't learn until the buffer is full
- [ ] Reach out to Vatsal for some help/suggestions (Mak)
- [ ] Analyze RL runtime bottleneck (local v. AWS v. GPU) (Mak)
- [ ] Test different state encodings (Cam & Mak)
- [ ] Convolutional jawn (Sam + Salib)
  - [ ] Rip the CNN 🎆
  - [ ] Model
  - [ ] Getting screenshots in the train loop (not just GUI)
    - [ ] Figure out a good frequency for this

### Lower priority

- [ ] Smarter datastructure for checking collisions
- [ ] Move with mouse? Could move in directions other than the 8 we currently support
- [ ] Shooting viruses
- [ ] safe guards for when you're dead you don't have actions
  - currently assumes game state properly keeps track which is ok i guess
  - edge case is manualy start a game where agentmodel.done = True
- [ ] Figure out what logs / graphs we will want on the writeup
- [ ] Have a policy net and a target net
