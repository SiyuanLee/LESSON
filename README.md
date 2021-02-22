# Learning Subgoal Representations with Slow Dynamics
We propose a slowness objective to effectively learn the subgoal representation
for goal-conditioned hierarchical reinforcement learning. [Our paper](https://openreview.net/pdf?id=wxRwhSdORKG) is accepted by ICLR 2021. 

The python dependencies are as follows.
* Python 3.6 or above
* [PyTorch](https://pytorch.org/)
* [Gym](https://gym.openai.com/)
* [Mujoco](https://www.roboti.us)

Run the codes with ``python train_hier_sac.py``. The tensorboard files are saved in the ``runs`` folder and the 
trained models are saved in the ``saved_models`` folder.
