import abc


class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    def get_action(self, obs_np, **kwargs):
        pass

    @abc.abstractmethod
    def get_actions(self, obs_np, **kwargs):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class Maze2dPolicy(Policy):
    def __init__(self, pipeline, pipeline_kwargs, fix_batch_observations=None):
        self.pipeline = pipeline
        self.pipeline_kwargs = pipeline_kwargs
        self.plan_hor = pipeline_kwargs['planning_horizon']
        self.t = None
        self.batch_observations = None
        self.fix_batch_observations = fix_batch_observations # for debugging predefined plan

    def get_action(self, obs_np, goal_np):
        actions = self.get_actions(obs_np[None], goal_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np, goal_np):
        if self.t == 0:
            if self.fix_batch_observations is None:
                conditions = {
                    0: obs_np,
                    self.plan_hor - 1: goal_np,
                }
                pipe_result = self.pipeline(
                    conditions, return_chain_observations=False, **self.pipeline_kwargs
                ) 

                # [batch_size x plan_hor x num_plan x observation_dim]
                self.batch_observations = pipe_result.observations[:, 0, :, :]
            else:
                self.batch_observations = self.fix_batch_observations

        if self.t < self.plan_hor - 1:
            next_waypoint = self.batch_observations[:, self.t + 1]
        else:
            next_waypoint = self.batch_observations[:, -1].copy()
            next_waypoint[:, 2:] = 0

        actions = next_waypoint[:, :2] - obs_np[:, :2] + (next_waypoint[:, 2:] - obs_np[:, 2:])
        self.t += 1 
        return actions

    def reset(self):
        self.t = 0
    

class LocomotionPolicy(Policy):
    def __init__(self, pipeline, pipeline_kwargs):
        self.pipeline = pipeline
        self.pipeline_kwargs = pipeline_kwargs
        self.t = None

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np):
        conditions = {
            0: obs_np,
        }
        pipe_result = self.pipeline(
            conditions, return_chain_observations=False, **self.pipeline_kwargs
        ) 
        self.t += 1 
        return pipe_result.action             

    def reset(self):
        self.t = 0