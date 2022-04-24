import torch


class CartpoleTipCost:
    """
    Cost is euclidean distance of cartpole end effector from some goal position in R^2
    """
    def __init__(self, goal, l=1.):
        self.goal = goal
        self.l = l
        self.uncertainty_cost = 0.0
        self.iter = 0.0

    def set_goal(self, goal):
        self.goal = goal

    def set_max_std(self, max_std):
        self.max_std = max_std

    def check_rope_collision(self, state):
        " checks if rope collides with object"
        N, T, _ = state.shape

        zeros = torch.zeros(N, T, device=state.device)
        ones = torch.ones(N, T, device=state.device)

        base_x = state[:, :, 0]
        tip_x = state[:, :, 1]
        tip_y = state[:, :, 2]
        rope_theta = torch.atan2(tip_y, tip_x - base_x)
        goal_theta = torch.atan2(torch.tensor(self.goal[1], device=state.device),
                                 self.goal[0] - base_x)
        # If approx same theta, then there may be a collision
        collision = torch.where(torch.abs(rope_theta - goal_theta) < 0.2, ones, zeros)

        if N == 1:
            print('angle check')
            print(collision)

        # If the length is greater than the length of the rope (minus the mass), then no collision
        length = torch.sqrt((tip_x - base_x)**2 + tip_y**2)
        base_to_target_d = torch.sqrt((self.goal[0] - base_x)**2 + (self.goal[1])**2)

        # When goal is further than length, clearly rope cannot collide
        collision = torch.where(base_to_target_d > 0.9 * length, zeros, collision)
        if N == 1:
            print('length check')
            print(collision)

        # When base to target id is too low
        collision = torch.where(base_to_target_d < 0.2, ones, collision)
        if N == 1:
            print('base check')
            print(collision)

        return collision

    def compute_cost(self, state, actions=None, verbose=False):
        N, T, _ = state.shape
        base_x = state[:, :, 0]
        tip_x = state[:, :, 1]
        tip_y = state[:, :, 2]

        uncertainty = state[:, :, 5:]

        # Target cost -- 0 if goal reached in horizon, else it is distance of the end state from goal
        dist_2_goal = torch.sqrt((self.goal[0] - tip_x) ** 2 + (self.goal[1] - tip_y) ** 2)

        collisions = self.check_rope_collision(state)

        if N == 1:
            print(collisions)

        # Get components where this is zero
        at_target = (dist_2_goal < 0.1).nonzero()
        in_collision = (collisions == 1).nonzero()

        from_centre_cost = (base_x).clamp(min=1.5) - 1.5
        vel_cost = state[:, :, 4:5].abs().sum(dim=2)
        uncertainty_cost = uncertainty.clone() ** 2

        for t in range(T - 1, -1, -1):
            # count down
            at_target_idx = (at_target[:, 1] == t).nonzero()
            at_target_t = at_target[at_target_idx, 0]

            in_collision_idx = (in_collision[:, 1] == t).nonzero()
            in_collision_t = in_collision[in_collision_idx, 0]

            collisions[at_target_t, t + 1:] = 0
            dist_2_goal[at_target_t, t + 1:] = 0
            dist_2_goal[in_collision_t, t:] = 10 * 0.9**(t)
            #uncertainty_cost[in_collision_t, t:] = uncertainty_cost[in_collision_t, t - 1].unsqueeze(2)
            #uncertainty_cost[at_target_t, t + 1:] = 0


        if N == 1:
            print(dist_2_goal)
        alphas = torch.arange(0, T, device=state.device)
        gammas = torch.pow(torch.tensor(.9, device=state.device), alphas)
        from_centre_cost *= gammas
        cost = dist_2_goal + 10.0 * from_centre_cost + 1e-5 * vel_cost# + 100 * collision_cost

        #uncertainty_cost = uncertainty_cost * gammas.view(1, T)
        uncertainty_cost = uncertainty_cost @ torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01], device=uncertainty_cost.device).unsqueeze(1)
        uncertainty_cost = uncertainty_cost.sum(dim=1).squeeze(1)
        uncertainty_cost = uncertainty_cost - uncertainty_cost.mean()

        if False:
            print(uncertainty_cost.mean())
            print(uncertainty_cost.max())
            print(uncertainty_cost.min())
            print(cost.sum(dim=1).mean())
            print(cost.sum(dim=1).max())
            print(cost.sum(dim=1).min())

        uncertainty_weight = 0.2 * self.iter

        return cost.sum(dim=1) + uncertainty_weight * uncertainty_cost


# TODO: Add a cartpole cost function that makes it swing up and reach the point with low speed for slackness to follow
# TODO: Add a Ball cost function that makes Ball go into cup by aiming for robot position
