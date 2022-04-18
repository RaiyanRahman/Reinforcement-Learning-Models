import torch
import torch.nn as nn
import numpy as np
from utils.test_env import EnvTest
from q2_schedule import LinearExploration, LinearSchedule
from q3_linear_torch import Linear

from configs.q7_dddqn import config


class DDDQN(Linear):
    """
    Implementing the Duelling Double Deep Q Network.
    """

    def initialize_models(self):
        """The input to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        The network has the following architecture:
            - Conv2d with 32 8x8 filters and stride 4 + ReLU activation
            - Conv2d with 64 4x4 filters and stride 2 + ReLU activation
            - Conv2d with 64 3x3 filters and stride 1 + ReLU activation
            - Flatten
            - Linear with output 512. What is the size of the input?
                you need to calculate this img_height, img_width, and number of filter.
            - Relu

        One head gives the state value function V:
            - Linear with output 256 + ReLU activation
            - Linear with output 1
        One head gives the action advantage value A:
            - Linear with output 256 + ReLU activation
            - Linear with output num_actions

        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        strides = np.array([4, 2, 1])  # The stride size for every conv2d layer
        filter_sizes = np.array([8, 4, 3])  # The filter size for every conv2d layer
        numb_filters = np.array([32, 64, 64])  # number of filters for every conv2d layer

        # Set input size and paddings.
        input_size = n_channels * self.config.state_history
        paddings = ((strides - 1) * img_height - strides + filter_sizes) // 2

        # Calculate the shape of the input to the FC layer.
        fc_input_h = img_height
        fc_input_w = img_width
        for i in range(3):
            fc_input_h = ((fc_input_h + (2 * paddings[i]) - filter_sizes[i]) // strides[i]) + 1
            fc_input_w = ((fc_input_w + (2 * paddings[i]) - filter_sizes[i]) // strides[i]) + 1

        # Q network initialization.
        self.q_network = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=numb_filters[0],
                      kernel_size=filter_sizes[0],
                      stride=strides[0],
                      padding=paddings[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=numb_filters[0],
                      out_channels=numb_filters[1],
                      kernel_size=filter_sizes[1],
                      stride=strides[1],
                      padding=paddings[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=numb_filters[1],
                      out_channels=numb_filters[2],
                      kernel_size=filter_sizes[2],
                      stride=strides[2],
                      padding=paddings[2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=fc_input_h * fc_input_w * numb_filters[2],
                      out_features=512),
            nn.ReLU()
        )

        # Target network initialization.
        self.target_network = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=numb_filters[0],
                      kernel_size=filter_sizes[0],
                      stride=strides[0],
                      padding=paddings[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=numb_filters[0],
                      out_channels=numb_filters[1],
                      kernel_size=filter_sizes[1],
                      stride=strides[1],
                      padding=paddings[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=numb_filters[1],
                      out_channels=numb_filters[2],
                      kernel_size=filter_sizes[2],
                      stride=strides[2],
                      padding=paddings[2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=fc_input_h * fc_input_w * numb_filters[2],
                      out_features=512),
            nn.ReLU()
        )

        self.V = nn.Linear(in_features=512, out_features=1).to(self.device)
        self.A = nn.Linear(in_features=512, out_features=num_actions).to(self.device)
        self.dddqn = True   # Flag for saving the V and A layers.

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)
        """

        # Rearrange the input tensor dimensions.
        in_state = state.permute(0, 3, 1, 2)
        # Get the initial network output.
        if network == 'q_network':
            out = self.q_network(in_state)
        else:
            out = self.target_network(in_state)
        # Calculate the Q values using the state-value function and the advantages.
        V = self.V(out)
        A = self.A(out)
        out = V + (A - torch.mean(A, dim=1, keepdim=True))
        return out


"""
Use DDDQN for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # train model
    model = DDDQN(env, config)
    model.run(exp_schedule, lr_schedule)
