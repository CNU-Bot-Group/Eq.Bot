import torch


class ESCNNEquivariantNetwork(torch.nn.Module):
    """
    This class represents an Equivariant Convolutional Neural Network (Equivariant CNN).

    The network is equivariant to a group of transformations, which can be either rotations or roto-reflections.
    The network consists of a sequence of equivariant convolutional layers, each followed by batch normalization, a ReLU activation function, and dropout. The number of output channels of the convolutional layers is the same for all layers.

    Methods:
        __init__: Initializes the ESCNNEquivariantNetwork instance.
        forward: Performs a forward pass through the network.
    """

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        kernel_size: int,
        group_type: str = "rotation",
        num_rotations: int = 4,
        num_layers: int = 1,
    ):
        """
        Initializes the ESCNNEquivariantNetwork instance.

        Args:
            in_shape (tuple): The shape of the input data. It should be a tuple of the form (in_channels, height, width).
            out_channels (int): The number of output channels of the convolutional layers.
            kernel_size (int): The size of the kernel of the convolutional layers.
            group_type (str, optional): The type of the group of transformations. It can be either "rotation" or "roto-reflection". Defaults to "rotation".
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.
            num_layers (int, optional): The number of convolutional layers. Defaults to 1.
        """
        super().__init__()
        from e2cnn import gspaces
        from e2cnn import nn as equivariant_nn

        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.num_rotations = num_rotations

        if group_type == "rotation":
            self.gspace = gspaces.Rot2dOnR2(num_rotations)
        elif group_type == "roto-reflection":
            self.gspace = gspaces.FlipRot2dOnR2(num_rotations)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")

        # If the group is roto-reflection, then the number of group elements is twice the number of rotations
        self.num_group_elements = (
            num_rotations if group_type == "rotation" else 2 * num_rotations
        )

        r1 = equivariant_nn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.in_channels
        )
        r2 = equivariant_nn.FieldType(self.gspace, [self.gspace.regular_repr] * out_channels)

        self.in_type = r1
        self.out_type = r2

        modules = [
            equivariant_nn.R2Conv(self.in_type, self.out_type, kernel_size),
            equivariant_nn.InnerBatchNorm(self.out_type, momentum=0.9),
            equivariant_nn.ReLU(self.out_type, inplace=True),
            equivariant_nn.PointwiseDropout(self.out_type, p=0.5),
        ]
        for _ in range(num_layers - 2):
            modules.append(
                equivariant_nn.R2Conv(self.out_type, self.out_type, kernel_size),
            )
            modules.append(
                equivariant_nn.InnerBatchNorm(self.out_type, momentum=0.9),
            )
            modules.append(
                equivariant_nn.ReLU(self.out_type, inplace=True),
            )
            modules.append(
                equivariant_nn.PointwiseDropout(self.out_type, p=0.5),
            )

        modules.append(
            equivariant_nn.R2Conv(self.out_type, self.out_type, kernel_size),
        )

        self.eqv_network = equivariant_nn.SequentialModule(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, num_group_elements).
        """
        from e2cnn.nn import GeometricTensor

        x = GeometricTensor(x, self.in_type)
        out = self.eqv_network(x)

        feature_map = out.tensor
        feature_map = feature_map.reshape(
            feature_map.shape[0],
            self.out_channels,
            self.num_group_elements,
            feature_map.shape[-2],
            feature_map.shape[-1],
        )

        group_activations = torch.mean(feature_map, dim=(1, 3, 4))

        return group_activations


class ESCNNSteerableNetwork(torch.nn.Module):
    """Estimate continuous planar orientations from steerable vector features."""

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        kernel_size: int = 9,
        group_type: str = "rotation",
        num_layers: int = 1,
    ):
        super().__init__()
        from escnn import gspaces
        from escnn import nn as steerable

        if group_type != "rotation":
            raise ValueError("The continuous encoder supports planar rotations only")
        self.group_type = group_type
        self.gspace = gspaces.rot2dOnR2(maximum_frequency=4)
        self.in_type = steerable.FieldType(
            self.gspace, [self.gspace.trivial_repr] * in_shape[0]
        )
        in_type = self.in_type
        layers = []
        for _ in range(num_layers):
            activation = steerable.FourierELU(
                self.gspace, out_channels, [(frequency,) for frequency in range(5)],
                N=16, inplace=True,
            )
            layers.extend((
                steerable.R2Conv(in_type, activation.in_type, kernel_size, bias=False),
                steerable.GNormBatchNorm(activation.in_type),
                activation,
            ))
            in_type = activation.out_type
        out_type = steerable.FieldType(self.gspace, [self.gspace.irrep(1)] * 2)
        layers.append(steerable.R2Conv(in_type, out_type, kernel_size, bias=False))
        self.block = steerable.SequentialModule(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from escnn.nn import GeometricTensor

        features = self.block(GeometricTensor(x, self.in_type)).tensor
        return features.mean(dim=(-1, -2)).reshape(x.shape[0], 2, 2)

