# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import inspect
import os

import torch
import torch.nn as nn
import torchvision.models as models


def get_norm(norm_type, dim):
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim)
    elif norm_type is None:
        return None
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_type="layer_norm", activation="SiLU"):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        norm = get_norm(norm_type, dim)
        if norm:
            layers.append(norm)
        layers.append(getattr(nn, activation)())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, depth, norm="layer_norm", activation="SiLU"
    ):
        super().__init__()

        # Input projection
        input_layers = [nn.Linear(input_dim, hidden_dim)]
        norm_layer = get_norm(norm, hidden_dim)
        if norm_layer:
            input_layers.append(norm_layer)
        input_layers.append(getattr(nn, activation)())
        self.input_layer = nn.Sequential(*input_layers)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, norm_type=norm, activation=activation)
                for _ in range(depth)
            ]
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


class BaseModule(nn.Module):
    def __init__(
        self,
        obs_dim_dict=None,
        module_config_dict=None,
        module_dim_dict={},
        env_config=None,
        algo_config=None,
        process_output_dim=False,
    ):
        super(BaseModule, self).__init__()

        self._batch_norm_hooks = (
            []
        )  # hold on to batch norm layers to set them to eval mode if not training backbone
        self.env_config = env_config
        self.algo_config = algo_config
        if obs_dim_dict is None:
            self.obs_dim_dict = env_config.robot.algo_obs_dim_dict
        else:
            self.obs_dim_dict = obs_dim_dict

        self.module_config_dict = module_config_dict
        if process_output_dim:
            self.module_config_dict = self._process_module_config(
                self.module_config_dict, self.env_config.robot.actions_dim
            )

        self.module_dim_dict = module_dim_dict

        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict.layer_config)

    def _process_module_config(self, module_config_dict, num_actions):
        output_dim_list = module_config_dict["output_dim"]
        if isinstance(output_dim_list, int):
            output_dim_list = [output_dim_list]

        for idx, output_dim in enumerate(output_dim_list):
            if output_dim == "robot_action_dim":
                module_config_dict["output_dim"][idx] = num_actions
        return module_config_dict

    def _calculate_input_dim(self):
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in self.module_config_dict["input_dim"]:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim += each_input
            elif each_input in self.module_dim_dict:
                input_dim += self.module_dim_dict[each_input]
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")

        self.input_dim = input_dim

    def _calculate_output_dim(self):
        output_dim = 0
        output_dim_list = self.module_config_dict["output_dim"]
        if isinstance(output_dim_list, int) or isinstance(output_dim_list, str):
            output_dim_list = [output_dim_list]

        for each_output in output_dim_list:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            elif each_output in self.module_dim_dict:
                output_dim += self.module_dim_dict[each_output]
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")

        self.output_dim = output_dim

    def _build_network_layer(self, layer_config):
        if layer_config["type"] == "MLP":
            self._build_mlp_layer(layer_config)
        elif layer_config["type"] == "CNN":
            self._build_cnn_layer(layer_config)
        elif layer_config["type"] == "GRU":
            self._build_gru_layer(layer_config)
        elif layer_config["type"] == "ResidualMLP":
            self._build_residual_mlp_layer(layer_config)
        elif layer_config["type"] == "ResNet":
            self._build_resnet_layer(layer_config)
        elif layer_config["type"] == "DINOv3":
            self._build_dinov3_layer(layer_config)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_config['type']}")

    def _build_mlp_layer(self, layer_config):
        layers = []
        hidden_dims = layer_config["hidden_dims"]
        output_dim = self.output_dim
        activation = getattr(nn, layer_config["activation"])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def _build_cnn_layer(self, layer_config):
        layers = []
        channel_dims = layer_config["channel_dims"]
        activation = getattr(nn, layer_config["activation"])()

        # Get input dimensions from env_config camera settings
        camera_config = self.env_config.simulator.config.cameras
        input_height = camera_config.camera_resolutions[0]
        input_width = camera_config.camera_resolutions[1]

        # Determine number of channels from camera types
        input_channels = 0
        for camera_type in camera_config.camera_types:
            if camera_type.get("rgb", False):
                input_channels += 3
            if camera_type.get("depth", False):
                input_channels += 1

        # If no channels found, default to 1
        if input_channels == 0:
            input_channels = 1

        vision_obs_dim = [input_width, input_height, input_channels]
        print("vision_obs_dim", vision_obs_dim)
        assert (
            vision_obs_dim[0] * vision_obs_dim[1] * vision_obs_dim[2]
            == self.obs_dim_dict["vision_obs"]
        )
        if len(vision_obs_dim) != 3:
            raise ValueError(
                f"vision_obs dimension should be (width, height, channels), got {vision_obs_dim}"
            )
        input_width, input_height, input_channels = vision_obs_dim

        # Get layer configurations
        layer_configs = layer_config.get("layers", [])
        use_batch_norm = layer_config.get("norm_config", {}).get("use_batch_norm", False)

        # Track spatial dimensions and channels
        current_height, current_width = input_height, input_width
        current_channels = input_channels
        conv_idx = 0  # Track which conv layer we're on for channel dimensions

        for layer_cfg in layer_configs:
            layer_type = layer_cfg["type"]

            if layer_type == "conv":
                # Get conv parameters
                kernel_size = layer_cfg.get("kernel_size", 3)
                stride = layer_cfg.get("stride", 1)
                padding = layer_cfg.get("padding", 1)

                # Determine output channels
                if conv_idx < len(channel_dims):
                    out_channels = channel_dims[conv_idx]
                else:
                    out_channels = self.output_dim

                # Add conv layer
                layers.append(
                    nn.Conv2d(
                        current_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )

                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(activation)

                # Update dimensions
                current_channels = out_channels
                current_height = (current_height - kernel_size + 2 * padding) // stride + 1
                current_width = (current_width - kernel_size + 2 * padding) // stride + 1
                conv_idx += 1

            elif layer_type == "pool":
                # Get pool parameters
                kernel_size = layer_cfg.get("kernel_size", 2)
                stride = layer_cfg.get("stride", 2)

                # Add pooling layer if dimensions allow
                if current_height >= kernel_size and current_width >= kernel_size:
                    layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
                    current_height = current_height // stride
                    current_width = current_width // stride

        # Add global average pooling if spatial dimensions are too small
        # if current_height * current_width > 1:
        #     # import ipdb; ipdb.set_trace()
        #     layers.append(nn.AdaptiveAvgPool2d(1))

        layers.append(nn.Flatten())

        layers.append(nn.Linear(current_channels * current_height * current_width, self.output_dim))

        self.module = nn.Sequential(*layers)

    def forward_without_hidden_state(self, input):
        return self.module(input)

    def forward_with_hidden_state(self, input, hidden_state):
        # import ipdb; ipdb.set_trace()
        output, hidden_state = self.module(input, hidden_state)
        return output, hidden_state

    def forward(self, input, hidden_state=None):
        if hidden_state is None:
            return self.forward_without_hidden_state(input)
        else:
            return self.forward_with_hidden_state(input, hidden_state)

    def _build_gru_layer(self, layer_config):
        self.module = nn.GRU(
            input_size=self.input_dim,
            hidden_size=layer_config["hidden_dim"],
            num_layers=layer_config["num_layers"],
            batch_first=True,
        )

    def _build_resnet_layer(self, layer_config):
        print("Building ResNet layer")
        resnet_type = layer_config.get("resnet_type", "resnet18")  # Default to resnet18
        pretrained = layer_config.get("pretrained", True)
        trainable = layer_config.get("trainable", True)

        if resnet_type == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        resnet = nn.SyncBatchNorm.convert_sync_batchnorm(resnet)
        resnet_features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

        if resnet_type in ["resnet18", "resnet34"]:
            resnet_feature_dim = 512
        else:  # resnet50, resnet101, resnet152
            resnet_feature_dim = 2048

        def modify_batch_norm_momentum(module):
            for name, child in module.named_children():
                if isinstance(child, nn.SyncBatchNorm):
                    print(child.momentum)
                    child.momentum = 0.001
                else:
                    modify_batch_norm_momentum(child)

        modify_batch_norm_momentum(resnet_features)

        # Freeze ResNet parameters if not trainable
        if not trainable:
            for param in resnet_features.parameters():
                param.requires_grad = False

            def register_batch_norm_hooks(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.SyncBatchNorm):
                        self._batch_norm_hooks.append(child)
                    else:
                        register_batch_norm_hooks(child)

            register_batch_norm_hooks(resnet_features)

        # Add a final linear layer to match output_dim
        layers = [
            resnet_features,
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(resnet_feature_dim, self.output_dim),
        ]

        self.module = nn.Sequential(*layers)

    def _build_residual_mlp_layer(self, layer_config):
        self.module = ResidualMLP(
            input_dim=self.input_dim,
            hidden_dim=layer_config["hidden_dim"],
            output_dim=self.output_dim,
            depth=layer_config["depth"],
            norm=layer_config.get("norm", "layer_norm"),
            activation=layer_config.get("activation", "SiLU"),
        )

    def _build_dinov3_layer(self, layer_config):
        print("Building DINOv3 layer")

        # Mapping of model types to their checkpoint files
        DINOV3_MODEL_WEIGHTS = {
            "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
            # Add more models as needed
        }
        DINOV3_WEIGHTS_DIR = "DINOv3_models"

        dinov3_type = layer_config.get("dinov3_type", "dinov3_vits16")  # Default to small model
        pretrained = layer_config.get("pretrained", True)
        trainable = layer_config.get("trainable", True)
        repo_dir = layer_config.get("repo_dir", None)  # Optional local repo directory
        weights_path = layer_config.get("weights_path", None)  # Optional weights path

        # Auto-detect weights path if not provided and model type is known
        if weights_path is None and pretrained and dinov3_type in DINOV3_MODEL_WEIGHTS:
            weights_path = os.path.join(DINOV3_WEIGHTS_DIR, DINOV3_MODEL_WEIGHTS[dinov3_type])
            if os.path.exists(weights_path):
                print(f"Auto-detected weights path: {weights_path}")
            else:
                print(f"Warning: Auto-detected weights path does not exist: {weights_path}")
                weights_path = None  # Reset to None if file doesn't exist

        # Get input dimensions from env_config camera settings
        camera_config = self.env_config.simulator.config.cameras
        input_height = camera_config.camera_resolutions[0]
        input_width = camera_config.camera_resolutions[1]

        # Determine number of channels from camera types
        input_channels = 0
        for camera_type in camera_config.camera_types:
            if camera_type.get("rgb", False):
                input_channels += 3
            if camera_type.get("depth", False):
                input_channels += 1

        # If no channels found, default to 3 (RGB)
        if input_channels == 0:
            input_channels = 3

        # Check that dimensions are multiples of 16 (required by DINOv3)
        if input_height % 16 != 0 or input_width % 16 != 0:
            raise ValueError(
                f"DINOv3 requires image dimensions to be divisible by 16. "
                f"Got height={input_height}, width={input_width}. "
                f"Please adjust camera resolution to be multiples of 16."
            )

        print(f"DINOv3 input: {input_height}x{input_width}")

        # Load DINOv3 model
        if pretrained:
            print(f"Loading pretrained DINOv3 model: {dinov3_type}")
            if repo_dir is not None:
                # Load from local repository
                print(f"Loading from local repository: {repo_dir}")
                if weights_path is not None:
                    # Load model without pretrained weights first, then load from custom path
                    print(f"Loading custom weights from: {weights_path}")
                    dinov3_model = torch.hub.load(
                        repo_dir, dinov3_type, source="local", pretrained=False
                    )
                    state_dict = torch.load(weights_path, map_location="cpu")
                    dinov3_model.load_state_dict(state_dict, strict=True)
                else:
                    # Load with pretrained=True (will download if not cached)
                    dinov3_model = torch.hub.load(
                        repo_dir, dinov3_type, source="local", pretrained=True
                    )
            else:
                # Load from GitHub (may hit rate limits)
                dinov3_model = torch.hub.load("facebookresearch/dinov3", dinov3_type)

        else:
            raise ValueError("DINOv3 currently only supports pretrained models")

        # Get feature dimension based on model type
        # Available models: dinov3_vits16, dinov3_vits16plus, dinov3_vitb16,
        #                   dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16
        if dinov3_type == "dinov3_vits16":
            dinov3_feature_dim = 384
        elif dinov3_type == "dinov3_vits16plus":
            dinov3_feature_dim = 384
        elif dinov3_type == "dinov3_vitb16":
            dinov3_feature_dim = 768
        elif dinov3_type == "dinov3_vitl16":
            dinov3_feature_dim = 1024
        elif dinov3_type == "dinov3_vith16plus":
            dinov3_feature_dim = 1280
        elif dinov3_type == "dinov3_vit7b16":
            dinov3_feature_dim = 1536
        else:
            raise ValueError(
                f"Unsupported DINOv3 type: {dinov3_type}. "
                f"Available options: dinov3_vits16, dinov3_vits16plus, dinov3_vitb16, "
                f"dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16"
            )

        # Freeze DINOv3 parameters if not trainable
        if not trainable:
            for param in dinov3_model.parameters():
                param.requires_grad = False

        # Print the number of trainable parameters in the DINOv3 model
        num_trainable_params = sum(p.numel() for p in dinov3_model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in DINOv3 model: {num_trainable_params}")

        # Create wrapper module that handles channel conversion
        class DINOv3Wrapper(nn.Module):
            def __init__(self, dinov3_model, input_channels, output_dim):
                super().__init__()
                self.dinov3_model = dinov3_model

                # If input is not 3 channels (RGB), add a conversion layer
                if input_channels != 3:
                    self.channel_converter = nn.Conv2d(input_channels, 3, kernel_size=1)
                else:
                    self.channel_converter = None

                # Add final projection layer to match output_dim
                self.projection = nn.Linear(dinov3_feature_dim, output_dim)

            def forward(self, x):
                # x shape: (batch, channels, height, width)

                # Convert channels if needed
                if self.channel_converter is not None:
                    x = self.channel_converter(x)

                # DINOv3 expects input in range [0, 1] or ImageNet normalized
                # Assuming input is already normalized appropriately

                # Get features from DINOv3 (returns dict with 'x_norm_clstoken' and 'x_norm_patchtokens')
                features = self.dinov3_model(x)

                # Use the CLS token as the global feature
                if isinstance(features, dict):
                    cls_token = features["x_norm_clstoken"]
                else:
                    # If it returns a tensor directly, use it
                    cls_token = features

                # Project to output dimension
                output = self.projection(cls_token)
                return output

        self.module = DINOv3Wrapper(
            dinov3_model=dinov3_model, input_channels=input_channels, output_dim=self.output_dim
        )

    def forward(self, input, **kwargs):
        if isinstance(input, dict):
            input_obs_key = self.module_config_dict["input_dim"][0]
            input = input[input_obs_key]
        return self.module(input)

    def train(self, mode=True):
        super().train(mode)
        for param in self._batch_norm_hooks:
            param.eval()
        return self
