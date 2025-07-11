# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import Tensor

try:
    import dgl  # noqa: F401 for docs

    warnings.warn(
        "DGL version of MeshGraphNet will soon be deprecated. "
        "Please use PyG version instead.",
        DeprecationWarning,
    )
except ImportError:
    warnings.warn(
        "Note: This only applies if you're using DGL.\n"
        "MeshGraphNet (DGL version) requires the DGL library.\n"
        "Install it with your preferred CUDA version from:\n"
        "https://www.dgl.ai/pages/start.html\n"
    )

try:
    import torch_scatter  # noqa: F401
except ImportError:
    # TODO(akamenev): warning for now to maintain temporary backwards compatibility
    # with DGL version. Replace with ImportError after DGL is removed.
    warnings.warn(
        "MeshGraphNet will soon require PyTorch Geometric and torch_scatter.\n"
        "Install it from here:\n"
        "https://github.com/rusty1s/pytorch_scatter\n"
    )

from dataclasses import dataclass
from itertools import chain
from typing import Callable, List, Tuple, Union
from warnings import warn

import physicsnemo  # noqa: F401 for docs
from physicsnemo.models.gnn_layers.mesh_edge_block import MeshEdgeBlock
from physicsnemo.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.models.gnn_layers.mesh_node_block import MeshNodeBlock
from physicsnemo.models.gnn_layers.utils import GraphType, set_checkpoint_fn
from physicsnemo.models.layers import get_activation
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module
from physicsnemo.utils.profiling import profile


@dataclass
class MetaData(ModelMetaData):
    name: str = "MeshGraphNet"
    # Optimization, no JIT as DGLGraph causes trouble
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


class MeshGraphNet(Module):
    """MeshGraphNet network architecture

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features
    input_dim_edges : int
        Number of edge features
    output_dim : int
        Number of outputs
    processor_size : int, optional
        Number of message passing blocks, by default 15
    mlp_activation_fn : Union[str, List[str]], optional
        Activation function to use, by default 'relu'
    num_layers_node_processor : int, optional
        Number of MLP layers for processing nodes in each message passing block, by default 2
    num_layers_edge_processor : int, optional
        Number of MLP layers for processing edge features in each message passing block, by default 2
    hidden_dim_processor : int, optional
        Hidden layer size for the message passing blocks, by default 128
    hidden_dim_node_encoder : int, optional
        Hidden layer size for the node feature encoder, by default 128
    num_layers_node_encoder : Union[int, None], optional
        Number of MLP layers for the node feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no node encoder
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoder, by default 128
    num_layers_edge_encoder : Union[int, None], optional
        Number of MLP layers for the edge feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no edge encoder
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128
    num_layers_node_decoder : Union[int, None], optional
        Number of MLP layers for the node feature decoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no decoder
    aggregation: str, optional
        Message aggregation type, by default "sum"
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    num_processor_checkpoint_segments: int, optional
        Number of processor segments for gradient checkpointing, by default 0 (checkpointing disabled)
    checkpoint_offloading: bool, optional
        Whether to offload the checkpointing to the CPU, by default False

    Example
    -------
    >>> model = physicsnemo.models.meshgraphnet.MeshGraphNet(
    ...         input_dim_nodes=4,
    ...         input_dim_edges=3,
    ...         output_dim=2,
    ...     )
    >>> graph = dgl.rand_graph(10, 5)
    >>> node_features = torch.randn(10, 4)
    >>> edge_features = torch.randn(5, 3)
    >>> output = model(node_features, edge_features, graph)
    >>> output.size()
    torch.Size([10, 2])

    Note
    ----
    Reference: Pfaff, Tobias, et al. "Learning mesh-based simulation with graph networks."
    arXiv preprint arXiv:2010.03409 (2020).
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        mlp_activation_fn: Union[str, List[str]] = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Union[int, None] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        norm_type="LayerNorm",
    ):
        super().__init__(meta=MetaData())

        activation_fn = get_activation(mlp_activation_fn)

        if norm_type not in ["LayerNorm", "TELayerNorm"]:
            raise ValueError("Norm type should be either 'LayerNorm' or 'TELayerNorm'")

        if not torch.cuda.is_available() and norm_type == "TELayerNorm":
            warn("TELayerNorm is not supported on CPU. Switching to LayerNorm.")
            norm_type = "LayerNorm"

        self.edge_encoder = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        self.node_encoder = MeshGraphMLP(
            input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type=norm_type,
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            checkpoint_offloading=checkpoint_offloading,
        )

    @profile
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: GraphType,
        **kwargs,
    ) -> Tensor:
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, graph)
        x = self.node_decoder(x)
        return x


class MeshGraphNetProcessor(nn.Module):
    """MeshGraphNet processor block"""

    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        activation_fn: nn.Module = nn.ReLU(),
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
    ):
        super().__init__()
        self.processor_size = processor_size
        self.num_processor_checkpoint_segments = num_processor_checkpoint_segments
        self.checkpoint_offloading = (
            checkpoint_offloading if (num_processor_checkpoint_segments > 0) else False
        )

        edge_block_invars = (
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_edge,
            activation_fn,
            norm_type,
            do_concat_trick,
            False,
        )
        node_block_invars = (
            aggregation,
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_node,
            activation_fn,
            norm_type,
            False,
        )

        edge_blocks = [
            MeshEdgeBlock(*edge_block_invars) for _ in range(self.processor_size)
        ]
        node_blocks = [
            MeshNodeBlock(*node_block_invars) for _ in range(self.processor_size)
        ]
        layers = list(chain(*zip(edge_blocks, node_blocks)))

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)
        self.set_checkpoint_segments(self.num_processor_checkpoint_segments)
        self.set_checkpoint_offload_ctx(self.checkpoint_offloading)

    def set_checkpoint_offload_ctx(self, enabled: bool):
        """
        Set the context for CPU offloading of checkpoints

        Parameters
        ----------
        checkpoint_offloading : bool
            whether to offload the checkpointing to the CPU
        """
        if enabled:
            self.checkpoint_offload_ctx = torch.autograd.graph.save_on_cpu(
                pin_memory=True
            )
        else:
            self.checkpoint_offload_ctx = nullcontext()

    def set_checkpoint_segments(self, checkpoint_segments: int):
        """
        Set the number of checkpoint segments

        Parameters
        ----------
        checkpoint_segments : int
            number of checkpoint segments

        Raises
        ------
        ValueError
            if the number of processor layers is not a multiple of the number of
            checkpoint segments
        """
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError(
                    "Processor layers must be a multiple of checkpoint_segments"
                )
            segment_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, segment_size):
                self.checkpoint_segments.append((i, i + segment_size))
            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    @profile
    def run_function(
        self, segment_start: int, segment_end: int
    ) -> Callable[[Tensor, Tensor, GraphType], Tuple[Tensor, Tensor]]:
        """Custom forward for gradient checkpointing

        Parameters
        ----------
        segment_start : int
            Layer index as start of the segment
        segment_end : int
            Layer index as end of the segment

        Returns
        -------
        Callable
            Custom forward function
        """
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(
            node_features: Tensor,
            edge_features: Tensor,
            graph: GraphType,
        ) -> Tuple[Tensor, Tensor]:
            """Custom forward function"""
            for module in segment:
                edge_features, node_features = module(
                    edge_features, node_features, graph
                )
            return edge_features, node_features

        return custom_forward

    @profile
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: GraphType,
    ) -> Tensor:
        with self.checkpoint_offload_ctx:
            for segment_start, segment_end in self.checkpoint_segments:
                edge_features, node_features = self.checkpoint_fn(
                    self.run_function(segment_start, segment_end),
                    node_features,
                    edge_features,
                    graph,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

        return node_features
