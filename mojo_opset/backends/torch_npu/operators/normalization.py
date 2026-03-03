import torch
import torch_npu

from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoRMSNorm
from mojo_opset.core import MojoNormQuant
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoResidualAddLayerNorm



class TorchNpuLayerNorm(MojoLayerNorm):
    supported_platforms_list = ["npu"]


class TorchNpuRMSNorm(MojoRMSNorm):
    supported_platforms_list = ["npu"]
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:

        return torch_npu.npu_rms_norm(hidden_state, self.weight, eps=self.variance_epsilon)

class TorchNpuResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu"] 

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_state, _, residual = torch_npu.npu_add_rms_norm(hidden_state, residual, self.weight, eps=self.variance_epsilon)
        if self.norm_pos != "pre":
            residual = hidden_state
        return hidden_state, residual