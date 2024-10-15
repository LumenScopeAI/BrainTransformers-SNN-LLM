import inspect
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_braingpt import BrainGPTConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func # type: ignore
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # type: ignore # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = ""
_CONFIG_FOR_DOC = "BrainGPTConfig"

BrainGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
    
# class EI_IFNeuron(nn.Module):
#     """
#     EI_IFNeuron神经元，具有自适应阈值和膜电位衰减。

#     动力学方程:
#     1. 膜电位更新: V(t) = V(t-1) + I(t)
#        其中，V(t) 是当前时刻的膜电位，I(t) 是输入电流

#     2. 自适应阈值: θ(t) = θ_base + t * alpha
#        其中，θ(t) 是当前时刻的阈值，θ_base 是基础阈值，
#        t 是时间步，alpha 是自适应调节权重

#     3. 脉冲生成: S(t) = {
#            1,  如果 V(t) ≥ θ_activation(t)
#           -1,  如果 V(t) ≤ -θ_inhibition(t)
#            0,  其他情况
#        }

#     4. 膜电位衰减: 
#        - 正电位: V(t) = max(0, V(t) * (1 - attenuation_rate))
#        - 负电位: V(t) = min(0, V(t) * (1 - attenuation_rate))
#     """

#     def __init__(self, base_threshold=0.5, attenuation_rate=1.0, alpha=1.0):
#         """
#         初始化IF神经元。

#         参数:
#         - base_threshold (float): 基础阈值 θ_base。默认为0.5。
#         - attenuation_rate (float): 膜电位衰减率。默认为1.0（100%衰减，即完全重置）。
#         - alpha (float): 自适应调节权重。默认为1.0。
#         """
#         super(EI_IFNeuron, self).__init__()
#         self.base_threshold = base_threshold
#         self.attenuation_rate = attenuation_rate
#         self.alpha = alpha

#         # 初始化时将 threshold 设为 None
#         self.register_buffer('threshold', None)
#         self.register_buffer('membrane_potential', None)
#         self.register_buffer('time_step', torch.tensor(0))
#         self.register_buffer('total_output', torch.tensor(0.0))

#     def forward(self, input_current):
#         """
#         前向传播，处理输入并产生输出脉冲。

#         参数:
#         - input_current (Tensor): 输入电流 I(t)

#         返回:
#         - output (Tensor): 输出脉冲 S(t)
#         """
#         self._initialize_membrane_potential(input_current)
#         self._calculate_adaptive_threshold()
#         self._update_membrane_potential(input_current)
#         output = self._generate_spike()
#         self._attenuate_membrane_potential()
#         self._update_stats(output)
#         return output

#     def _initialize_membrane_potential(self, input_current):
#         """初始化膜电位，如果尚未初始化或形状不匹配"""
#         if (self.membrane_potential is None or 
#             self.membrane_potential.shape != input_current.shape):
#             self.membrane_potential = torch.zeros_like(input_current)

#     def _calculate_adaptive_threshold(self):
#         """计算自适应阈值 θ(t) = θ_base + t * alpha，并存储为 self.threshold"""
#         self.threshold = self.base_threshold + self.time_step.float() * self.alpha

#     def _update_membrane_potential(self, input_current):
#         """更新膜电位 V(t) = V(t-1) + I(t)"""
#         self.membrane_potential += input_current

#     def _generate_spike(self):
#         """生成输出脉冲 S(t)"""
#         condition = torch.abs(self.membrane_potential) >= self.threshold
#         return torch.where(
#             condition,
#             torch.sign(self.membrane_potential),
#             torch.zeros_like(self.membrane_potential)
#         )

#     def _attenuate_membrane_potential(self):
#         """按固定百分比衰减累积的膜电位"""
#         attenuation_factor = 1 - self.attenuation_rate
#         self.membrane_potential *= attenuation_factor
#         # 衰减后膜电位裁剪
#         self.membrane_potential.clamp_(min=-self.threshold, max=self.threshold)

#     def _update_stats(self, output):
#         """更新总输出和时间步"""
#         self.total_output += output.sum().item()
#         self.time_step += 1

#     def reset(self):
#         """重置神经元状态"""
#         self.membrane_potential = None
#         self.threshold = None
#         self.time_step.zero_()
#         self.total_output.zero_()

#     def forward_multi_step(self, input_current, t=None):
#         """
#         多时间步前向传播，处理多个时间步的输入，并在方法内部比较并打印每个时间步的输出。
#         此方法返回累积的输出，以符合调用代码的期望。

#         参数:
#         - input_current (Tensor): 输入电流 I(t)，形状为 [batch_size, seq_len, neurons] 或 [batch_size, neurons]
#         - t (int, optional): 时间步数。如果 input_current 的形状为 [batch_size, seq_len, neurons]，则 t 从 seq_len 推断。

#         返回:
#         - accumulated_output (Tensor): 累积的输出脉冲 S(t)，形状为 [batch_size, neurons]
#         - None
#         """
#         # 确保 input_current 具有时间维度
#         if input_current.dim() == 2:
#             if t is None:
#                 raise ValueError("当 input_current 没有时间维度时，必须指定时间步数 't'")
#             input_current = input_current.unsqueeze(1).expand(-1, t, -1)
#         elif input_current.dim() == 3:
#             t = input_current.size(1)
#         else:
#             raise ValueError("input_current 必须是 2D 或 3D 张量")

#         batch_size, seq_len, neurons = input_current.shape
#         device = input_current.device
#         # **第一部分：完全并行的操作**

#         # 由于衰减率为 100%，V(t) = I(t)
#         V_t = input_current  # 形状：[batch_size, seq_len, neurons]

#         # 计算每个时间步的阈值
#         time_steps = self.time_step.item() + torch.arange(seq_len, device=device, dtype=torch.float32)
#         thresholds = self.base_threshold + time_steps.view(1, -1, 1) * self.alpha  # 形状：[1, seq_len, 1]

#         # 并行生成脉冲
#         spikes_parallel = torch.zeros_like(V_t)
#         spikes_parallel[V_t >= thresholds] = 1.0
#         spikes_parallel[V_t <= -thresholds] = -1.0

#         # 计算累积输出
#         accumulated_output = spikes_parallel.sum(dim=1)
#         return accumulated_output, spikes_parallel  # 返回累积的输出，以符合调用代码的期望

#     def forward_multi_step_(self, input_current, t=None):
#         '''
#         ANN2SNN的无损转化标准函数实现如下所述,由于无损转化中泄露率为1.0,因此可以优化成可并行的
#         '''
#         if input_current.dim() == 2:
#             if t is None:
#                 raise ValueError("当 input_current 没有时间维度时，必须指定时间步数 't'")
#             input_current = input_current.unsqueeze(1).expand(-1, t, -1)
#         elif input_current.dim() == 3:
#             t = input_current.size(1)
#         else:
#             raise ValueError("input_current 必须是 2D 或 3D 张量")

#         batch_size, seq_len, num_elements = input_current.shape
#         device = input_current.device

#         # 初始化膜电位
#         V_t = torch.zeros(batch_size, num_elements, device=device)

#         # 初始化累积输出
#         accumulated_output = torch.zeros(batch_size, num_elements, device=device)

#         for i in range(seq_len):
#             # 计算自适应阈值
#             current_time_step = self.time_step.float() + i
#             threshold = self.base_threshold + current_time_step * self.alpha

#             # 更新膜电位，包含衰减
#             V_t = V_t * (1 - self.attenuation_rate) + input_current[:, i, :]

#             # 生成脉冲
#             output_spikes = torch.zeros_like(V_t)
#             output_spikes[V_t >= threshold] = 1.0
#             output_spikes[V_t <= -threshold] = -1.0

#             # 累积输出
#             accumulated_output += output_spikes

#             # 膜电位裁剪
#             V_t = torch.clamp(V_t, min=-threshold, max=threshold)

#         # 更新时间步
#         self.time_step += seq_len

#         return accumulated_output, None

# class Synapsis(nn.Module):
#     def __init__(self, pre_ifneuron, layer, post_ifneuron, bits=8):
#         super().__init__()
#         self.pre_ifneuron = pre_ifneuron
#         self.layer = layer
#         self.post_ifneuron = post_ifneuron
#         self.bits = bits
#         self.max_value = 2**(bits - 1) - 1
#         self.if_STDP_Inspire=False
    
#     def _get_time_steps(self):
#         return self.max_value
    
#     def _compute_scaling_factor(self, x):
#         max_abs = x.abs().max()
#         return max_abs / self.max_value if max_abs != 0 else 1.0
    
#     def _forward_ifneuron(self, x, ifneuron, layer=None):
#         time_steps = self._get_time_steps()
#         scaling_factor = self._compute_scaling_factor(x)
        
#         x_scaled = x / scaling_factor

#         # 将输入展平成二维张量 [batch_size, num_elements]
#         x_flat = x_scaled.view(x_scaled.size(0), -1)

#         # 使用 forward_multi_step 方法
#         accumulated_output, _ = ifneuron.forward_multi_step(x_flat, time_steps)
        
#         # 将 accumulated_output reshaped 回输入 x 的形状
#         accumulated_output = accumulated_output.view_as(x)
        
#         # 返回调整后的输出
#         return (accumulated_output * scaling_factor).to(x.dtype)
    
#     def _forward_ifneuron_single_steps(self, x, ifneuron):
#         time_steps = self._get_time_steps(x)
#         scaling_factor = self._compute_scaling_factor(x)
        
#         x_scaled = x / scaling_factor
#         if time_steps == 0:
#             return x  # 直接返回输入，不进行任何处理

#         outputs = []
#         for _ in range(time_steps):
#             output = ifneuron(x_scaled)
#             outputs.append(output)
#         ifneuron.reset()
#         return torch.stack(outputs* scaling_factor).sum(dim=0)
    
#     def forward(self, x):
#         if not self.if_STDP_Inspire:
#             if isinstance(self.layer, nn.Embedding):
#                 x = self.layer(x)
#                 x = self._forward_ifneuron(x, self.post_ifneuron)
#                 return x
#             else:
#                 x = self._forward_ifneuron(x, self.pre_ifneuron)
#                 x = self.layer(x)
#                 x = self._forward_ifneuron(x, self.post_ifneuron)
#                 return x
#         else:
#             if isinstance(self.layer, nn.Embedding):
#                 x = self.layer(x)
#                 x = self._forward_ifneuron_single_steps(x, self.post_ifneuron)
#                 return x
#             else:
#                 x = self._forward_ifneuron_single_steps(x, self.pre_ifneuron)
#                 x = self.layer(x)
#                 x = self._forward_ifneuron_single_steps(x, self.post_ifneuron)
#                 return x            
            
#     def __getattr__(self, name):
#         if name in ['weight', 'bias']:
#             return getattr(self.layer, name)
#         return super().__getattr__(name)
    
#     def reset_parameters(self):
#         if hasattr(self.layer, 'reset_parameters'):
#             self.layer.reset_parameters()
#         for ifneuron in [self.pre_ifneuron, self.post_ifneuron]:
#             if hasattr(ifneuron, 'reset_parameters'):
#                 ifneuron.reset_parameters()
    
#     def train(self, mode=True):
#         super().train(mode)
#         self.pre_ifneuron.train(mode)
#         self.layer.train(mode)
#         self.post_ifneuron.train(mode)
#         return self
    
#     def eval(self):
#         super().eval()
#         self.pre_ifneuron.eval()
#         self.layer.eval()
#         self.post_ifneuron.eval()
#         return self
    
#     def parameters(self):
#         return (
#             list(self.pre_ifneuron.parameters()) +
#             list(self.layer.parameters()) +
#             list(self.post_ifneuron.parameters()) 
#         )
class EI_IFNeuron(nn.Module):
    """
    EI_IFNeuron神经元，具有自适应阈值和膜电位衰减。

    动力学方程:
    1. 膜电位更新: V(t) = V(t-1) + I(t)
       其中，V(t) 是当前时刻的膜电位，I(t) 是输入电流

    2. 自适应阈值: θ(t) = θ_base + t * alpha
       其中，θ(t) 是当前时刻的阈值，θ_base 是基础阈值，
       t 是时间步，alpha 是自适应调节权重

    3. 脉冲生成: S(t) = {
           1,  如果 V(t) ≥ θ_activation(t)
          -1,  如果 V(t) ≤ -θ_inhibition(t)
           0,  其他情况
       }

    4. 膜电位衰减: 
       - 正电位: V(t) = max(0, V(t) * (1 - attenuation_rate))
       - 负电位: V(t) = min(0, V(t) * (1 - attenuation_rate))
    """

    def __init__(self, base_threshold=0.5, attenuation_rate=1.0, alpha=1.0):
        """
        初始化IF神经元。

        参数:
        - base_threshold (float): 基础阈值 θ_base。默认为0.5。
        - attenuation_rate (float): 膜电位衰减率。默认为1.0（100%衰减，即完全重置）。
        - alpha (float): 自适应调节权重。默认为1.0。
        """
        super(EI_IFNeuron, self).__init__()
        self.base_threshold = base_threshold
        self.attenuation_rate = attenuation_rate
        self.alpha = alpha

        # 初始化时将 threshold 设为 None
        self.register_buffer('threshold', None)
        self.register_buffer('membrane_potential', None)
        self.register_buffer('time_step', torch.tensor(0))
        self.register_buffer('total_output', torch.tensor(0.0))

        # STDP相关参数
        self.register_buffer('last_spike_time', torch.tensor(float('-inf')))
        self.register_buffer('spike_trace', torch.tensor(0.0))
        self.tau_trace = 20.0  # 尖峰跟踪的时间常数

    def forward(self, input_current):
        """
        前向传播，处理输入并产生输出脉冲。

        参数:
        - input_current (Tensor): 输入电流 I(t)

        返回:
        - output (Tensor): 输出脉冲 S(t)
        """
        self._initialize_membrane_potential(input_current)
        self._calculate_adaptive_threshold()
        self._update_membrane_potential(input_current)
        output = self._generate_spike()
        self._attenuate_membrane_potential()
        self._update_stats(output)
        self._update_spike_trace(output)
        return output

    def _initialize_membrane_potential(self, input_current):
        """初始化膜电位，如果尚未初始化或形状不匹配"""
        if (self.membrane_potential is None or 
            self.membrane_potential.shape != input_current.shape):
            self.membrane_potential = torch.zeros_like(input_current)

    def _calculate_adaptive_threshold(self):
        """计算自适应阈值 θ(t) = θ_base + t * alpha，并存储为 self.threshold"""
        self.threshold = self.base_threshold + self.time_step.float() * self.alpha

    def _update_membrane_potential(self, input_current):
        """更新膜电位 V(t) = V(t-1) + I(t)"""
        self.membrane_potential += input_current

    def _generate_spike(self):
        """生成输出脉冲 S(t)"""
        condition = torch.abs(self.membrane_potential) >= self.threshold
        return torch.where(
            condition,
            torch.sign(self.membrane_potential),
            torch.zeros_like(self.membrane_potential)
        )

    def _attenuate_membrane_potential(self):
        """按固定百分比衰减累积的膜电位"""
        attenuation_factor = 1 - self.attenuation_rate
        self.membrane_potential *= attenuation_factor
        # 衰减后膜电位裁剪
        self.membrane_potential.clamp_(min=-self.threshold, max=self.threshold)

    def _update_stats(self, output):
        """更新总输出和时间步"""
        self.total_output += output.sum().item()
        self.time_step += 1

    def _update_spike_trace(self, output):
        """更新尖峰跟踪"""
        self.spike_trace *= torch.exp(-1 / self.tau_trace)
        self.spike_trace += output
        if output.sum() > 0:
            self.last_spike_time = self.time_step.float()

    def reset(self):
        """重置神经元状态"""
        self.membrane_potential = None
        self.threshold = None
        self.time_step.zero_()
        self.total_output.zero_()
        self.last_spike_time.fill_(float('-inf'))
        self.spike_trace.zero_()

    def forward_multi_step(self, input_current, t=None):
        """
        多时间步前向传播，处理多个时间步的输入，并在方法内部比较并打印每个时间步的输出。
        此方法返回累积的输出，以符合调用代码的期望。

        参数:
        - input_current (Tensor): 输入电流 I(t)，形状为 [batch_size, seq_len, neurons] 或 [batch_size, neurons]
        - t (int, optional): 时间步数。如果 input_current 的形状为 [batch_size, seq_len, neurons]，则 t 从 seq_len 推断。

        返回:
        - accumulated_output (Tensor): 累积的输出脉冲 S(t)，形状为 [batch_size, neurons]
        - None
        """
        # 确保 input_current 具有时间维度
        if input_current.dim() == 2:
            if t is None:
                raise ValueError("当 input_current 没有时间维度时，必须指定时间步数 't'")
            input_current = input_current.unsqueeze(1).expand(-1, t, -1)
        elif input_current.dim() == 3:
            t = input_current.size(1)
        else:
            raise ValueError("input_current 必须是 2D 或 3D 张量")

        batch_size, seq_len, neurons = input_current.shape
        device = input_current.device
        # **第一部分：完全并行的操作**

        # 由于衰减率为 100%，V(t) = I(t)
        V_t = input_current  # 形状：[batch_size, seq_len, neurons]

        # 计算每个时间步的阈值
        time_steps = self.time_step.item() + torch.arange(seq_len, device=device, dtype=torch.float32)
        thresholds = self.base_threshold + time_steps.view(1, -1, 1) * self.alpha  # 形状：[1, seq_len, 1]

        # 并行生成脉冲
        spikes_parallel = torch.zeros_like(V_t)
        spikes_parallel[V_t >= thresholds] = 1.0
        spikes_parallel[V_t <= -thresholds] = -1.0

        # 计算累积输出
        accumulated_output = spikes_parallel.sum(dim=1)
        return accumulated_output, spikes_parallel  # 返回累积的输出，以符合调用代码的期望

    def forward_multi_step_(self, input_current, t=None):
        '''
        ANN2SNN的无损转化标准函数实现如下所述,由于无损转化中泄露率为1.0,因此可以优化成可并行的
        '''
        if input_current.dim() == 2:
            if t is None:
                raise ValueError("当 input_current 没有时间维度时，必须指定时间步数 't'")
            input_current = input_current.unsqueeze(1).expand(-1, t, -1)
        elif input_current.dim() == 3:
            t = input_current.size(1)
        else:
            raise ValueError("input_current 必须是 2D 或 3D 张量")

        batch_size, seq_len, num_elements = input_current.shape
        device = input_current.device

        # 初始化膜电位
        V_t = torch.zeros(batch_size, num_elements, device=device)

        # 初始化累积输出
        accumulated_output = torch.zeros(batch_size, num_elements, device=device)

        for i in range(seq_len):
            # 计算自适应阈值
            current_time_step = self.time_step.float() + i
            threshold = self.base_threshold + current_time_step * self.alpha

            # 更新膜电位，包含衰减
            V_t = V_t * (1 - self.attenuation_rate) + input_current[:, i, :]

            # 生成脉冲
            output_spikes = torch.zeros_like(V_t)
            output_spikes[V_t >= threshold] = 1.0
            output_spikes[V_t <= -threshold] = -1.0

            # 累积输出
            accumulated_output += output_spikes

            # 膜电位裁剪
            V_t = torch.clamp(V_t, min=-threshold, max=threshold)

        # 更新时间步
        self.time_step += seq_len

        return accumulated_output, None

class Synapsis(nn.Module):
    def __init__(self, pre_ifneuron, layer, post_ifneuron, bits=8):
        super().__init__()
        self.pre_ifneuron = pre_ifneuron
        self.layer = layer
        self.post_ifneuron = post_ifneuron
        self.bits = bits
        self.max_value = 2**(bits - 1) - 1
        self.if_STDP_Inspire=False
    
        # STDP 相关参数
        self.learning_rate = 0.001
        self.A_plus = 0.1
        self.A_minus = 0.12
        self.tau_plus = 20
        self.tau_minus = 20
        self.tag = torch.zeros_like(self.layer.weight)
        self.current_task_loss = None

    def _get_time_steps(self):
        return self.max_value
    
    def _compute_scaling_factor(self, x):
        max_abs = x.abs().max()
        return max_abs / self.max_value if max_abs != 0 else 1.0
    
    def _forward_ifneuron(self, x, ifneuron, layer=None):
        time_steps = self._get_time_steps()
        scaling_factor = self._compute_scaling_factor(x)
        
        x_scaled = x / scaling_factor

        # 将输入展平成二维张量 [batch_size, num_elements]
        x_flat = x_scaled.view(x_scaled.size(0), -1)

        # 使用 forward_multi_step 方法
        accumulated_output, _ = ifneuron.forward_multi_step(x_flat, time_steps)
        
        # 将 accumulated_output reshaped 回输入 x 的形状
        accumulated_output = accumulated_output.view_as(x)
        
        # 返回调整后的输出
        return (accumulated_output * scaling_factor).to(x.dtype)
    
    def _forward_ifneuron_single_steps(self, x, ifneuron):
        time_steps = self._get_time_steps()
        scaling_factor = self._compute_scaling_factor(x)
        
        x_scaled = x / scaling_factor
        if time_steps == 0:
            return x  # 直接返回输入，不进行任何处理

        outputs = []
        for _ in range(time_steps):
            output = ifneuron(x_scaled)
            outputs.append(output)
            if self.if_STDP_Inspire and self.training:
                self.apply_stdp(x_scaled, output)
        ifneuron.reset()
        return torch.stack(outputs * scaling_factor).sum(dim=0)
    
    def forward(self, x):
        if not self.if_STDP_Inspire:
            if isinstance(self.layer, nn.Embedding):
                x = self.layer(x)
                x = self._forward_ifneuron(x, self.post_ifneuron)
                return x
            else:
                x = self._forward_ifneuron(x, self.pre_ifneuron)
                x = self.layer(x)
                x = self._forward_ifneuron(x, self.post_ifneuron)
                return x
        else:
            if isinstance(self.layer, nn.Embedding):
                x = self.layer(x)
                x = self._forward_ifneuron_single_steps(x, self.post_ifneuron)
                return x
            else:
                x = self._forward_ifneuron_single_steps(x, self.pre_ifneuron)
                x = self.layer(x)
                x = self._forward_ifneuron_single_steps(x, self.post_ifneuron)
                return x            
    
    def apply_stdp(self, pre_activity, post_activity):
        if not isinstance(self.layer, (nn.Linear, nn.Embedding)):
            return

        self.tag = torch.sigmoid(pre_activity.abs().mean() + post_activity.abs().mean() - self.current_task_loss)
        
        delta_t = self.post_ifneuron.last_spike_time - self.pre_ifneuron.last_spike_time
        if delta_t > 0:
            delta_w = self.A_plus * torch.exp(-delta_t / self.tau_plus)
        else:
            delta_w = -self.A_minus * torch.exp(delta_t / self.tau_minus)
        
        weight_update = self.learning_rate * self.tag * (delta_w - self.layer.weight)
        self.layer.weight.data += weight_update

    def __getattr__(self, name):
        if name in ['weight', 'bias']:
            return getattr(self.layer, name)
        return super().__getattr__(name)
    
    def reset_parameters(self):
        if hasattr(self.layer, 'reset_parameters'):
            self.layer.reset_parameters()
        for ifneuron in [self.pre_ifneuron, self.post_ifneuron]:
            if hasattr(ifneuron, 'reset_parameters'):
                ifneuron.reset_parameters()
    
    def train(self, mode=True):
        super().train(mode)
        self.pre_ifneuron.train(mode)
        self.layer.train(mode)
        self.post_ifneuron.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.pre_ifneuron.eval()
        self.layer.eval()
        self.post_ifneuron.eval()
        return self
    
    def parameters(self):
        return (
            list(self.pre_ifneuron.parameters()) +
            list(self.layer.parameters()) +
            list(self.post_ifneuron.parameters()) 
        )
        
class SNNMatmul(nn.Module):
    def __init__(self, neuron_q, neuron_k, bits=8):
        super(SNNMatmul, self).__init__()
        self.neuron_q = neuron_q
        self.neuron_k = neuron_k
        self.bits = bits
        self.max_value = 2**(bits - 1) - 1

    def _get_time_steps(self):
        return self.max_value

    def _compute_scaling_factor(self, x):
        max_abs = x.abs().max()
        return max_abs / self.max_value if max_abs != 0 else 1.0

    def forward(self, Q, K, time_steps=None):
        """
        参数：
            Q: 张量，形状为 [batch_size, num_heads, seq_len_q, head_dim]
            K: 张量，形状为 [batch_size, num_heads, seq_len_k, head_dim]
            time_steps: 神经元的时间步数
        返回：
            A_T: 张量，形状为 [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        if Q.dim() != 4 or K.dim() != 4:
            raise ValueError("Q 和 K 应该是 4D 张量")

        batch_size, num_heads, seq_len_q, head_dim_q = Q.shape
        _, _, seq_len_k, head_dim_k = K.shape
        if head_dim_q != head_dim_k:
            raise ValueError("Q 和 K 的 head_dim 必须匹配")

        device = Q.device

        if time_steps is None:
            time_steps = self._get_time_steps()

        # 计算缩放因子
        scaling_factor_q = self._compute_scaling_factor(Q)
        scaling_factor_k = self._compute_scaling_factor(K)

        # 缩放输入
        Q_scaled = Q / scaling_factor_q
        K_scaled = K / scaling_factor_k

        # 初始化变量
        S_Q_t_positive = torch.zeros(batch_size, num_heads, seq_len_q, head_dim_q, device=device)
        S_Q_t_negative = torch.zeros_like(S_Q_t_positive)
        S_K_t_positive = torch.zeros(batch_size, num_heads, seq_len_k, head_dim_k, device=device)
        S_K_t_negative = torch.zeros_like(S_K_t_positive)
        A_T = torch.zeros(batch_size, num_heads, seq_len_q, seq_len_k, device=device)

        # 重置神经元
        self.neuron_q.reset()
        self.neuron_k.reset()

        for t in range(time_steps):
            # 获取神经元的脉冲输出
            Q_s_t = self.neuron_q(Q_scaled)  # [batch_size, num_heads, seq_len_q, head_dim]
            K_s_t = self.neuron_k(K_scaled)  # [batch_size, num_heads, seq_len_k, head_dim]

            # 在脉冲生成后立即恢复缩放
            Q_s_t = Q_s_t * scaling_factor_q
            K_s_t = K_s_t * scaling_factor_k

            # 分离正负脉冲
            Q_s_t_positive = torch.clamp(Q_s_t, min=0)
            Q_s_t_negative = torch.clamp(Q_s_t, max=0)
            K_s_t_positive = torch.clamp(K_s_t, min=0)
            K_s_t_negative = torch.clamp(K_s_t, max=0)

            # 更新累积和
            S_Q_t_positive += Q_s_t_positive
            S_Q_t_negative += Q_s_t_negative
            S_K_t_positive += K_s_t_positive
            S_K_t_negative += K_s_t_negative

            # 计算各项
            # term1 = S_Q_t * K_s_t^T
            term1 = torch.matmul(S_Q_t_positive, K_s_t_positive.transpose(-1, -2)) + \
                    torch.matmul(S_Q_t_negative, K_s_t_negative.transpose(-1, -2)) + \
                    torch.matmul(S_Q_t_positive, K_s_t_negative.transpose(-1, -2)) + \
                    torch.matmul(S_Q_t_negative, K_s_t_positive.transpose(-1, -2))

            # term2 = Q_s_t * S_K_t^T
            term2 = torch.matmul(Q_s_t_positive, S_K_t_positive.transpose(-1, -2)) + \
                    torch.matmul(Q_s_t_negative, S_K_t_negative.transpose(-1, -2)) + \
                    torch.matmul(Q_s_t_positive, S_K_t_negative.transpose(-1, -2)) + \
                    torch.matmul(Q_s_t_negative, S_K_t_positive.transpose(-1, -2))

            # term3 = Q_s_t * K_s_t^T
            term3 = torch.matmul(Q_s_t_positive, K_s_t_positive.transpose(-1, -2)) + \
                    torch.matmul(Q_s_t_negative, K_s_t_negative.transpose(-1, -2)) + \
                    torch.matmul(Q_s_t_positive, K_s_t_negative.transpose(-1, -2)) + \
                    torch.matmul(Q_s_t_negative, K_s_t_positive.transpose(-1, -2))

            # 累加结果
            A_T += term1 + term2 - term3

        return A_T

class SNNSoftmax(nn.Module):
    def __init__(self, neuron=None, bits=8, dim=-1, dtype=None):
        super(SNNSoftmax, self).__init__()
        self.time_steps = None
        self.bits = bits
        self.max_value = 2 ** (bits - 1) - 1
        self.neuron = neuron if neuron is not None else EI_IFNeuron()
        self.dim = dim  # 添加 dim 参数
        self.dtype = dtype  # 添加 dtype 参数
    
    def _get_time_steps(self):
        return self.max_value

    def _compute_scaling_factor(self, x):
        max_abs = x.abs().max()
        return max_abs / self.max_value if max_abs != 0 else 1.0

    def forward(self, inputs):
        if self.time_steps is None:
            self.time_steps = self._get_time_steps()
        batch_size = inputs.size(0)

        # 计算缩放因子，防止输入过大导致神经元过载
        scaling_factor = self._compute_scaling_factor(inputs)
        inputs_scaled = inputs / scaling_factor

        # 重置神经元状态
        self.neuron.reset()

        # 初始化累积输入和累积输出
        accumulated_input = torch.zeros_like(inputs)
        accumulated_output = torch.zeros_like(inputs)
        previous_output = torch.zeros_like(inputs)

        # 在每个时间步处理输入
        for t in range(self.time_steps):
            # 神经元前向传播，获取脉冲输出
            input_spikes = self.neuron(inputs_scaled)
            # 恢复缩放
            input_spikes = input_spikes * scaling_factor

            # 累积输入
            accumulated_input += input_spikes

            # 在累积输入上计算 Softmax，使用添加的 dim 和 dtype 参数
            softmax_output = F.softmax(accumulated_input, dim=self.dim, dtype=self.dtype)

            # 计算当前时间步的输出脉冲
            output_spikes = softmax_output - previous_output
            previous_output = softmax_output

            # 累积输出
            accumulated_output += output_spikes

        return accumulated_output

class SNNDropout(nn.Module):
    def __init__(self, neuron=None, bits=8):
        super(SNNDropout, self).__init__()
        self.time_steps = None
        self.bits = bits
        self.max_value = 2 ** (bits - 1) - 1
        self.neuron = neuron if neuron is not None else EI_IFNeuron()
    
    def _get_time_steps(self):
        return self.max_value

    def _compute_scaling_factor(self, x):
        max_abs = x.abs().max()
        return max_abs / self.max_value if max_abs != 0 else 1.0

    def forward(self, inputs,p,if_train):
        if self.time_steps is None:
            self.time_steps = self._get_time_steps()
        batch_size = inputs.size(0)

        # 计算缩放因子，防止输入过大导致神经元过载
        scaling_factor = self._compute_scaling_factor(inputs)
        inputs_scaled = inputs / scaling_factor

        # 重置神经元状态
        self.neuron.reset()

        # 初始化累积输入和累积输出
        accumulated_input = torch.zeros_like(inputs)
        accumulated_output = torch.zeros_like(inputs)
        previous_output = torch.zeros_like(inputs)

        # 在每个时间步处理输入
        for t in range(self.time_steps):
            # 神经元前向传播，获取脉冲输出
            input_spikes = self.neuron(inputs_scaled)
            # 恢复缩放
            output_spikes = input_spikes * scaling_factor
            # 应用dropout
            if if_train:
                mask = torch.bernoulli(torch.full_like(output_spikes, 1 - p))
                output_spikes = output_spikes * mask / (1 - p)
            # 累积输出
            accumulated_output += output_spikes

        return accumulated_output
'''
事实上，SNNRMSNorm可以使用以下纯SNN组件替换，鉴于现阶段算子仍未适配，因此暂时使用SNNRMSNorm的ANN混合版本。
同理Synapsis可以使用加法代替乘法，由于每一时间步神经元的输出只包含[-1,0,1]因此直接对layer.weight进行相应的加法操作，
再做时间步的累计即可实现。

    def _spike_layer(self, x, layer):
        if isinstance(layer, nn.Linear):
            return self._spike_linear(x, layer)
        elif isinstance(layer, nn.Conv1d):
            return self._spike_conv1d(x, layer)
        elif isinstance(layer, nn.Conv2d):
            return self._spike_conv2d(x, layer)
        else:
            raise ValueError(f"不支持的层类型: {type(layer)}")

    def _spike_linear(self, x, layer):
        w = layer.weight.to(x.device)
        b = layer.bias.to(x.device) if layer.bias is not None else None

        if x.dim() == 1:
            x = x.unsqueeze(0)  # 如果缺少批次维度,则添加

        x_pos = (x == 1).float()
        x_neg = (x == -1).float()

        out = torch.matmul(x_pos - x_neg, w.t())

        if b is not None:
            out += b

        return out.to(x.dtype)

    def _spike_conv1d(self, x, layer):
        w = layer.weight.to(x.device)
        b = layer.bias.to(x.device) if layer.bias is not None else None
        stride = layer.stride[0]
        padding = layer.padding[0]
        dilation = layer.dilation[0]

        if x.dim() == 2:
            x = x.unsqueeze(0)  # 如果缺少批次维度,则添加

        batch_size, in_channels, in_width = x.shape
        out_channels, _, kernel_size = w.shape

        # 添加padding
        x_padded = torch.nn.functional.pad(x, (padding, padding))

        # 创建滑动窗口视图
        x_windows = x_padded.unfold(2, kernel_size * dilation, stride)

        # 调整权重形状以匹配输入窗口
        w_adjusted = w.view(out_channels, -1, 1)

        # 计算卷积
        x_pos = (x_windows == 1).float()
        x_neg = (x_windows == -1).float()
        out = torch.matmul(x_pos - x_neg, w_adjusted).squeeze(-1)

        if b is not None:
            out += b.view(1, -1, 1)

        return out.to(x.dtype)

    def _spike_conv2d(self, x, layer):
        w = layer.weight.to(x.device)
        b = layer.bias.to(x.device) if layer.bias is not None else None
        stride = layer.stride
        padding = layer.padding
        dilation = layer.dilation

        if x.dim() == 3:
            x = x.unsqueeze(0)  # 如果缺少批次维度,则添加

        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, _, kernel_height, kernel_width = w.shape

        # 添加padding
        x_padded = torch.nn.functional.pad(x, (padding[1], padding[1], padding[0], padding[0]))

        # 创建滑动窗口视图
        x_windows = x_padded.unfold(2, kernel_height * dilation[0], stride[0]).unfold(3, kernel_width * dilation[1], stride[1])

        # 调整权重形状以匹配输入窗口
        w_adjusted = w.view(out_channels, -1)

        # 计算卷积
        x_pos = (x_windows == 1).float()
        x_neg = (x_windows == -1).float()
        out = torch.matmul((x_pos - x_neg).reshape(batch_size, in_channels, -1, x_windows.shape[-2] * x_windows.shape[-1]).permute(0, 2, 1, 3), w_adjusted.t()).permute(0, 2, 1)

        if b is not None:
            out += b.view(1, -1, 1, 1)

        return out.to(x.dtype)

'''

# # 定义 CustomNeuron 类
# class CustomNeuron(nn.Module):
#     def __init__(self,
#                  base_threshold=1.0,
#                  decay_rate=0.99,
#                  time_steps=100):
#         super(CustomNeuron, self).__init__()
#         self.base_threshold = base_threshold
#         self.decay_rate = decay_rate
#         self.time_steps = int(time_steps)

#         # 可训练参数：阈值和脉冲幅度
#         self.threshold_params = nn.Parameter(torch.zeros(self.time_steps))
#         self.pulse_params = nn.Parameter(torch.ones(self.time_steps) * 0.1)

#     def forward(self, x):
#         device = x.device
#         dtype = x.dtype
#         batch_size = x.shape[0]

#         V = torch.zeros(batch_size, device=device, dtype=dtype)
#         output = torch.zeros(batch_size, device=device, dtype=dtype)

#         for t in range(self.time_steps):
#             # 获取当前阈值和脉冲幅度
#             threshold = self.base_threshold + self.threshold_params[t]
#             pulse = self.pulse_params[t]

#             # 更新膜电位
#             V = V * self.decay_rate + x.squeeze()

#             # 初始化脉冲
#             spike = torch.zeros_like(V, device=device, dtype=dtype)

#             # 脉冲发放
#             spike_mask = (V >= threshold)
#             spike += spike_mask.float() * pulse

#             # 将超过阈值的膜电位重置
#             V[spike_mask] = 0.0

#             # 累计输出脉冲
#             output += spike

#         return output.unsqueeze(1)

# # 定义用于异常值区间的 CustomNeuronOutlier 类
# class CustomNeuronOutlier(nn.Module):
#     def __init__(self, base_threshold=0.0, decay_rate=1.0, time_steps=20):
#         super(CustomNeuronOutlier, self).__init__()
#         self.base_threshold = base_threshold  # 基础阈值
#         self.decay_rate = decay_rate          # 衰减率
#         self.time_steps = int(time_steps)     # 时间步数

#         # 可训练参数：阈值和脉冲幅度
#         self.threshold_params = nn.Parameter(
#             torch.zeros(self.time_steps, dtype=torch.float32), requires_grad=True)
#         self.pulse_params = nn.Parameter(
#             torch.zeros(self.time_steps, dtype=torch.float32), requires_grad=True)

#     def forward(self, x):
#         device = x.device
#         dtype = x.dtype
#         batch_size = x.shape[0]

#         output = torch.zeros(batch_size, device=device, dtype=dtype)

#         for t in range(self.time_steps):
#             # 获取当前时间步的阈值和脉冲幅度
#             threshold = self.base_threshold + self.threshold_params[t]
#             pulse = self.pulse_params[t]

#             # 膜电位
#             V = x.squeeze()

#             # 生成脉冲
#             spike_mask = (V >= threshold)
#             spike = spike_mask.float() * pulse

#             # 累计输出
#             output += spike

#         return output.unsqueeze(1)

# # 定义 SquareApproximator 类
# class SquareApproximator(nn.Module):
#     def __init__(self, time_steps=256, N=64, p=1.2):
#         super(SquareApproximator, self).__init__()
#         self.time_steps = int(time_steps)
#         self.N = N       # 区间数量
#         self.p = p       # 控制密集程度的参数，p > 1

#         # 为每个区间创建一个神经元
#         self.neurons = nn.ModuleList([
#             CustomNeuron(
#                 base_threshold=1.0,
#                 decay_rate=0.99,
#                 time_steps=int(self.time_steps / self.N)
#             ) for _ in range(self.N)
#         ])

#         # 使用非均匀划分定义区间边界
#         b = 4  # 定义域的最大值
#         i_values = torch.arange(self.N + 1, dtype=torch.float32)
#         s_i = (i_values / self.N)
#         self.x_edges = b * s_i ** self.p  # 基于 s_i^p 的非均匀划分

#         # 提取区间的起始和结束点
#         x_edges_list = self.x_edges.tolist()
#         self.x_starts = x_edges_list[:-1]
#         self.x_ends = x_edges_list[1:]

#     def forward(self, x):
#         original_shape = x.shape
#         device = x.device
#         dtype = x.dtype
        
#         # 保持原有的squeeze操作
#         x_squeezed = x.squeeze()
#         x_abs = x_squeezed.abs() + 0.5  # 保持原有的加0.5操作
#         output = torch.zeros_like(x_abs, device=device, dtype=dtype)

#         # 根据输入值，通过适当的神经元处理
#         for i in range(self.N):
#             x_start = self.x_starts[i]
#             x_end = self.x_ends[i]
#             mask = (x_abs >= x_start) & (x_abs <= x_end) if i == 0 else (x_abs > x_start) & (x_abs <= x_end)

#             if mask.any():
#                 x_segment = x_abs[mask]
#                 neuron = self.neurons[i]
#                 out = neuron(x_segment.unsqueeze(1))
#                 output[mask] = out.squeeze()

#         # 如果原始输入是1D或2D，则在最后添加一个维度
#         if len(original_shape) <= 2:
#             output = output.unsqueeze(-1)
#         else:
#             # 如果是高维输入，恢复到原始形状
#             output = output.view(original_shape)

#         return output

# # 定义用于异常值区间的 SquareApproximatorOutlier 类
# class SquareApproximatorOutlier(nn.Module):
#     def __init__(self, shift_unit=8, time_steps=20, N=None, p=None):
#         super(SquareApproximatorOutlier, self).__init__()
#         self.shift_unit = shift_unit
#         self.time_steps = time_steps
#         self.N = N  # 未使用，但为了一致性包括在内
#         self.p = p  # 未使用，但为了一致性包括在内

#         # 创建专用于异常值区间的神经元
#         self.neuron = CustomNeuronOutlier(
#             base_threshold=0.0,
#             decay_rate=1.0,
#             time_steps=time_steps
#         )

#     def forward(self, x):
#         return self.neuron(x.abs())

# # 定义集成模型 SNNSqrtNeuro 类
# class SNNSqareNeuro(nn.Module):
#     def __init__(self, models, device):
#         super(SNNSqareNeuro, self).__init__()
#         self.device = device

#         self.shift_units = [0, 2, 4, 6, 7, 'outlier']
#         self.models = nn.ModuleDict(models)
#         self.thresholds = torch.tensor([0, 2, 4, 6, 7, 8], device=device, dtype=torch.float32)

#     def forward(self, x):
#         device = x.device
#         dtype = x.dtype
#         x_input = x.squeeze()
#         abs_x = torch.abs(x_input)

#         output = torch.zeros_like(x_input, device=device, dtype=dtype)

#         for i in range(len(self.thresholds) - 1):
#             shift_unit = self.shift_units[i]
#             model = self.models[str(shift_unit)]

#             # 神经元激活条件
#             mask = (abs_x >= self.thresholds[i]) & (abs_x < self.thresholds[i + 1])

#             if mask.any():
#                 x_segment = abs_x[mask]
#                 x_adjusted = x_segment.clone()
#                 if shift_unit != 0 and shift_unit != 'outlier':
#                     x_adjusted = x_adjusted - shift_unit  # 减去 shift_unit
#                 x_adjusted = x_adjusted.unsqueeze(1)
#                 # 通过对应的模型
#                 out = model(x_adjusted)
#                 # 输出调整
#                 if shift_unit != 0 and shift_unit != 'outlier':
#                     out = out + shift_unit ** 2
#                 output[mask] = out.squeeze()

#         # 处理异常值区间
#         shift_unit = 'outlier'
#         model = self.models[shift_unit]
#         mask = (abs_x >= self.thresholds[-1])
#         if mask.any():
#             x_segment = x_input[mask]
#             x_adjusted = x_segment.clone()
#             x_adjusted = x_adjusted.unsqueeze(1)
#             out = model(x_adjusted) + 64  # 根据您的代码，加上 64
#             output[mask] = out.squeeze()

#         return output.unsqueeze(1)

# # 定义加载已保存的集成模型的函数
# def load_ensemble_model(model_path, device):
#     # 加载检查点
#     checkpoint = torch.load(model_path, map_location=device)

#     # 重建模型
#     models = {}
#     shift_units = checkpoint['shift_units']

#     for shift_unit in shift_units:
#         sub_model_state = checkpoint['sub_models'][str(shift_unit)]
#         init_params = sub_model_state['init_params']
#         if shift_unit == 'outlier':
#             model = SquareApproximatorOutlier(
#                 shift_unit=init_params['shift_unit'],
#                 time_steps=init_params['time_steps'],
#                 N=init_params['N'],
#                 p=init_params['p']
#             ).to(device)
#         else:
#             model = SquareApproximator(
#                 time_steps=init_params['time_steps'],
#                 N=init_params['N'],
#                 p=init_params['p']
#             ).to(device)
#         model.load_state_dict(sub_model_state['state_dict'])
#         model.eval()
#         models[str(shift_unit)] = model

#     ensemble_model = SNNSqareNeuro(models=models, device=device).to(device)
#     ensemble_model.load_state_dict(checkpoint['model_state_dict'])
#     ensemble_model.eval()

#     return ensemble_model

# class SNNRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6, square_approx_model=None):
#         """
#         SNNRMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps
#         self.square_approx_model = square_approx_model  # 引入平方近似模型

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.float()
#         device = hidden_states.device

#         if self.square_approx_model is None:
#             # 使用原始的 pow(2)
#             variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         else:
#             # 使用平方近似模型
#             original_shape = hidden_states.shape
#             hidden_states_flat = hidden_states.view(-1, 1)

#             with torch.no_grad():
#                 squared = self.square_approx_model(hidden_states_flat).view(original_shape)

#             variance = squared.mean(-1, keepdim=True)

#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

# class SqrtNeuron(nn.Module):
#     def __init__(self, 
#                  base_threshold=1.0, 
#                  decay_rate=0.9,
#                  leak=0.1,
#                  time_steps=100):
#         super(SqrtNeuron, self).__init__()
#         self.base_threshold = base_threshold
#         self.decay_rate = decay_rate
#         self.leak = leak
#         self.time_steps = int(time_steps)

#         self.threshold_params = nn.Parameter(torch.zeros(self.time_steps))
#         self.pulse_params = nn.Parameter(torch.ones(self.time_steps) * 0.1)

#     def forward(self, x):
#         device = x.device
#         dtype = x.dtype
#         batch_size = x.shape[0]

#         V = torch.zeros(batch_size, device=device, dtype=dtype)
#         output = torch.zeros(batch_size, device=device, dtype=dtype)

#         for t in range(self.time_steps):
#             threshold = self.base_threshold + self.threshold_params[t]
#             pulse = self.pulse_params[t]

#             V = V * self.decay_rate + x.squeeze() - self.leak * V

#             spike_mask = (V >= threshold)
#             spike = spike_mask.float() * pulse

#             V[spike_mask] = 0.0

#             output += spike

#         return output.unsqueeze(1)

# def calculate_x_edges(N, x_start=0.0, x_end=0.01):
#     return torch.logspace(np.log10(x_start + 1e-6), np.log10(x_end), steps=N + 1)

# class SqrtApproximator(nn.Module):
#     def __init__(self, time_steps=100, N=64, x_edges=None):
#         super(SqrtApproximator, self).__init__()
#         self.time_steps = time_steps
#         self.N = N

#         self.neurons = nn.ModuleList([
#             SqrtNeuron(time_steps=int(self.time_steps / self.N))
#             for _ in range(self.N)
#         ])

#         if x_edges is None:
#             x_edges = calculate_x_edges(N)
#         self.register_buffer('x_edges', x_edges)

#     def forward(self, x):
#         device = x.device
#         dtype = x.dtype
#         output = torch.zeros_like(x, device=device, dtype=dtype)

#         for i, neuron in enumerate(self.neurons):
#             x_low = self.x_edges[i]
#             x_high = self.x_edges[i+1]
            
#             if i == 0:
#                 mask = (x >= x_low) & (x <= x_high)
#             else:
#                 mask = (x > x_low) & (x <= x_high)

#             if mask.any():
#                 x_segment = x[mask]
#                 out = neuron(x_segment)
#                 output[mask] = out.squeeze()

#         return output
    
    
# def load_sqrt_approximator(model_path, device='cuda'):
#     """
#     加载预训练的 SqrtApproximator 模型。

#     参数:
#     model_path (str): 保存的模型文件路径
#     device (str): 要加载模型的设备 ('cuda' 或 'cpu')

#     返回:
#     SqrtApproximator: 加载好的模型
#     """
#     device = torch.device(device)
    
#     # 加载检查点
#     checkpoint = torch.load(model_path, map_location=device)
    
#     # 获取初始化参数
#     init_params = checkpoint['init_params']
#     time_steps = init_params['time_steps']
#     N = init_params['N']
    
#     # 获取保存的 x_edges
#     x_edges = checkpoint['x_edges'].to(device)
    
#     # 创建模型实例
#     model = SqrtApproximator(time_steps=time_steps, N=N, x_edges=x_edges).to(device)
    
#     # 加载模型状态
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     # 设置为评估模式
#     model.eval()
    
#     print(f"模型已从 {model_path} 加载到 {device} 设备")
    
#     return model

# class SNNRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6, square_approx_model=None, sqrt_approx_model=None):
#         """
#         SNNRMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps
#         self.square_approx_model = square_approx_model  # 引入平方近似模型
#         self.sqrt_approx_model = sqrt_approx_model      # 引入平方根近似模型

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.float()
#         device = hidden_states.device

#         if self.square_approx_model is None:
#             # 使用原始的 pow(2)
#             variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         else:
#             # 使用平方近似模型
#             original_shape = hidden_states.shape
#             hidden_states_flat = hidden_states.view(-1, 1)

#             with torch.no_grad():
#                 squared = self.square_approx_model(hidden_states_flat).view(original_shape)

#             variance = squared.mean(-1, keepdim=True)

#         if self.sqrt_approx_model is None:
#             # 使用原始的 rsqrt
#             hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         else:
#             # 使用平方根近似模型
#             variance_adjusted = variance + self.variance_epsilon
#             variance_flat = variance_adjusted.view(-1, 1)

#             with torch.no_grad():
#                 inv_std = 1.0 / self.sqrt_approx_model(variance_flat)

#             inv_std = inv_std.view_as(variance_adjusted)
#             hidden_states = hidden_states * inv_std

#         return self.weight * hidden_states.to(input_dtype)

# class Synapsis(nn.Module):
#     '''
#     突触后神经元直接累计再发放，后续涉及
#     '''
#     def __init__(self, pre_ifneuron, layer, post_ifneuron, bits=8):
#         super().__init__()
#         self.pre_ifneuron = pre_ifneuron
#         self.layer = layer
#         self.post_ifneuron = post_ifneuron
#         self.bits = bits
#         self.max_value = 2**(bits - 1) - 1
#         self.pre_t = 0  # 初始化时间步计数器
#         self.post_t = 0

#     def _get_time_steps(self):
#         return self.max_value
    
#     def _reset_time(self):
#         self.pre_t = 0  # 初始化时间步计数器
#         self.post_t = 0

#     def _compute_scaling_factor(self, x):
#         return x.abs().max() / self.max_value

#     def _forward_ifneuron(self, x, ifneuron,t , layer=None):
#         time_steps = self._get_time_steps()
#         scaling_factor = self._compute_scaling_factor(x)
        
#         x_scaled = x / scaling_factor

#         outputs = []
#         for _ in range(time_steps):
#             output = ifneuron(x_scaled)
#             if output.abs().sum() == 0:  # 如果输出全为0，则提前结束
#                 break
#             t += 1  # 增加时间步计数
#             outputs.append(output)
#         ifneuron.reset()
#         accumulated_output = torch.stack(outputs).sum(dim=0)
        
#         return (accumulated_output * scaling_factor).to(x.dtype), t

#     def forward(self, x):
#         self._reset_time()
#         if isinstance(self.layer, nn.Embedding):
#             x = self.layer(x)
#             self.pre_t = 0  # 初始化时间步计数器
#             x, self.post_t = self._forward_ifneuron(x, self.post_ifneuron,self.post_t)
#             return x
#         else:
#             x, self.pre_t = self._forward_ifneuron(x, self.pre_ifneuron, self.pre_t)
#             x = self.layer(x)
#             x, self.post_t = self._forward_ifneuron(x, self.post_ifneuron, self.post_t)
#             return x
        
#     def __getattr__(self, name):
#         if name in ['weight', 'bias']:
#             return getattr(self.layer, name)
#         return super().__getattr__(name)

#     def reset_parameters(self):
#         if hasattr(self.layer, 'reset_parameters'):
#             self.layer.reset_parameters()
#         for ifneuron in [self.pre_ifneuron, self.post_ifneuron]:
#             if hasattr(ifneuron, 'reset_parameters'):
#                 ifneuron.reset_parameters()

#     def train(self, mode=True):
#         super().train(mode)
#         self.pre_ifneuron.train(mode)
#         self.layer.train(mode)
#         self.post_ifneuron.train(mode)
#         return self

#     def eval(self):
#         super().eval()
#         self.pre_ifneuron.eval()
#         self.layer.eval()
#         self.post_ifneuron.eval()
#         return self

#     def parameters(self):
#         return (
#             list(self.pre_ifneuron.parameters()) +
#             list(self.layer.parameters()) +
#             list(self.post_ifneuron.parameters()) 
#         )


# 定义自定义神经元类（用于正数部分）
class CustomNeuronPositive(nn.Module):
    def __init__(self, 
                 base_threshold=1.0, 
                 decay_rate=1.0,
                 time_steps=10):
        super(CustomNeuronPositive, self).__init__()
        self.base_threshold = base_threshold
        self.decay_rate = decay_rate
        self.time_steps = time_steps

        # 可训练参数：阈值和脉冲幅度在每个时间步上的值
        self.pos_threshold_params = nn.Parameter(torch.zeros(time_steps))  
        self.neg_threshold_params = nn.Parameter(torch.zeros(time_steps))  

        self.pos_pulse_params = nn.Parameter(torch.ones(time_steps) * 0.02)  
        self.neg_pulse_params = nn.Parameter(torch.ones(time_steps) * -0.02)  

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        
        V = torch.zeros(batch_size, device=device, dtype=dtype)
        output = torch.zeros(batch_size, device=device, dtype=dtype)
        
        for t in range(self.time_steps):
            # 获取当前时间步的阈值和脉冲幅度
            pos_threshold = self.base_threshold + self.pos_threshold_params[t]
            neg_threshold = -self.base_threshold + self.neg_threshold_params[t]
            pos_pulse = self.pos_pulse_params[t]
            neg_pulse = self.neg_pulse_params[t]
            
            # 更新膜电位
            V = V * self.decay_rate + x.squeeze()
            
            # 初始化脉冲为0
            spike = torch.zeros_like(V, device=device, dtype=dtype)
            
            # 正脉冲发放
            pos_spike_mask = (V >= pos_threshold)
            spike += pos_spike_mask.float() * pos_pulse
            
            # 负脉冲发放
            neg_spike_mask = (V <= neg_threshold)
            spike += neg_spike_mask.float() * neg_pulse
            
            # 累计输出
            output += spike

        return output.unsqueeze(1)

# 定义自定义神经元类（用于负数部分）
class CustomNeuronNegative(nn.Module):
    def __init__(self, 
                 pos_base_threshold=1.0, 
                 neg_base_threshold=-1.0, 
                 decay_rate=1.0,
                 time_steps=10):
        super(CustomNeuronNegative, self).__init__()
        self.pos_base_threshold = pos_base_threshold
        self.neg_base_threshold = neg_base_threshold
        self.decay_rate = decay_rate
        self.time_steps = int(time_steps)  # 确保 time_steps 为整数

        # 可训练参数：阈值和脉冲幅度在每个时间步上的值
        self.pos_threshold_params = nn.Parameter(torch.zeros(self.time_steps))  
        self.neg_threshold_params = nn.Parameter(torch.zeros(self.time_steps))  

        self.pos_pulse_params = nn.Parameter(torch.ones(self.time_steps) * 0.005)  
        self.neg_pulse_params = nn.Parameter(torch.ones(self.time_steps) * -0.005)  

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        
        V = torch.zeros(batch_size, device=device, dtype=dtype)
        output = torch.zeros(batch_size, device=device, dtype=dtype)
        
        for t in range(self.time_steps):
            # 获取当前时间步的阈值和脉冲幅度
            pos_threshold = self.pos_base_threshold + self.pos_threshold_params[t]
            neg_threshold = self.neg_base_threshold + self.neg_threshold_params[t]
            pos_pulse = self.pos_pulse_params[t]
            neg_pulse = self.neg_pulse_params[t]
            
            # 更新膜电位，对于负输入，使用 -x
            V = V * self.decay_rate - x.squeeze()
            
            # 初始化脉冲为0
            spike = torch.zeros_like(V, device=device, dtype=dtype)
            
            # 正脉冲发放
            pos_spike_mask = (V >= pos_threshold)
            spike += pos_spike_mask.float() * pos_pulse
            
            # 负脉冲发放
            neg_spike_mask = (V <= neg_threshold)
            spike += neg_spike_mask.float() * neg_pulse
            
            # 累计输出
            output += spike

        return output.unsqueeze(1)

# 定义 SiLU 正数部分近似模型
class SiLUPositiveApproximator(nn.Module):
    def __init__(self, time_steps=10):
        super(SiLUPositiveApproximator, self).__init__()
        # neuron1和neuron2用于拟合[0,1]区间的SiLU函数
        self.neuron1 = CustomNeuronPositive(
            base_threshold=0.3,  
            decay_rate=0.95,     
            time_steps=time_steps)
        
        self.neuron2 = CustomNeuronPositive(
            base_threshold=0.3,
            decay_rate=0.95,
            time_steps=time_steps)
        
        # neuron3用于拟合x > 1时的y = x
        self.neuron3 = CustomNeuronPositive(
            base_threshold=5.0,  
            decay_rate=1.0,
            time_steps=time_steps)
        
    def forward(self, x):
        out1 = self.neuron1(x)
        out2 = self.neuron2(x)
        out3 = self.neuron3(x)
        return out1 + out2 + out3

# 定义 SiLU 负数部分近似模型
class SiLUNegativeApproximator(nn.Module):
    def __init__(self):
        super(SiLUNegativeApproximator, self).__init__()
        # 神经元 A：接近 0 的负数区域，时间步数较多
        self.neuronA = CustomNeuronNegative(
            pos_base_threshold=1.0,  
            neg_base_threshold=-0.5,  
            decay_rate=0.995,         
            time_steps=20)           # 增加时间步数
        
        # 神经元 B：中间的负数区域，时间步数保持不变
        self.neuronB = CustomNeuronNegative(
            pos_base_threshold=1.0,  
            neg_base_threshold=-2.0,  
            decay_rate=0.99,
            time_steps=10)  # 时间步数为100
        
        # 神经元 C：远离 0 的负数区域，时间步数较少
        self.neuronC = CustomNeuronNegative(
            pos_base_threshold=1.0,  
            neg_base_threshold=-4.0,  
            decay_rate=0.99,
            time_steps=5)  # 时间步数较少
        
        # 分界点，设为可训练参数
        self.cutoff1 = nn.Parameter(torch.tensor(-2.0))
        self.cutoff2 = nn.Parameter(torch.tensor(-4.0))
    
    def forward(self, x):
        device = x.device
        dtype = x.dtype

        # 创建掩码，转换为布尔类型
        mask_A = (x >= self.cutoff1) & (x < 0)
        mask_B = (x >= self.cutoff2) & (x < self.cutoff1)
        mask_C = (x >= -6) & (x < self.cutoff2)
        mask_zero = x < -6  # 当 x < -6 时，输出为 0

        output = torch.zeros_like(x, device=device, dtype=dtype)

        # 计算神经元 A 的输出
        if mask_A.any():
            x_A = x[mask_A]
            outA = self.neuronA(x_A)
            output[mask_A] += outA.squeeze()

        # 计算神经元 B 的输出
        if mask_B.any():
            x_B = x[mask_B]
            outB = self.neuronB(x_B)
            output[mask_B] += outB.squeeze()

        # 计算神经元 C 的输出
        if mask_C.any():
            x_C = x[mask_C]
            outC = self.neuronC(x_C)
            output[mask_C] += outC.squeeze()

        # 对于 x < -6 的区域，输出为 0（无需处理，因为默认初始化为 0）

        return output

# 定义完整的 SiLU 近似模型
class SiLUApproximator(nn.Module):
    def __init__(self):
        super(SiLUApproximator, self).__init__()
        self.pos_model = SiLUPositiveApproximator(time_steps=10)
        self.neg_model = SiLUNegativeApproximator()
        
    def forward(self, x):
        device = x.device
        dtype = x.dtype

        # 创建掩码
        mask_pos = x >= 0
        mask_neg = x < 0

        output = torch.zeros_like(x, device=device, dtype=dtype)

        # 计算正数部分的输出
        if mask_pos.any():
            x_pos = x[mask_pos]
            out_pos = self.pos_model(x_pos)
            output[mask_pos] += out_pos.squeeze()

        # 计算负数部分的输出
        if mask_neg.any():
            x_neg = x[mask_neg]
            out_neg = self.neg_model(x_neg)
            output[mask_neg] += out_neg.squeeze()

        return output

# class SNNRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6, square_approx_model=None, sqrt_approx_model=None):
#         """
#         SNNRMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps
#         self.square_approx_model = square_approx_model  # 引入平方近似模型
#         self.sqrt_approx_model = sqrt_approx_model      # 引入平方根近似模型

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.float()
#         device = hidden_states.device

#         if self.square_approx_model is None:
#             # 使用原始的 pow(2)
#             variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         else:
#             # 使用平方近似模型
#             original_shape = hidden_states.shape
#             hidden_states_flat = hidden_states.view(-1, 1)

#             with torch.no_grad():
#                 squared = self.square_approx_model(hidden_states_flat).view(original_shape)

#             variance = squared.mean(-1, keepdim=True)

#         if self.sqrt_approx_model is None:
#             # 使用原始的 rsqrt
#             hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         else:
#             # 使用平方根近似模型
#             variance_adjusted = variance + self.variance_epsilon
#             variance_flat = variance_adjusted.view(-1, 1)

#             with torch.no_grad():
#                 inv_std = 1.0 / self.sqrt_approx_model(variance_flat)

#             inv_std = inv_std.view_as(variance_adjusted)
#             hidden_states = hidden_states * inv_std

#         return self.weight * hidden_states.to(input_dtype)

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->BrainGPT
class SNNRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        SNNRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->BrainGPT
class BrainGPTRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def load_silu_approximator(device, dtype):
    act_fn = SiLUApproximator().to(device).to(dtype)
    pos_checkpoint = os.path.join(os.path.dirname(__file__), 'model_pos.pth')
    neg_checkpoint = os.path.join(os.path.dirname(__file__), 'model_neg.pth')
    
    if os.path.exists(pos_checkpoint) and os.path.exists(neg_checkpoint):
        act_fn.pos_model.load_state_dict(
            torch.load(pos_checkpoint, map_location=device)
        )
        act_fn.neg_model.load_state_dict(
            torch.load(neg_checkpoint, map_location=device)
        )
    else:
        raise FileNotFoundError(
            f"SiLUApproximator parameters not found at {pos_checkpoint} and {neg_checkpoint}"
        )
    return act_fn

# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->BrainGPT
class BrainGPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # 初始化自定义神经元
        pre_ifneuron = EI_IFNeuron()
        post_ifneuron = EI_IFNeuron()
        
        self.gate_proj = Synapsis(pre_ifneuron, nn.Linear(self.hidden_size, self.intermediate_size, bias=False), post_ifneuron)
        self.up_proj = Synapsis(pre_ifneuron, nn.Linear(self.hidden_size, self.intermediate_size, bias=False), post_ifneuron)
        self.down_proj = Synapsis(pre_ifneuron, nn.Linear(self.intermediate_size, self.hidden_size, bias=False), post_ifneuron)
        
        # 加载 SiLUApproximator
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.act_fn = load_silu_approximator(device, dtype)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class BrainGPTAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: BrainGPTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # 初始化自定义神经元
        pre_ifneuron = EI_IFNeuron()
        post_ifneuron = EI_IFNeuron()
        
        self.q_proj = Synapsis(pre_ifneuron, nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True), post_ifneuron)
        self.k_proj = Synapsis(pre_ifneuron, nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True), post_ifneuron)
        self.v_proj = Synapsis(pre_ifneuron, nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True), post_ifneuron)
        self.o_proj = Synapsis(pre_ifneuron, nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False), post_ifneuron)

        self.rotary_emb = BrainGPTRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()
        # 初始化 SNNMatmul 实例
        matmul_pre_ifneuron = EI_IFNeuron()
        matmul_post_ifneuron = EI_IFNeuron()
        snn_matmul = SNNMatmul(matmul_pre_ifneuron, matmul_post_ifneuron, bits=8)  # 根据需要调整 bits
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        

        attn_weights = snn_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        soft_max_neuron = EI_IFNeuron()
        snn_softmax = SNNSoftmax(neuron=soft_max_neuron, dim=-1, dtype=torch.float32)
        attn_weights = snn_softmax(attn_weights).to(query_states.dtype)
        snn_dropout = SNNDropout(neuron=soft_max_neuron)
        attn_weights = snn_dropout(attn_weights, p=self.attention_dropout, if_train=self.training)
        attn_output = snn_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BrainGPTFlashAttention2(BrainGPTAttention):
    """
    BrainGPT flash attention module, following BrainGPT attention module. This module inherits from `BrainGPTAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and self.config.use_sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Decide whether to use SWA or not by layer index.
        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.mistral.modeling_mistral.MistralSdpaAttention with Mistral->BrainGPT
class BrainGPTSdpaAttention(BrainGPTAttention):
    """
    BrainGPT attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `BrainGPTAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from BrainGPTAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "BrainGPTModel is using BrainGPTSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


BrainGPT_ATTENTION_CLASSES = {
    "eager": BrainGPTAttention,
    "flash_attention_2": BrainGPTFlashAttention2,
    "sdpa": BrainGPTSdpaAttention,
}


class BrainGPTDecoderLayer(nn.Module):
    def __init__(self, config: BrainGPTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = BrainGPT_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = BrainGPTMLP(config)
        self.input_layernorm = SNNRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SNNRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


BrainGPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BrainGPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare BrainGPT Model outputting raw hidden-states without any specific head on top.",
    BrainGPT_START_DOCSTRING,
)
class BrainGPTPreTrainedModel(PreTrainedModel):
    config_class = BrainGPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BrainGPTDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear or isinstance(module.layer, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding or isinstance(module.layer, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

BrainGPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare BrainGPT Model outputting raw hidden-states without any specific head on top.",
    BrainGPT_START_DOCSTRING,
)
class BrainGPTModel(BrainGPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`BrainGPTDecoderLayer`]

    Args:
        config: BrainGPTConfig
    """

    def __init__(self, config: BrainGPTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # 初始化自定义神经元
        pre_ifneuron = EI_IFNeuron()
        post_ifneuron = EI_IFNeuron()
        
        
        self.embed_tokens = Synapsis(pre_ifneuron, nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx), post_ifneuron)
        self.layers = nn.ModuleList(
            [BrainGPTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = SNNRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.baseline_loss = None
        self.loss_history = []
        self.window_size = 100
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(BrainGPT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of BrainGPT. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    def compute_global_modulation(self, current_loss):
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        
        self.baseline_loss = sum(self.loss_history) / len(self.loss_history)
        return torch.sigmoid(self.config.beta * (self.baseline_loss - current_loss))

    def update_neuron_parameters(self, G):
        for module in self.modules():
            if isinstance(module, EI_IFNeuron):
                module.base_threshold += self.config.eta_theta * G * (self.config.S_target - module.total_output / module.time_step)
                module.alpha += self.config.eta_alpha * G * (module.membrane_potential.mean() - self.config.V_target)
                module.attenuation_rate += self.config.eta_r * G * (module.membrane_potential.mean() - self.config.V_rest)

    def enable_stdp_mode(self):
        for module in self.modules():
            if isinstance(module, Synapsis):
                module.if_STDP_Inspire = True

    def disable_stdp_mode(self):
        for module in self.modules():
            if isinstance(module, Synapsis):
                module.if_STDP_Inspire = False

    def update_current_task_loss(self, loss):
        for module in self.modules():
            if isinstance(module, Synapsis):
                module.current_task_loss = loss

class BrainGPTForCausalLM(BrainGPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = BrainGPTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head =nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(BrainGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BrainGPTForCausalLM

        >>> model = BrainGPTForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            self.model.update_current_task_loss(loss.item())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_loss(self, logits, labels):
        # 计算任务特定损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        task_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # 计算其他损失组件
        stdp_loss = self.compute_stdp_loss()
        neuron_loss = self.compute_neuron_loss()
        time_step_loss = self.compute_time_step_loss()
        synaptic_norm_loss = self.compute_synaptic_normalization_loss()
        regularization_loss = self.compute_regularization_loss()

        # 组合所有损失组件
        total_loss = (
            self.config.lambda_task * task_loss +
            self.config.lambda_stdp * stdp_loss +
            self.config.lambda_neuron * neuron_loss +
            self.config.lambda_time * time_step_loss +
            self.config.lambda_C * synaptic_norm_loss +
            self.config.lambda_reg * regularization_loss
        )

        return total_loss

    def compute_stdp_loss(self):
        return sum(((module.layer.weight - module.tag) ** 2).sum()
                   for module in self.modules() if isinstance(module, Synapsis))

    def compute_neuron_loss(self):
        return sum((module.total_output / module.time_step - self.config.S_target) ** 2 +
                   (module.membrane_potential.mean() - self.config.V_target) ** 2 +
                   (module.membrane_potential.mean() - self.config.V_rest) ** 2
                   for module in self.modules() if isinstance(module, EI_IFNeuron))

    def compute_time_step_loss(self):
        T = sum(module.time_step * torch.sigmoid((module.membrane_potential - module.threshold) / self.config.lambda_T)
                for module in self.modules() if isinstance(module, EI_IFNeuron))
        return (T - self.config.T_target) ** 2

    def compute_synaptic_normalization_loss(self):
        return sum((module.layer.weight.sum(dim=1) - self.config.C) ** 2
                   for module in self.modules() if isinstance(module, Synapsis))

    def compute_regularization_loss(self):
        return sum(p.pow(2).sum() for p in self.parameters())

    def train_step(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        self.model.enable_stdp_mode()
        outputs = self(input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
        loss = outputs.loss
        G = self.model.compute_global_modulation(loss.item())
        self.model.update_neuron_parameters(G)
        self.model.disable_stdp_mode()
        return loss.item()
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The BrainGPT Model transformer with a sequence classification head on top (linear layer).

    [`BrainGPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    BrainGPT_START_DOCSTRING,
)
class BrainGPTForSequenceClassification(BrainGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = BrainGPTModel(config)
        pre_ifneuron = EI_IFNeuron()
        post_ifneuron = EI_IFNeuron()
        
        
        self.score = Synapsis(pre_ifneuron, nn.Linear(config.hidden_size, self.num_labels, bias=False), post_ifneuron)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(BrainGPT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
