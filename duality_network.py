import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, List, Union, Dict, Any

class DualityLayer(nn.Module):
    """
    A neural network layer implementing the duality between sum and product operations.
    This layer processes inputs through both a standard linear transformation (sum path)
    and a multiplicative transformation (product path), then combines the results based
    on a learnable parameter alpha.
    
    Based on the First Fundamental Theorem of Analysis:
    - Sum Path: linear combination Σ w_i * x_i
    - Product Path: multiplicative combination Π x_i^w_i
    - Conversion: ln(Π x_i^w_i) = Σ w_i * ln(x_i)
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        adaptive_alpha: bool = True,
        initial_alpha: float = 0.5,
        epsilon: float = 1e-6,
        activation: str = 'relu'
    ):
        """
        Initialize the DualityLayer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias parameters
            adaptive_alpha: Whether to learn alpha or use fixed value
            initial_alpha: Initial value for alpha (0 = pure product, 1 = pure sum)
            epsilon: Small constant for numerical stability
            activation: Activation function to use
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adaptive_alpha = adaptive_alpha
        self.epsilon = epsilon
        
        # Weights and biases for sum path
        self.weight_sum = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_sum = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_sum', None)
            
        # Weights and biases for product path
        self.weight_prod = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_prod = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_prod', None)
        
        # Alpha parameter for mixing sum and product paths
        if adaptive_alpha:
            self.alpha_weights = nn.Parameter(torch.Tensor(out_features, in_features))
            self.alpha_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_buffer('alpha', torch.tensor(initial_alpha))
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = lambda x: x
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters."""
        nn.init.kaiming_uniform_(self.weight_sum, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_prod, a=math.sqrt(5))
        
        if self.bias_sum is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_sum)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_sum, -bound, bound)
            
        if self.bias_prod is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_prod)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_prod, -bound, bound)
            
        if self.adaptive_alpha:
            nn.init.zeros_(self.alpha_weights)
            # Initialize alpha_bias to logit of initial_alpha (0.5)
            nn.init.constant_(self.alpha_bias, 0.0) 
    
    def compute_alpha(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the adaptive alpha value.
        
        Args:
            x: Input tensor [batch_size, in_features]
            
        Returns:
            Alpha tensor [batch_size, out_features]
        """
        if self.adaptive_alpha:
            # Compute alpha based on input
            alpha = torch.sigmoid(F.linear(x, self.alpha_weights, self.alpha_bias))
            return alpha
        else:
            # Use fixed alpha value
            return self.alpha.expand(x.size(0), self.out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DualityLayer.
        
        Args:
            x: Input tensor [batch_size, in_features]
            
        Returns:
            Output tensor [batch_size, out_features]
        """
        # Ensure input is positive for log operation in product path
        x_safe = x + self.epsilon
        
        # Compute sum path: Σ w_i * x_i
        sum_path = F.linear(x, self.weight_sum, self.bias_sum)
        
        # Compute product path: Π x_i^w_i = exp(Σ w_i * ln(x_i))
        log_x = torch.log(x_safe)
        prod_path_log = F.linear(log_x, self.weight_prod)
        prod_path = torch.exp(prod_path_log)
        if self.bias_prod is not None:
            prod_path = prod_path + self.bias_prod
        
        # Compute mixing parameter alpha
        alpha = self.compute_alpha(x)
        
        # Combine sum and product paths
        output = alpha * sum_path + (1 - alpha) * prod_path
        
        # Apply activation function
        output = self.activation(output)
        
        return output


class DualityConv2d(nn.Module):
    """
    A convolutional layer implementing the duality between sum and product operations.
    This layer processes inputs through both a standard convolution (sum path)
    and a multiplicative convolution (product path), then combines the results.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        adaptive_alpha: bool = True,
        initial_alpha: float = 0.5,
        epsilon: float = 1e-6,
        activation: str = 'relu'
    ):
        """
        Initialize the DualityConv2d layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output channels
            bias: Whether to include bias parameters
            adaptive_alpha: Whether to learn alpha or use fixed value
            initial_alpha: Initial value for alpha (0 = pure product, 1 = pure sum)
            epsilon: Small constant for numerical stability
            activation: Activation function to use
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.adaptive_alpha = adaptive_alpha
        self.epsilon = epsilon
        
        # Convolution for sum path
        self.conv_sum = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        
        # Convolution for product path
        self.conv_prod = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        
        # Alpha parameter for mixing sum and product paths
        if adaptive_alpha:
            self.alpha_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
            )
        else:
            self.register_buffer('alpha', torch.tensor(initial_alpha))
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = lambda x: x
    
    def compute_alpha(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the adaptive alpha value.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Alpha tensor [batch_size, out_channels, out_height, out_width]
        """
        if self.adaptive_alpha:
            # Compute alpha based on input
            alpha = torch.sigmoid(self.alpha_conv(x))
            return alpha
        else:
            # Calculate output dimensions
            h_out = (x.size(2) + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
            w_out = (x.size(3) + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
            # Use fixed alpha value
            return self.alpha.expand(x.size(0), self.out_channels, h_out, w_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DualityConv2d layer.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor [batch_size, out_channels, out_height, out_width]
        """
        # Ensure input is positive for log operation in product path
        x_safe = x + self.epsilon
        
        # Compute sum path: standard convolution
        sum_path = self.conv_sum(x)
        
        # Compute product path: convolution in log domain, then exp
        log_x = torch.log(x_safe)
        prod_path_log = self.conv_prod(log_x)
        prod_path = torch.exp(prod_path_log)
        
        # Compute mixing parameter alpha
        alpha = self.compute_alpha(x)
        
        # Combine sum and product paths
        output = alpha * sum_path + (1 - alpha) * prod_path
        
        # Apply activation function
        output = self.activation(output)
        
        return output


class DualityRNN(nn.Module):
    """
    A recurrent neural network cell implementing the duality between sum and product operations.
    This layer processes inputs through both a standard RNN update (sum path)
    and a multiplicative update (product path), then combines the results.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        adaptive_alpha: bool = True,
        initial_alpha: float = 0.5,
        epsilon: float = 1e-6,
        activation: str = 'tanh'
    ):
        """
        Initialize the DualityRNN cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            bias: Whether to include bias parameters
            adaptive_alpha: Whether to learn alpha or use fixed value
            initial_alpha: Initial value for alpha (0 = pure product, 1 = pure sum)
            epsilon: Small constant for numerical stability
            activation: Activation function to use
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.adaptive_alpha = adaptive_alpha
        self.epsilon = epsilon
        
        # Sum path weights
        self.weight_ih_sum = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh_sum = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih_sum = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh_sum = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih_sum', None)
            self.register_parameter('bias_hh_sum', None)
        
        # Product path weights
        self.weight_ih_prod = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh_prod = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih_prod = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh_prod = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih_prod', None)
            self.register_parameter('bias_hh_prod', None)
        
        # Alpha parameter for mixing sum and product paths
        if adaptive_alpha:
            self.weight_ih_alpha = nn.Parameter(torch.Tensor(hidden_size, input_size))
            self.weight_hh_alpha = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.bias_alpha = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_buffer('alpha', torch.tensor(initial_alpha))
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = lambda x: x
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters."""
        # Initialize weights with orthogonal initialization
        nn.init.orthogonal_(self.weight_ih_sum)
        nn.init.orthogonal_(self.weight_hh_sum)
        nn.init.orthogonal_(self.weight_ih_prod)
        nn.init.orthogonal_(self.weight_hh_prod)
        
        # Initialize biases to zero
        if self.bias_ih_sum is not None:
            nn.init.zeros_(self.bias_ih_sum)
            nn.init.zeros_(self.bias_hh_sum)
            nn.init.zeros_(self.bias_ih_prod)
            nn.init.zeros_(self.bias_hh_prod)
        
        # Initialize alpha weights
        if self.adaptive_alpha:
            nn.init.zeros_(self.weight_ih_alpha)
            nn.init.zeros_(self.weight_hh_alpha)
            # Initialize alpha_bias to logit of initial_alpha (0.5)
            nn.init.constant_(self.bias_alpha, 0.0)
    
    def compute_alpha(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the adaptive alpha value.
        
        Args:
            x: Input tensor [batch_size, input_size]
            h: Hidden state tensor [batch_size, hidden_size]
            
        Returns:
            Alpha tensor [batch_size, hidden_size]
        """
        if self.adaptive_alpha:
            # Compute alpha based on input and hidden state
            ih = torch.mm(x, self.weight_ih_alpha.t())
            hh = torch.mm(h, self.weight_hh_alpha.t())
            alpha = torch.sigmoid(ih + hh + self.bias_alpha)
            return alpha
        else:
            # Use fixed alpha value
            return self.alpha.expand(x.size(0), self.hidden_size)
    
    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the DualityRNN cell.
        
        Args:
            x: Input tensor [batch_size, input_size]
            hx: Hidden state tensor [batch_size, hidden_size] or None
            
        Returns:
            New hidden state [batch_size, hidden_size]
        """
        if hx is None:
            hx = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        # Ensure input and hidden state are positive for log operation
        x_safe = x + self.epsilon
        hx_safe = hx + self.epsilon
        
        # Compute sum path
        ih_sum = F.linear(x, self.weight_ih_sum, self.bias_ih_sum)
        hh_sum = F.linear(hx, self.weight_hh_sum, self.bias_hh_sum)
        sum_path = ih_sum + hh_sum
        
        # Compute product path
        log_x = torch.log(x_safe)
        log_hx = torch.log(hx_safe)
        ih_prod_log = F.linear(log_x, self.weight_ih_prod)
        hh_prod_log = F.linear(log_hx, self.weight_hh_prod)
        prod_path_log = ih_prod_log + hh_prod_log
        prod_path = torch.exp(prod_path_log)
        if self.bias_ih_prod is not None:
            prod_path = prod_path + self.bias_ih_prod + self.bias_hh_prod
        
        # Compute mixing parameter alpha
        alpha = self.compute_alpha(x, hx)
        
        # Combine sum and product paths
        output = alpha * sum_path + (1 - alpha) * prod_path
        
        # Apply activation function
        output = self.activation(output)
        
        return output


class DualityLSTM(nn.Module):
    """
    An LSTM cell implementing the duality between sum and product operations.
    This layer processes inputs through both standard LSTM gates (sum path)
    and multiplicative gates (product path), then combines the results.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        adaptive_alpha: bool = True,
        initial_alpha: float = 0.5,
        epsilon: float = 1e-6
    ):
        """
        Initialize the DualityLSTM cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            bias: Whether to include bias parameters
            adaptive_alpha: Whether to learn alpha or use fixed value
            initial_alpha: Initial value for alpha (0 = pure product, 1 = pure sum)
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.adaptive_alpha = adaptive_alpha
        self.epsilon = epsilon
        
        # Sum path: standard LSTM
        self.lstm_sum = nn.LSTMCell(input_size, hidden_size, bias)
        
        # Product path: multiplicative LSTM
        # For each gate (input, forget, cell, output)
        self.weight_ih_prod = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh_prod = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih_prod = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh_prod = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih_prod', None)
            self.register_parameter('bias_hh_prod', None)
        
        # Alpha parameter for mixing sum and product paths
        if adaptive_alpha:
            self.weight_ih_alpha = nn.Parameter(torch.Tensor(hidden_size, input_size))
            self.weight_hh_alpha = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.bias_alpha = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_buffer('alpha', torch.tensor(initial_alpha))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters."""
        # Initialize product path weights
        nn.init.orthogonal_(self.weight_ih_prod)
        nn.init.orthogonal_(self.weight_hh_prod)
        
        # Initialize biases
        if self.bias_ih_prod is not None:
            nn.init.zeros_(self.bias_ih_prod)
            nn.init.zeros_(self.bias_hh_prod)
            # Set forget gate bias to 1
            nn.init.constant_(self.bias_ih_prod[self.hidden_size:2*self.hidden_size], 1.0)
        
        # Initialize alpha weights
        if self.adaptive_alpha:
            nn.init.zeros_(self.weight_ih_alpha)
            nn.init.zeros_(self.weight_hh_alpha)
            # Initialize alpha_bias to logit of initial_alpha (0.5)
            nn.init.constant_(self.bias_alpha, 0.0)
    
    def compute_alpha(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the adaptive alpha value.
        
        Args:
            x: Input tensor [batch_size, input_size]
            h: Hidden state tensor [batch_size, hidden_size]
            
        Returns:
            Alpha tensor [batch_size, hidden_size]
        """
        if self.adaptive_alpha:
            # Compute alpha based on input and hidden state
            ih = torch.mm(x, self.weight_ih_alpha.t())
            hh = torch.mm(h, self.weight_hh_alpha.t())
            alpha = torch.sigmoid(ih + hh + self.bias_alpha)
            return alpha
        else:
            # Use fixed alpha value
            return self.alpha.expand(x.size(0), self.hidden_size)
    
    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the DualityLSTM cell.
        
        Args:
            x: Input tensor [batch_size, input_size]
            state: Tuple of (h, c) where h is the hidden state and c is the cell state
                  Each has shape [batch_size, hidden_size]
            
        Returns:
            Tuple of (h', (h', c')) where h' is the new hidden state and c' is the new cell state
        """
        h, c = state
        
        # Ensure input and states are positive for log operations
        x_safe = x + self.epsilon
        h_safe = h + self.epsilon
        
        # Compute sum path (standard LSTM)
        h_sum, c_sum = self.lstm_sum(x, (h, c))
        
        # Compute product path (multiplicative LSTM in log domain)
        # Gate indices
        i_s, f_s, g_s, o_s = 0, self.hidden_size, 2*self.hidden_size, 3*self.hidden_size
        i_e, f_e, g_e, o_e = i_s+self.hidden_size, f_s+self.hidden_size, g_s+self.hidden_size, o_s+self.hidden_size
        
        # Log domain transformations
        log_x = torch.log(x_safe)
        log_h = torch.log(h_safe)
        
        # Compute gates in log domain, then transform back
        gates_log = F.linear(log_x, self.weight_ih_prod) + F.linear(log_h, self.weight_hh_prod)
        if self.bias_ih_prod is not None:
            gates_log = gates_log + self.bias_ih_prod + self.bias_hh_prod
        
        # Apply non-linearities
        i = torch.sigmoid(gates_log[:, i_s:i_e])
        f = torch.sigmoid(gates_log[:, f_s:f_e])
        g = torch.tanh(gates_log[:, g_s:g_e])
        o = torch.sigmoid(gates_log[:, o_s:o_e])
        
        # Update cell state and hidden state
        c_prod = f * c + i * g
        h_prod = o * torch.tanh(c_prod)
        
        # Compute mixing parameter alpha
        alpha = self.compute_alpha(x, h)
        
        # Combine sum and product paths
        c_new = alpha * c_sum + (1 - alpha) * c_prod
        h_new = alpha * h_sum + (1 - alpha) * h_prod
        
        return h_new, (h_new, c_new)


class DualityNetwork(nn.Module):
    """
    A network composed of duality layers that can switch between summing and product operations.
    This implementation allows for building either feedforward or convolutional architectures.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        out_features: int,
        network_type: str = 'feedforward',
        adaptive_alpha: bool = True,
        initial_alpha: float = 0.5,
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        """
        Initialize the DualityNetwork.
        
        Args:
            in_features: Size of input features (or channels for CNN)
            hidden_dims: List of hidden dimensions
            out_features: Size of output features (or channels for CNN)
            network_type: Type of network ('feedforward', 'cnn', or 'rnn')
            adaptive_alpha: Whether to learn alpha or use fixed value
            initial_alpha: Initial value for alpha (0 = pure product, 1 = pure sum)
            dropout: Dropout probability
            activation: Activation function to use
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.network_type = network_type
        
        layers = []
        
        if network_type == 'feedforward':
            # Feedforward network with DualityLayers
            input_dim = in_features
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(
                    DualityLayer(
                        input_dim, hidden_dim, 
                        adaptive_alpha=adaptive_alpha, 
                        initial_alpha=initial_alpha,
                        activation=activation
                    )
                )
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            # Output layer
            layers.append(
                DualityLayer(
                    input_dim, out_features, 
                    adaptive_alpha=adaptive_alpha, 
                    initial_alpha=initial_alpha,
                    activation='linear'
                )
            )
        
        elif network_type == 'cnn':
            # Convolutional network with DualityConv2d layers
            in_channels = in_features
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(
                    DualityConv2d(
                        in_channels, hidden_dim, 
                        kernel_size=3, stride=1, padding=1,
                        adaptive_alpha=adaptive_alpha, 
                        initial_alpha=initial_alpha,
                        activation=activation
                    )
                )
                if i % 2 == 1:  # Downsample every other layer
                    layers.append(nn.MaxPool2d(2))
                if dropout > 0:
                    layers.append(nn.Dropout2d(dropout))
                in_channels = hidden_dim
            
            # Global average pooling and final classification layer
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(nn.Flatten())
            layers.append(
                DualityLayer(
                    in_channels, out_features,
                    adaptive_alpha=adaptive_alpha,
                    initial_alpha=initial_alpha,
                    activation='linear'
                )
            )
        
        elif network_type == 'rnn':
            # RNN with DualityRNN layers
            self.rnn_layers = nn.ModuleList([
                DualityRNN(
                    in_features if i == 0 else hidden_dims[i-1],
                    hidden_dims[i],
                    adaptive_alpha=adaptive_alpha,
                    initial_alpha=initial_alpha,
                    activation='tanh'
                )
                for i in range(len(hidden_dims))
            ])
            
            # Output layer
            layers.append(
                DualityLayer(
                    hidden_dims[-1], out_features,
                    adaptive_alpha=adaptive_alpha,
                    initial_alpha=initial_alpha,
                    activation='linear'
                )
            )
        
        self.layers = nn.Sequential(*layers) if network_type != 'rnn' else None
    
    def forward(self, x: torch.Tensor, hidden: Optional[List[torch.Tensor]] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of the DualityNetwork.
        
        Args:
            x: Input tensor
                - For feedforward: [batch_size, in_features]
                - For CNN: [batch_size, in_channels, height, width]
                - For RNN: [batch_size, seq_len, in_features]
            hidden: Initial hidden states for RNN (optional)
            
        Returns:
            Output tensor or tuple of (output, hidden_states) for RNN
        """
        if self.network_type != 'rnn':
            return self.layers(x)
        else:
            # Process sequence through RNN layers
            batch_size, seq_len, _ = x.shape
            
            # Initialize hidden states if not provided
            if hidden is None:
                hidden = [torch.zeros(batch_size, layer.hidden_size, device=x.device)
                         for layer in self.rnn_layers]
            
            # Process each time step
            outputs = []
            for t in range(seq_len):
                # Get input at current time step
                x_t = x[:, t, :]
                
                # Process through each RNN layer
                for i, rnn_layer in enumerate(self.rnn_layers):
                    x_t = rnn_layer(x_t, hidden[i])
                    hidden[i] = x_t
                
                outputs.append(x_t)
            
            # Stack outputs along sequence dimension
            outputs = torch.stack(outputs, dim=1)
            
            # Apply output layer to final output
            if len(self.layers) > 0:
                final_output = self.layers[0](outputs[:, -1, :])
            else:
                final_output = outputs[:, -1, :]
            
            return final_output, hidden


# Example usage
def create_duality_classifier(input_dim, num_classes, hidden_dims=[128, 64], network_type='feedforward'):
    """Create a classification model using DualityNetwork."""
    model = DualityNetwork(
        in_features=input_dim,
        hidden_dims=hidden_dims,
        out_features=num_classes,
        network_type=network_type,
        adaptive_alpha=True,
        initial_alpha=0.5,
        dropout=0.2
    )
    return model


def create_duality_image_classifier(in_channels, num_classes, hidden_dims=[32, 64, 128, 256]):
    """Create an image classification model using DualityNetwork with CNN architecture."""
    model = DualityNetwork(
        in_features=in_channels,
        hidden_dims=hidden_dims,
        out_features=num_classes,
        network_type='cnn',
        adaptive_alpha=True,
        initial_alpha=0.5,
        dropout=0.2
    )
    return model


def create_duality_sequence_model(input_dim, hidden_dims=[64, 128], output_dim=1):
    """Create a sequence model using DualityNetwork with RNN architecture."""
    model = DualityNetwork(
        in_features=input_dim,
        hidden_dims=hidden_dims,
        out_features=output_dim,
        network_type='rnn',
        adaptive_alpha=True,
        initial_alpha=0.5,
        dropout=0.2
    )
    return model


def test_sum_product_duality():
    """Test the duality between sum and product operations using log transformations."""
    # Create random input and weights
    x = torch.rand(10, 5) + 0.1  # Ensure positive values
    w = torch.rand(5)
    
    # Sum operation: Σ w_i * x_i
    sum_result = torch.sum(w * x, dim=1)
    
    # Product operation: Π x_i^w_i
    prod_result = torch.prod(x ** w, dim=1)
    
    # Log transformation: ln(Π x_i^w_i) = Σ w_i * ln(x_i)
    log_prod_result = torch.sum(w * torch.log(x), dim=1)
    
    # Compare results
    print("Direct product:", prod_result[:3])
    print("Exp(Sum of logs):", torch.exp(log_prod_result[:3]))
    print("Difference:", torch.abs(prod_result - torch.exp(log_prod_result)).max().item())


if __name__ == "__main__":
    # Test sum-product duality
    test_sum_product_duality()
    
    # Create and test models
    # Feedforward model for classification
    ff_model = create_duality_classifier(input_dim=784, num_classes=10)
    dummy_input = torch.randn(32, 784)
    output = ff_model(dummy_input)
    print(f"Feedforward model output shape: {output.shape}")
    
    # CNN model for image classification
    cnn_model = create_duality_image_classifier(in_channels=3, num_classes=10)
    dummy_img = torch.randn(32, 3, 32, 32)
    output = cnn_model(dummy_img)
    print(f"CNN model output shape: {output.shape}")
    
    # RNN model for sequence data
    rnn_model = create_duality_sequence_model(input_dim=10)
    dummy_seq = torch.randn(32, 20, 10)  # batch_size, seq_len, input_dim
    output, hidden = rnn_model(dummy_seq)
    print(f"RNN model output shape: {output.shape}")
    print(f"RNN hidden states: {len(hidden)}")