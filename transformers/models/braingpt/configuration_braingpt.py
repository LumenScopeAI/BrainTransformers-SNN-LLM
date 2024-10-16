class BrainGPTConfig(PretrainedConfig):
    model_type = "braingpt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        
        # STDP 和神经元参数
        beta=0.1,
        S_target=0.1,
        V_target=-65.0,
        V_rest=-70.0,
        eta_theta=0.01,
        eta_alpha=0.01,
        eta_r=0.01,
        lambda_T=0.1,
        T_target=10,
        C=1.0,

        # 损失函数权重
        lambda_task=1.0,
        lambda_stdp=0.1,
        lambda_neuron=0.1,
        lambda_time=0.1,
        lambda_C=0.1,
        lambda_reg=0.01,
        
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        # STDP 和神经元参数
        self.beta = beta
        self.S_target = S_target
        self.V_target = V_target
        self.V_rest = V_rest
        self.eta_theta = eta_theta
        self.eta_alpha = eta_alpha
        self.eta_r = eta_r
        self.lambda_T = lambda_T
        self.T_target = T_target
        self.C = C

        # 损失函数权重
        self.lambda_task = lambda_task
        self.lambda_stdp = lambda_stdp
        self.lambda_neuron = lambda_neuron
        self.lambda_time = lambda_time
        self.lambda_C = lambda_C
        self.lambda_reg = lambda_reg

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
