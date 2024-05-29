class FrozenDict:
    def __init__(self, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", trained_betas=None, 
                 solver_order=2, prediction_type="epsilon", thresholding=False, dynamic_thresholding_ratio=0.995, 
                 sample_max_value=1.0, algorithm_type="dpmsolver++", solver_type="midpoint", lower_order_final=True, 
                 clip_sample=False, clip_sample_range=1.0):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.trained_betas = trained_betas if trained_betas is not None else []
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
