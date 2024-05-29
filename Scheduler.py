import numpy as np
import onnxruntime as ort
from typing import List
from FrozenDict import FrozenDict
import MyTensor  # Assuming MyTensor is defined as before
import ArrayUtils  # Assuming ArrayUtils is defined as before

class Scheduler:
    def set_timesteps(self, num_inference_steps: int) -> List[int]:
        pass
    
    def scale_model_input(self, sample: MyTensor, step_index: int) -> MyTensor:
        pass
    
    def step(self, model_output: MyTensor, step_index: int, sample: MyTensor) -> MyTensor:
        pass
    
    def get_init_noise_sigma(self) -> float:
        pass

class EulerAncestralDiscreteScheduler(Scheduler):
    def __init__(self, config: FrozenDict):
        self.config = config
        self.random = np.random.default_rng()
        self.betas = self._initialize_betas(config)
        self.alphas = [1.0 - beta for beta in self.betas]
        self.alphas_cumprod = self._compute_alphas_cumprod(self.alphas)
        self.sigmas = self._initialize_sigmas(self.alphas_cumprod)
        self.timesteps = list(reversed(np.arange(0, len(self.alphas_cumprod), dtype=int)))
        self.init_noise_sigma = max(self.sigmas)
        self.is_scale_input_called = False
        self.num_inference_steps = 0
        self.num_train_timesteps = 1000

    def _initialize_betas(self, config: FrozenDict) -> List[float]:
        if config.trained_betas:
            return config.trained_betas
        elif config.beta_schedule == "linear":
            return ArrayUtils.linspace(config.beta_start, config.beta_end, self.num_train_timesteps)
        elif config.beta_schedule == "scaled_linear":
            return [pow(x, 2) for x in ArrayUtils.linspace(pow(config.beta_start, 0.5), pow(config.beta_end, 0.5), self.num_train_timesteps)]
        elif config.beta_schedule == "squaredcos_cap_v2":
            return self.betas_for_alpha_bar(self.num_train_timesteps, 0.999)
        return []

    def _compute_alphas_cumprod(self, alphas: List[float]) -> List[float]:
        alphas_cumprod = [alphas[0]]
        for alpha in alphas[1:]:
            alphas_cumprod.append(alphas_cumprod[-1] * alpha)
        return alphas_cumprod

    def _initialize_sigmas(self, alphas_cumprod: List[float]) -> List[float]:
        sigmas = [np.sqrt((1 - alpha) / alpha) for alpha in alphas_cumprod]
        sigmas.reverse()
        sigmas.append(0.0)
        return sigmas

    def set_timesteps(self, num_inference_steps: int) -> List[int]:
        self.num_inference_steps = num_inference_steps
        timesteps = ArrayUtils.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
        sigmas = np.interp(timesteps, np.arange(len(self.alphas_cumprod)), self.sigmas)
        sigmas = np.append(sigmas, sigmas[-1])
        self.sigmas = sigmas.tolist()
        self.timesteps = timesteps[::-1].astype(int).tolist()
        return self.timesteps

    def scale_model_input(self, sample: MyTensor, step_index: int) -> MyTensor:
        sigma = self.sigmas[step_index]
        sample_array = sample.get_tensor().numpy()
        scaled_array = sample_array / np.sqrt(sigma**2 + 1)
        self.is_scale_input_called = True
        return MyTensor(ort.OrtValue.ortvalue_from_numpy(scaled_array))

    def step(self, model_output: MyTensor, step_index: int, sample: MyTensor) -> MyTensor:
        sample_array = sample.get_tensor().numpy()
        output_array = model_output.get_tensor().numpy()
        sigma = self.sigmas[step_index]

        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample_array - sigma * output_array
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = output_array * np.sqrt(-sigma / (sigma**2 + 1)) + sample_array / (sigma**2 + 1)
        
        sigma_from = self.sigmas[step_index]
        sigma_to = self.sigmas[step_index + 1]
        sigma_up = np.sqrt((sigma_to**2 * (sigma_from**2 - sigma_to**2)) / sigma_from**2)
        sigma_down = np.sqrt(sigma_to**2 - sigma_up**2)

        derivative = (sample_array - pred_original_sample) / sigma
        dt = sigma_down - sigma
        prev_sample = sample_array + derivative * dt

        noise = self.random.normal(size=prev_sample.shape)
        prev_sample += noise * sigma_up

        return MyTensor(ort.OrtValue.ortvalue_from_numpy(prev_sample))

    def betas_for_alpha_bar(self, num_diffusion_timesteps: int, max_beta: float) -> List[float]:
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - self.alpha_bar(t2) / self.alpha_bar(t1), max_beta))
        return betas

    def alpha_bar(self, time_step: float) -> float:
        return np.cos((time_step + 0.008) / 1.008 * np.pi / 2)**2

    def get_init_noise_sigma(self) -> float:
        return self.init_noise_sigma
