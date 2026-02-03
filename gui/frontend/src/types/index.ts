// Training Configuration Types
export interface OptimizerConfig {
  optimizer_type: 'AdamW8bit' | 'AdamW' | 'Lion' | 'SGDNesterov' | 'DAdaptation' | 'AdaFactor'
  learning_rate: number
  lr_scheduler: 'linear' | 'cosine' | 'cosine_with_restarts' | 'polynomial' | 'constant' | 'constant_with_warmup'
  lr_warmup_steps: number
  lr_scheduler_num_cycles: number
  lr_scheduler_power: number
  max_grad_norm: number
}

export interface NetworkConfig {
  network_type: 'lora' | 'lokr'
  network_module: string
  network_dim: number
  network_alpha: number
  network_train_text_encoder_only: boolean
  network_train_unet_only: boolean
  network_weights?: string
  network_dropout?: number
  network_rank_dropout?: number
  network_module_dropout?: number
}

export interface NoiseConfig {
  noise_offset: number
  multires_noise_iterations: number
  multires_noise_discount: number
  adaptive_noise_scale: number
}

export interface CaptionConfig {
  caption_extension: string
  shuffle_caption: boolean
  keep_tokens: number
  max_token_length: number
  caption_dropout_rate: number
  caption_dropout_every_n_epochs: number
  caption_tag_dropout_rate: number
}

export interface TrainingConfig {
  // Model settings
  pretrained_model_name_or_path: string
  vae?: string
  
  // Dataset settings
  train_data_dir: string
  reg_data_dir?: string
  output_dir: string
  output_name: string
  
  // Training settings
  resolution: number
  enable_bucket: boolean
  min_bucket_reso: number
  max_bucket_reso: number
  bucket_reso_steps: number
  
  // Training parameters
  train_batch_size: number
  num_train_epochs: number
  max_train_steps?: number
  max_train_epochs?: number
  save_every_n_epochs: number
  save_every_n_steps?: number
  save_n_epoch_ratio?: number
  save_last_n_epochs?: number
  save_last_n_epochs_state?: number
  save_state?: boolean
  resume?: string
  
  // Validation
  validation_epochs?: number
  validation_steps?: number
  
  // Mixed precision
  mixed_precision: 'no' | 'fp16' | 'bf16' | 'fp8'
  full_fp16?: boolean
  full_bf16?: boolean
  fp8_base?: boolean
  
  // xformers & memory
  enable_xformers: boolean
  gradient_checkpointing: boolean
  gradient_accumulation_steps: number
  
  // Seed
  seed: number
  
  // Logging
  logging_dir: string
  log_with: 'tensorboard' | 'wandb' | 'all'
  log_prefix?: string
  
  // Other settings
  clip_skip: number
  max_token_length: number
  
  // Sub-configs
  optimizer: OptimizerConfig
  network: NetworkConfig
  noise: NoiseConfig
  caption: CaptionConfig
  
  // Additional options
  [key: string]: any
}

// Training Status Types
export type TrainingStatus = 'idle' | 'running' | 'paused' | 'stopped' | 'completed' | 'error'

export interface TrainingMetrics {
  loss: number[]
  learning_rate: number[]
  epoch: number[]
  step: number[]
  timestamp: string[]
}

export interface LogEntry {
  timestamp: string
  level: 'debug' | 'info' | 'warning' | 'error' | 'success'
  message: string
}

export interface CheckpointInfo {
  path: string
  name: string
  size: number
  created: string
  epoch: number
  step: number
  loss?: number
}

// System Types
export interface SystemInfo {
  platform: string
  python_version: string
  cuda_available: boolean
  cuda_version?: string
  torch_version: string
  diffusers_version: string
}

export interface GPUInfo {
  index: number
  name: string
  total_memory: number
  free_memory: number
  used_memory: number
  utilization: number
}

export interface DiskInfo {
  path: string
  total: number
  free: number
  used: number
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  message?: string
  error?: string
}

// Form Types
export interface SelectOption {
  label: string
  value: string | number | boolean
}

export interface ValidationResult {
  valid: boolean
  errors: string[]
}
