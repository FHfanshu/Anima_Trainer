import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { TrainingConfig, OptimizerConfig, NetworkConfig, NoiseConfig, CaptionConfig } from '@/types'

const defaultOptimizer: OptimizerConfig = {
  optimizer_type: 'AdamW8bit',
  learning_rate: 0.0001,
  lr_scheduler: 'cosine_with_restarts',
  lr_warmup_steps: 100,
  lr_scheduler_num_cycles: 3,
  lr_scheduler_power: 1.0,
  max_grad_norm: 1.0,
}

const defaultNetwork: NetworkConfig = {
  network_module: 'networks.lora',
  network_dim: 32,
  network_alpha: 16,
  network_train_text_encoder_only: false,
  network_train_unet_only: false,
}

const defaultNoise: NoiseConfig = {
  noise_offset: 0.0,
  multires_noise_iterations: 0,
  multires_noise_discount: 0.3,
  adaptive_noise_scale: 0.0,
}

const defaultCaption: CaptionConfig = {
  caption_extension: '.txt',
  shuffle_caption: true,
  keep_tokens: 0,
  max_token_length: 225,
  caption_dropout_rate: 0.0,
  caption_dropout_every_n_epochs: 0,
  caption_tag_dropout_rate: 0.0,
}

export const defaultTrainingConfig: TrainingConfig = {
  // Model settings
  pretrained_model_name_or_path: '',
  vae: '',
  
  // Dataset settings
  train_data_dir: '',
  reg_data_dir: '',
  output_dir: './outputs',
  output_name: 'anima_lora',
  
  // Training settings
  resolution: 512,
  enable_bucket: true,
  min_bucket_reso: 256,
  max_bucket_reso: 1024,
  bucket_reso_steps: 64,
  
  // Training parameters
  train_batch_size: 1,
  num_train_epochs: 10,
  max_train_steps: 0,
  max_train_epochs: 0,
  save_every_n_epochs: 1,
  save_every_n_steps: 0,
  save_n_epoch_ratio: 0,
  save_last_n_epochs: 0,
  save_last_n_epochs_state: 0,
  save_state: false,
  resume: '',
  
  // Validation
  validation_epochs: 0,
  validation_steps: 0,
  
  // Mixed precision
  mixed_precision: 'fp16',
  full_fp16: false,
  full_bf16: false,
  fp8_base: false,
  
  // xformers
  enable_xformers: true,
  gradient_checkpointing: true,
  gradient_accumulation_steps: 1,
  
  // Seed
  seed: 42,
  
  // Logging
  logging_dir: './logs',
  log_with: 'tensorboard',
  log_prefix: '',
  
  // Other settings
  clip_skip: 2,
  max_token_length: 225,
  
  // Configs
  optimizer: { ...defaultOptimizer },
  network: { ...defaultNetwork },
  noise: { ...defaultNoise },
  caption: { ...defaultCaption },
}

export const useConfigStore = defineStore('config', () => {
  // State
  const config = ref<TrainingConfig>({ ...defaultTrainingConfig })
  const savedConfigs = ref<string[]>([])
  const isLoading = ref(false)
  const lastSaved = ref<Date | null>(null)
  const hasChanges = ref(false)

  // Actions
  const updateConfig = (path: string, value: any) => {
    const keys = path.split('.')
    let target: any = config.value
    
    for (let i = 0; i < keys.length - 1; i++) {
      target = target[keys[i]]
    }
    
    target[keys[keys.length - 1]] = value
    hasChanges.value = true
  }

  const resetConfig = () => {
    config.value = { ...defaultTrainingConfig }
    hasChanges.value = true
  }

  const saveConfig = async (name?: string) => {
    // This would save to backend/localStorage
    lastSaved.value = new Date()
    hasChanges.value = false
    if (name && !savedConfigs.value.includes(name)) {
      savedConfigs.value.push(name)
    }
  }

  const loadConfig = async (name: string) => {
    isLoading.value = true
    try {
      // This would load from backend/localStorage
      // For now, just a placeholder
    } finally {
      isLoading.value = false
    }
  }

  const validateConfig = (): { valid: boolean; errors: string[] } => {
    const errors: string[] = []
    
    if (!config.value.pretrained_model_name_or_path) {
      errors.push('预训练模型路径不能为空')
    }
    if (!config.value.train_data_dir) {
      errors.push('训练数据目录不能为空')
    }
    if (!config.value.output_dir) {
      errors.push('输出目录不能为空')
    }
    if (config.value.resolution < 128 || config.value.resolution > 2048) {
      errors.push('分辨率应在 128-2048 之间')
    }
    if (config.value.train_batch_size < 1) {
      errors.push('批次大小至少为 1')
    }
    if (config.value.optimizer.learning_rate <= 0) {
      errors.push('学习率必须大于 0')
    }

    return { valid: errors.length === 0, errors }
  }

  // Getters
  const configJSON = computed(() => JSON.stringify(config.value, null, 2))
  const isValid = computed(() => validateConfig().valid)
  const validationErrors = computed(() => validateConfig().errors)

  return {
    config,
    savedConfigs,
    isLoading,
    lastSaved,
    hasChanges,
    updateConfig,
    resetConfig,
    saveConfig,
    loadConfig,
    validateConfig,
    configJSON,
    isValid,
    validationErrors
  }
})
