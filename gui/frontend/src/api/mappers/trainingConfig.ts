type AnyRecord = Record<string, any>

const asNumberOrUndefined = (value: any): number | undefined => {
  if (value === null || value === undefined) return undefined
  if (typeof value === 'number' && Number.isFinite(value)) return value
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

const asStringOrUndefined = (value: any): string | undefined => {
  if (value === null || value === undefined) return undefined
  return String(value)
}

const isPlainObject = (value: any): value is AnyRecord =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

export const toBackendTrainingConfig = (uiConfig: any): AnyRecord => {
  // Backend/train.py expects a mostly-flat schema (see TrainingConfig dataclass).
  // The frontend UI currently uses a kohya-style schema; normalize here to keep the API stable.
  const cfg: AnyRecord = isPlainObject(uiConfig) ? { ...uiConfig } : {}

  // Dataset path
  if (!cfg.data_root && cfg.train_data_dir) cfg.data_root = cfg.train_data_dir

  // Resume
  if (!cfg.resume_from_checkpoint && cfg.resume) cfg.resume_from_checkpoint = cfg.resume

  // Network / LoRA
  if (!cfg.lora_type && cfg.network?.network_type) cfg.lora_type = cfg.network.network_type
  if (!cfg.lora_rank && asNumberOrUndefined(cfg.network?.network_dim) !== undefined) cfg.lora_rank = cfg.network.network_dim
  if (!cfg.lora_alpha && asNumberOrUndefined(cfg.network?.network_alpha) !== undefined) cfg.lora_alpha = cfg.network.network_alpha
  if (cfg.lora_dropout === undefined && cfg.network?.network_dropout !== undefined) cfg.lora_dropout = cfg.network.network_dropout

  // Optimizer block (frontend) -> flat (backend)
  if (isPlainObject(cfg.optimizer)) {
    const optimizerObj = cfg.optimizer
    const optType = asStringOrUndefined(optimizerObj.optimizer_type)?.toLowerCase()
    if (optType) {
      cfg.optimizer =
        optType === 'adamw8bit'
          ? 'adamw8bit'
          : optType === 'adamw'
            ? 'adamw'
            : 'adamw'
    }

    if (cfg.learning_rate === undefined && asNumberOrUndefined(optimizerObj.learning_rate) !== undefined) {
      cfg.learning_rate = optimizerObj.learning_rate
    }
    if (!cfg.lr_scheduler && optimizerObj.lr_scheduler) cfg.lr_scheduler = optimizerObj.lr_scheduler
    if (cfg.lr_warmup_steps === undefined && asNumberOrUndefined(optimizerObj.lr_warmup_steps) !== undefined) {
      cfg.lr_warmup_steps = optimizerObj.lr_warmup_steps
    }
    if (cfg.lr_num_cycles === undefined && asNumberOrUndefined(optimizerObj.lr_scheduler_num_cycles) !== undefined) {
      cfg.lr_num_cycles = optimizerObj.lr_scheduler_num_cycles
    }
    if (cfg.lr_power === undefined && asNumberOrUndefined(optimizerObj.lr_scheduler_power) !== undefined) {
      cfg.lr_power = optimizerObj.lr_scheduler_power
    }
    if (cfg.max_grad_norm === undefined && asNumberOrUndefined(optimizerObj.max_grad_norm) !== undefined) {
      cfg.max_grad_norm = optimizerObj.max_grad_norm
    }
  }

  // Backend only supports 'no' | 'fp16' | 'bf16'
  if (cfg.mixed_precision && !['no', 'fp16', 'bf16'].includes(String(cfg.mixed_precision))) {
    cfg.mixed_precision = 'bf16'
  }

  // Backend defaults for these are fine; keep UI-provided values if present.
  if (cfg.center_crop === undefined) cfg.center_crop = true
  if (cfg.random_flip === undefined) cfg.random_flip = true

  // Tag dropout: use the most similar frontend field if available
  if (cfg.tag_dropout === undefined && asNumberOrUndefined(cfg.caption?.caption_tag_dropout_rate) !== undefined) {
    cfg.tag_dropout = cfg.caption.caption_tag_dropout_rate
  }

  // Backend expects output_dir, not output_name. Keep output_name as extra (harmless).
  return cfg
}
