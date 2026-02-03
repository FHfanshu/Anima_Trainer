<template>
  <div class="config-view">
    <el-page-header @back="$router.push('/')" title="训练配置" />
    
    <el-alert
      v-if="!configStore.isValid"
      :title="'配置错误: ' + configStore.validationErrors.join(', ')"
      type="error"
      show-icon
      class="mt-4"
    />

    <el-form
      ref="formRef"
      :model="configStore.config"
      label-position="top"
      class="config-form mt-4"
    >
      <el-tabs type="border-card" v-model="activeTab">
        <!-- 模型设置 -->
        <el-tab-pane label="模型设置" name="model">
          <el-row :gutter="20">
            <el-col :xs="24" :md="12">
              <el-form-item label="预训练模型路径">
                <el-input
                  v-model="configStore.config.pretrained_model_name_or_path"
                  placeholder="输入模型路径或 Hugging Face ID"
                  clearable
                >
                  <template #append>
                    <el-button @click="browseModel">浏览</el-button>
                  </template>
                </el-input>
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="12">
              <el-form-item label="VAE 路径 (可选)">
                <el-input
                  v-model="configStore.config.vae"
                  placeholder="输入 VAE 模型路径"
                  clearable
                >
                  <template #append>
                    <el-button @click="browseVAE">浏览</el-button>
                  </template>
                </el-input>
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="数据类型">
                <el-select v-model="configStore.config.mixed_precision" style="width: 100%">
                  <el-option label="FP16" value="fp16" />
                  <el-option label="BF16" value="bf16" />
                  <el-option label="FP32" value="no" />
                  <el-option label="FP8" value="fp8" />
                </el-select>
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item>
                <template #label>
                  <span>完整 FP16</span>
                  <el-tooltip content="使用完整的 FP16 精度（可能更快但不稳定）">
                    <el-icon class="ml-2"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </template>
                <el-switch v-model="configStore.config.full_fp16" />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item>
                <template #label>
                  <span>完整 BF16</span>
                </template>
                <el-switch v-model="configStore.config.full_bf16" />
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item>
                <template #label>
                  <span>启用 xFormers</span>
                </template>
                <el-switch v-model="configStore.config.enable_xformers" />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item>
                <template #label>
                  <span>梯度检查点</span>
                  <el-tooltip content="启用梯度检查点以节省显存">
                    <el-icon class="ml-2"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </template>
                <el-switch v-model="configStore.config.gradient_checkpointing" />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="梯度累积步数">
                <el-input-number 
                  v-model="configStore.config.gradient_accumulation_steps" 
                  :min="1" 
                  :max="32"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="CLIP 跳过层">
                <el-input-number 
                  v-model="configStore.config.clip_skip" 
                  :min="1" 
                  :max="12"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="最大 Token 长度">
                <el-input-number 
                  v-model="configStore.config.max_token_length" 
                  :min="75" 
                  :max="225"
                  :step="75"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="随机种子">
                <el-input-number 
                  v-model="configStore.config.seed" 
                  :min="-1" 
                  :max="2147483647"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </el-row>
        </el-tab-pane>

        <!-- 数据集设置 -->
        <el-tab-pane label="数据集设置" name="dataset">
          <el-row :gutter="20">
            <el-col :xs="24" :md="12">
              <el-form-item label="训练数据目录">
                <el-input
                  v-model="configStore.config.train_data_dir"
                  placeholder="选择训练图像文件夹"
                  clearable
                >
                  <template #append>
                    <el-button @click="browseTrainData">浏览</el-button>
                  </template>
                </el-input>
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="12">
              <el-form-item label="正则化数据目录 (可选)">
                <el-input
                  v-model="configStore.config.reg_data_dir"
                  placeholder="选择正则化图像文件夹"
                  clearable
                >
                  <template #append>
                    <el-button @click="browseRegData">浏览</el-button>
                  </template>
                </el-input>
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="12">
              <el-form-item label="输出目录">
                <el-input
                  v-model="configStore.config.output_dir"
                  placeholder="模型输出路径"
                  clearable
                >
                  <template #append>
                    <el-button @click="browseOutput">浏览</el-button>
                  </template>
                </el-input>
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="12">
              <el-form-item label="输出模型名称">
                <el-input
                  v-model="configStore.config.output_name"
                  placeholder="不包含扩展名的模型名称"
                  clearable
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-divider>图像桶设置</el-divider>

          <el-row :gutter="20">
            <el-col :xs="24" :md="6">
              <el-form-item>
                <template #label>
                  <span>启用图像桶</span>
                </template>
                <el-switch v-model="configStore.config.enable_bucket" />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="6">
              <el-form-item label="训练分辨率">
                <el-input-number 
                  v-model="configStore.config.resolution" 
                  :min="128" 
                  :max="2048"
                  :step="64"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="6">
              <el-form-item label="最小桶分辨率">
                <el-input-number 
                  v-model="configStore.config.min_bucket_reso" 
                  :min="128" 
                  :max="1024"
                  :step="64"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="6">
              <el-form-item label="最大桶分辨率">
                <el-input-number 
                  v-model="configStore.config.max_bucket_reso" 
                  :min="256" 
                  :max="2048"
                  :step="64"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-divider>标签设置</el-divider>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="标签文件扩展名">
                <el-select v-model="configStore.config.caption.caption_extension" style="width: 100%">
                  <el-option label=".txt" value=".txt" />
                  <el-option label=".caption" value=".caption" />
                </el-select>
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item>
                <template #label>
                  <span>打乱标签顺序</span>
                </template>
                <el-switch v-model="configStore.config.caption.shuffle_caption" />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="保留前 N 个标签">
                <el-input-number 
                  v-model="configStore.config.caption.keep_tokens" 
                  :min="0" 
                  :max="10"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="标签丢弃率">
                <el-slider 
                  v-model="configStore.config.caption.caption_dropout_rate" 
                  :min="0" 
                  :max="1"
                  :step="0.05"
                  show-input
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="每 N Epoch 丢弃标签">
                <el-input-number 
                  v-model="configStore.config.caption.caption_dropout_every_n_epochs" 
                  :min="0" 
                  :max="100"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="标签词丢弃率">
                <el-slider 
                  v-model="configStore.config.caption.caption_tag_dropout_rate" 
                  :min="0" 
                  :max="1"
                  :step="0.05"
                  show-input
                />
              </el-form-item>
            </el-col>
          </el-row>
        </el-tab-pane>

        <!-- 训练参数 -->
        <el-tab-pane label="训练参数" name="training">
          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="训练轮数 (Epochs)">
                <el-input-number 
                  v-model="configStore.config.num_train_epochs" 
                  :min="1" 
                  :max="1000"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="批次大小">
                <el-input-number 
                  v-model="configStore.config.train_batch_size" 
                  :min="1" 
                  :max="32"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="最大训练步数 (0=自动)">
                <el-input-number 
                  v-model="configStore.config.max_train_steps" 
                  :min="0" 
                  :max="1000000"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-divider>保存设置</el-divider>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="每 N Epoch 保存">
                <el-input-number 
                  v-model="configStore.config.save_every_n_epochs" 
                  :min="1" 
                  :max="100"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="每 N 步保存">
                <el-input-number 
                  v-model="configStore.config.save_every_n_steps" 
                  :min="0" 
                  :max="100000"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item>
                <template #label>
                  <span>保存训练状态</span>
                </template>
                <el-switch v-model="configStore.config.save_state" />
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="仅保留最后 N 个模型">
                <el-input-number 
                  v-model="configStore.config.save_last_n_epochs" 
                  :min="0" 
                  :max="100"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="仅保留最后 N 个状态">
                <el-input-number 
                  v-model="configStore.config.save_last_n_epochs_state" 
                  :min="0" 
                  :max="100"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="恢复训练路径">
                <el-input
                  v-model="configStore.config.resume"
                  placeholder="从状态文件恢复"
                  clearable
                />
              </el-form-item>
            </el-col>
          </el-row>
        </el-tab-pane>

        <!-- 优化器设置 -->
        <el-tab-pane label="优化器设置" name="optimizer">
          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="优化器类型">
                <el-select v-model="configStore.config.optimizer.optimizer_type" style="width: 100%">
                  <el-option label="AdamW 8bit" value="AdamW8bit" />
                  <el-option label="AdamW" value="AdamW" />
                  <el-option label="Lion" value="Lion" />
                  <el-option label="SGD Nesterov" value="SGDNesterov" />
                  <el-option label="DAdaptation" value="DAdaptation" />
                  <el-option label="AdaFactor" value="AdaFactor" />
                </el-select>
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="学习率">
                <el-input-number 
                  v-model="configStore.config.optimizer.learning_rate" 
                  :min="0.000001" 
                  :max="0.1"
                  :step="0.0001"
                  :precision="6"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="学习率调度器">
                <el-select v-model="configStore.config.optimizer.lr_scheduler" style="width: 100%">
                  <el-option label="线性" value="linear" />
                  <el-option label="余弦" value="cosine" />
                  <el-option label="余弦重启" value="cosine_with_restarts" />
                  <el-option label="多项式" value="polynomial" />
                  <el-option label="常数" value="constant" />
                  <el-option label="常数预热" value="constant_with_warmup" />
                </el-select>
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="预热步数">
                <el-input-number 
                  v-model="configStore.config.optimizer.lr_warmup_steps" 
                  :min="0" 
                  :max="10000"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="调度器周期数">
                <el-input-number 
                  v-model="configStore.config.optimizer.lr_scheduler_num_cycles" 
                  :min="1" 
                  :max="20"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="调度器幂次">
                <el-input-number 
                  v-model="configStore.config.optimizer.lr_scheduler_power" 
                  :min="0" 
                  :max="10"
                  :step="0.1"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="最大梯度范数">
                <el-input-number 
                  v-model="configStore.config.optimizer.max_grad_norm" 
                  :min="0" 
                  :max="10"
                  :step="0.1"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </el-row>
        </el-tab-pane>

        <!-- 网络设置 -->
        <el-tab-pane label="网络设置" name="network">
          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="网络类型">
                <el-select 
                  v-model="configStore.config.network.network_type" 
                  style="width: 100%"
                  @change="onNetworkTypeChange"
                >
                  <el-option label="LoRA" value="lora" />
                  <el-option label="LoKR (LyCORIS)" value="lokr" />
                </el-select>
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="网络维度 (Dim)">
                <el-input-number 
                  v-model="configStore.config.network.network_dim" 
                  :min="1" 
                  :max="1024"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="网络 Alpha">
                <el-input-number 
                  v-model="configStore.config.network.network_alpha" 
                  :min="1" 
                  :max="1024"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="12">
              <el-form-item label="网络权重 (可选)">
                <el-input
                  v-model="configStore.config.network.network_weights"
                  placeholder="从已有 LoRA/LoKR 继续训练"
                  clearable
                >
                  <template #append>
                    <el-button @click="browseNetworkWeights">浏览</el-button>
                  </template>
                </el-input>
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="12">
              <el-form-item label="网络 Dropout">
                <el-slider 
                  v-model="configStore.config.network.network_dropout" 
                  :min="0" 
                  :max="1"
                  :step="0.01"
                  show-input
                />
              </el-form-item>
            </el-col>
          </el-row>
        </el-tab-pane>

        <!-- 噪声设置 -->
        <el-tab-pane label="噪声设置" name="noise">
          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="噪声偏移">
                <el-slider 
                  v-model="configStore.config.noise.noise_offset" 
                  :min="0" 
                  :max="1"
                  :step="0.01"
                  show-input
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="自适应噪声缩放">
                <el-slider 
                  v-model="configStore.config.noise.adaptive_noise_scale" 
                  :min="0" 
                  :max="1"
                  :step="0.01"
                  show-input
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="8">
              <el-form-item label="多分辨率噪声迭代">
                <el-input-number 
                  v-model="configStore.config.noise.multires_noise_iterations" 
                  :min="0" 
                  :max="10"
                  style="width: 100%"
                />
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="8">
              <el-form-item label="多分辨率噪声折扣">
                <el-slider 
                  v-model="configStore.config.noise.multires_noise_discount" 
                  :min="0" 
                  :max="1"
                  :step="0.01"
                  show-input
                />
              </el-form-item>
            </el-col>
          </el-row>
        </el-tab-pane>

        <!-- 日志设置 -->
        <el-tab-pane label="日志设置" name="logging">
          <el-row :gutter="20">
            <el-col :xs="24" :md="12">
              <el-form-item label="日志目录">
                <el-input
                  v-model="configStore.config.logging_dir"
                  placeholder="日志文件保存路径"
                  clearable
                >
                  <template #append>
                    <el-button @click="browseLogDir">浏览</el-button>
                  </template>
                </el-input>
              </el-form-item>
            </el-col>
            
            <el-col :xs="24" :md="12">
              <el-form-item label="日志工具">
                <el-select v-model="configStore.config.log_with" style="width: 100%">
                  <el-option label="TensorBoard" value="tensorboard" />
                  <el-option label="Weights & Biases" value="wandb" />
                  <el-option label="全部启用" value="all" />
                </el-select>
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="20">
            <el-col :xs="24" :md="12">
              <el-form-item label="日志前缀">
                <el-input
                  v-model="configStore.config.log_prefix"
                  placeholder="日志文件前缀"
                  clearable
                />
              </el-form-item>
            </el-col>
          </el-row>
        </el-tab-pane>
      </el-tabs>

      <!-- Action Buttons -->
      <el-row :gutter="20" class="action-bar">
        <el-col :span="24">
          <el-button type="primary" @click="saveConfig" :loading="configStore.isLoading">
            <el-icon><Check /></el-icon>
            保存配置
          </el-button>
          <el-button @click="resetConfig">
            <el-icon><RefreshLeft /></el-icon>
            重置配置
          </el-button>
          <el-button type="success" @click="startTraining" :disabled="!configStore.isValid || trainStore.isRunning">
            <el-icon><VideoPlay /></el-icon>
            开始训练
          </el-button>
        </el-col>
      </el-row>
    </el-form>

    <!-- Configuration Preview -->
    <el-card class="config-preview mt-4">
      <template #header>
        <span>配置预览 (JSON)</span>
      </template>
      <pre class="config-json">{{ configStore.configJSON }}</pre>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useConfigStore } from '@/stores/config'
import { useTrainStore } from '@/stores/train'
import { api } from '@/api/client'
import {
  QuestionFilled,
  Check,
  RefreshLeft,
  VideoPlay
} from '@element-plus/icons-vue'

const router = useRouter()
const configStore = useConfigStore()
const trainStore = useTrainStore()
const formRef = ref()
const activeTab = ref('model')

// 网络类型改变时自动更新 network_module
const onNetworkTypeChange = (type: 'lora' | 'lokr') => {
  configStore.config.network.network_type = type
  if (type === 'lora') {
    configStore.config.network.network_module = 'networks.lora'
  } else if (type === 'lokr') {
    configStore.config.network.network_module = 'lycoris.kohya'
  }
}

const browseModel = async () => {
  try {
    const resp = await api.system.pickFile('model')
    const path = resp.data?.path
    if (path) configStore.config.pretrained_model_name_or_path = path
  } catch {
    ElMessage.error('打开模型选择对话框失败')
  }
}

const browseVAE = async () => {
  try {
    const resp = await api.system.pickFile('model')
    const path = resp.data?.path
    if (path) configStore.config.vae = path
  } catch {
    ElMessage.error('打开 VAE 选择对话框失败')
  }
}

const browseTrainData = async () => {
  try {
    const resp = await api.system.pickFolder()
    const path = resp.data?.path
    if (path) configStore.config.train_data_dir = path
  } catch {
    ElMessage.error('打开训练数据选择对话框失败')
  }
}

const browseRegData = async () => {
  try {
    const resp = await api.system.pickFolder()
    const path = resp.data?.path
    if (path) configStore.config.reg_data_dir = path
  } catch {
    ElMessage.error('打开正则化数据选择对话框失败')
  }
}

const browseOutput = async () => {
  try {
    const resp = await api.system.pickFolder()
    const path = resp.data?.path
    if (path) configStore.config.output_dir = path
  } catch {
    ElMessage.error('打开输出目录选择对话框失败')
  }
}

const browseNetworkWeights = async () => {
  try {
    const resp = await api.system.pickFile('model')
    const path = resp.data?.path
    if (path) configStore.config.network.network_weights = path
  } catch {
    ElMessage.error('打开网络权重选择对话框失败')
  }
}

const browseLogDir = async () => {
  try {
    const resp = await api.system.pickFolder()
    const path = resp.data?.path
    if (path) configStore.config.logging_dir = path
  } catch {
    ElMessage.error('打开日志目录选择对话框失败')
  }
}

const saveConfig = async () => {
  try {
    await api.config.save({ config: configStore.config })
    configStore.saveConfig()
    ElMessage.success('配置已保存为 TOML')
  } catch (error) {
    ElMessage.error('保存配置失败')
  }
}

const resetConfig = () => {
  configStore.resetConfig()
  ElMessage.success('配置已重置为默认值')
}

const startTraining = async () => {
  if (!configStore.isValid) {
    ElMessage.error('请修正配置错误后再开始训练')
    return
  }
  
  try {
    await api.train.start(configStore.config)
    trainStore.startTraining()
    ElMessage.success('训练已启动')
    router.push('/train')
  } catch (error) {
    ElMessage.error('启动训练失败')
  }
}
</script>

<style scoped>
.config-view {
  max-width: 1200px;
  margin: 0 auto;
}

.config-form {
  background: var(--el-bg-color);
}

.action-bar {
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid var(--el-border-color);
}

.config-preview {
  background: var(--el-bg-color);
}

.config-json {
  margin: 0;
  padding: 16px;
  background: var(--el-fill-color-light);
  border-radius: 4px;
  font-family: monospace;
  font-size: 12px;
  max-height: 400px;
  overflow: auto;
}

.mt-4 {
  margin-top: 16px;
}

.ml-2 {
  margin-left: 8px;
  cursor: help;
}
</style>
