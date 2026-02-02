<template>
  <div class="home-view">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="welcome-card">
          <div class="welcome-content">
            <el-icon :size="64" class="welcome-icon"><Cpu /></el-icon>
            <h1>欢迎使用 Anima LoRA Trainer</h1>
            <p class="subtitle">专业的 Stable Diffusion LoRA 模型训练工具</p>
            
            <div class="quick-actions">
              <el-button type="primary" size="large" @click="$router.push('/config')">
                <el-icon><Setting /></el-icon>
                开始配置训练
              </el-button>
              <el-button type="success" size="large" @click="$router.push('/train')" :disabled="!trainStore.canStart">
                <el-icon><VideoPlay /></el-icon>
                启动训练
              </el-button>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="mt-4">
      <el-col :xs="24" :sm="12" :md="8">
        <el-card class="stat-card">
          <div class="stat-item">
            <el-icon :size="40" class="stat-icon"><Collection /></el-icon>
            <div class="stat-info">
              <div class="stat-value">{{ systemStore.modelsCount }}</div>
              <div class="stat-label">可用模型</div>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <el-col :xs="24" :sm="12" :md="8">
        <el-card class="stat-card">
          <div class="stat-item">
            <el-icon :size="40" class="stat-icon"><DataAnalysis /></el-icon>
            <div class="stat-info">
              <div class="stat-value">{{ trainStore.checkpoints.length }}</div>
              <div class="stat-label">训练产出</div>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <el-col :xs="24" :sm="12" :md="8">
        <el-card class="stat-card">
          <div class="stat-item">
            <el-icon :size="40" class="stat-icon"><Timer /></el-icon>
            <div class="stat-info">
              <div class="stat-value">{{ trainStore.duration || '--:--:--' }}</div>
              <div class="stat-label">训练时长</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="mt-4">
      <el-col :xs="24" :md="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>系统信息</span>
              <el-button text @click="refreshSystemInfo">
                <el-icon><Refresh /></el-icon>
              </el-button>
            </div>
          </template>
          <el-descriptions :column="1" border>
            <el-descriptions-item label="平台">{{ systemInfo.platform }}</el-descriptions-item>
            <el-descriptions-item label="Python">{{ systemInfo.python_version }}</el-descriptions-item>
            <el-descriptions-item label="CUDA">
              <el-tag :type="systemInfo.cuda_available ? 'success' : 'danger'">
                {{ systemInfo.cuda_available ? systemInfo.cuda_version : '不可用' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="PyTorch">{{ systemInfo.torch_version }}</el-descriptions-item>
            <el-descriptions-item label="Diffusers">{{ systemInfo.diffusers_version }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
      
      <el-col :xs="24" :md="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>GPU 状态</span>
              <el-tag :type="gpuInfo.length > 0 ? 'success' : 'danger'">
                {{ gpuInfo.length > 0 ? `检测到 ${gpuInfo.length} 个 GPU` : '未检测到 GPU' }}
              </el-tag>
            </div>
          </template>
          <div v-if="gpuInfo.length > 0" class="gpu-list">
            <div v-for="(gpu, index) in gpuInfo" :key="index" class="gpu-item">
              <div class="gpu-header">
                <span class="gpu-name">{{ gpu.name }}</span>
                <span class="gpu-mem">{{ formatBytes(gpu.used_memory) }} / {{ formatBytes(gpu.total_memory) }}</span>
              </div>
              <el-progress 
                :percentage="Math.round((gpu.used_memory / gpu.total_memory) * 100)" 
                :status="getGPUStatus(gpu)"
              />
            </div>
          </div>
          <el-empty v-else description="未检测到 GPU 设备" />
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="mt-4">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>最近日志</span>
              <el-button text @click="$router.push('/train')">
                查看全部
              </el-button>
            </div>
          </template>
          <el-timeline v-if="recentLogs.length > 0">
            <el-timeline-item
              v-for="log in recentLogs"
              :key="log.timestamp"
              :type="getLogType(log.level)"
              :timestamp="formatTime(log.timestamp)"
            >
              {{ log.message }}
            </el-timeline-item>
          </el-timeline>
          <el-empty v-else description="暂无日志" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useTrainStore } from '@/stores/train'
import { api } from '@/api/client'
import type { SystemInfo, GPUInfo, LogEntry } from '@/types'
import {
  Cpu,
  Setting,
  VideoPlay,
  Collection,
  DataAnalysis,
  Timer,
  Refresh
} from '@element-plus/icons-vue'

const trainStore = useTrainStore()

const systemInfo = ref<SystemInfo>({
  platform: '',
  python_version: '',
  cuda_available: false,
  torch_version: '',
  diffusers_version: ''
})

const gpuInfo = ref<GPUInfo[]>([])

const recentLogs = computed(() => trainStore.logs.slice(-5))

const systemStore = ref({
  modelsCount: 0
})

const refreshSystemInfo = async () => {
  try {
    const infoResponse = await api.system.info()
    const info = infoResponse.data || {}
    systemInfo.value = {
      platform: info.platform || '',
      python_version: info.python_version || '',
      cuda_available: Boolean(info.cuda_version),
      cuda_version: info.cuda_version || '',
      torch_version: info.torch_version || '未知',
      diffusers_version: info.diffusers_version || '未知'
    }

    const gpuResponse = await api.system.gpuStatus()
    const gpus = gpuResponse.data?.gpus || []
    gpuInfo.value = gpus.map((gpu: any, index: number) => {
      const totalBytes = (gpu.total_gb || 0) * 1024 * 1024 * 1024
      const usedBytes = (gpu.allocated_gb || 0) * 1024 * 1024 * 1024
      return {
        index: gpu.id ?? index,
        name: gpu.name || 'Unknown GPU',
        total_memory: totalBytes,
        free_memory: Math.max(totalBytes - usedBytes, 0),
        used_memory: usedBytes,
        utilization: gpu.utilization || 0
      }
    })

    const checkpointsResponse = await api.checkpoint.list()
    systemStore.value.modelsCount = checkpointsResponse.data?.checkpoints?.length || 0
  } catch (error) {
    console.error('Failed to load system info:', error)
  }
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const getGPUStatus = (gpu: GPUInfo): string => {
  const usage = (gpu.used_memory / gpu.total_memory) * 100
  if (usage > 90) return 'exception'
  if (usage > 70) return 'warning'
  return ''
}

const getLogType = (level: LogEntry['level']): string => {
  switch (level) {
    case 'error': return 'danger'
    case 'warning': return 'warning'
    case 'success': return 'success'
    default: return 'info'
  }
}

const formatTime = (timestamp: string): string => {
  return new Date(timestamp).toLocaleString('zh-CN')
}

onMounted(() => {
  refreshSystemInfo()
})
</script>

<style scoped>
.home-view {
  padding-bottom: 20px;
}

.welcome-card {
  text-align: center;
  padding: 40px 20px;
}

.welcome-content {
  max-width: 600px;
  margin: 0 auto;
}

.welcome-icon {
  color: var(--el-color-primary);
  margin-bottom: 20px;
}

.welcome-content h1 {
  margin: 0 0 10px 0;
  font-size: 28px;
  color: var(--el-text-color-primary);
}

.subtitle {
  color: var(--el-text-color-secondary);
  margin-bottom: 30px;
}

.quick-actions {
  display: flex;
  gap: 16px;
  justify-content: center;
  flex-wrap: wrap;
}

.stat-card {
  height: 100%;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px;
}

.stat-icon {
  color: var(--el-color-primary);
}

.stat-info {
  flex: 1;
}

.stat-value {
  font-size: 24px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.stat-label {
  font-size: 14px;
  color: var(--el-text-color-secondary);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.gpu-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.gpu-item {
  padding: 12px;
  background: var(--el-fill-color-light);
  border-radius: 8px;
}

.gpu-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
}

.gpu-name {
  font-weight: 500;
}

.gpu-mem {
  color: var(--el-text-color-secondary);
}

.mt-4 {
  margin-top: 16px;
}

.ml-2 {
  margin-left: 8px;
}
</style>
