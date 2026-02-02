<template>
  <div class="train-view">
    <el-page-header @back="$router.push('/')" title="训练监控" />

    <!-- Training Status Bar -->
    <el-card class="status-bar mt-4">
      <el-row :gutter="20" align="middle">
        <el-col :xs="24" :md="6">
          <div class="status-item">
            <span class="status-label">状态</span>
            <el-tag :type="trainStore.statusType" size="large" effect="dark">
              {{ trainStore.statusText }}
            </el-tag>
          </div>
        </el-col>
        
        <el-col :xs="24" :md="6">
          <div class="status-item">
            <span class="status-label">进度</span>
            <div class="progress-info">
              <el-progress 
                :percentage="trainStore.progress" 
                :status="trainStore.status === 'error' ? 'exception' : ''"
              />
              <span class="progress-text">
                Epoch {{ trainStore.currentEpoch }}/{{ trainStore.totalEpochs }} - 
                Step {{ trainStore.currentStep }}/{{ trainStore.totalSteps }}
              </span>
            </div>
          </div>
        </el-col>
        
        <el-col :xs="24" :md="4">
          <div class="status-item">
            <span class="status-label">预计剩余</span>
            <el-text size="large">{{ trainStore.eta || '--:--:--' }}</el-text>
          </div>
        </el-col>
        
        <el-col :xs="24" :md="4">
          <div class="status-item">
            <span class="status-label">已运行</span>
            <el-text size="large">{{ trainStore.duration }}</el-text>
          </div>
        </el-col>
        
        <el-col :xs="24" :md="4">
          <div class="action-buttons">
            <el-button 
              type="primary" 
              @click="startTraining" 
              :disabled="!trainStore.canStart"
              :loading="isStarting"
            >
              <el-icon><VideoPlay /></el-icon>
            </el-button>
            <el-button 
              type="warning" 
              @click="pauseTraining" 
              :disabled="!trainStore.canPause"
            >
              <el-icon><VideoPause /></el-icon>
            </el-button>
            <el-button 
              type="success" 
              @click="resumeTraining" 
              :disabled="!trainStore.canResume"
            >
              <el-icon><RefreshRight /></el-icon>
            </el-button>
            <el-button 
              type="danger" 
              @click="stopTraining" 
              :disabled="!trainStore.canStop"
            >
              <el-icon><CircleClose /></el-icon>
            </el-button>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- Charts -->
    <el-row :gutter="20" class="mt-4">
      <el-col :xs="24" :md="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>损失曲线 (Loss)</span>
              <el-button-group>
                <el-button text @click="refreshCharts">
                  <el-icon><Refresh /></el-icon>
                </el-button>
              </el-button-group>
            </div>
          </template>
          <div ref="lossChartRef" class="chart-container"></div>
        </el-card>
      </el-col>
      
      <el-col :xs="24" :md="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>学习率 (Learning Rate)</span>
              <el-button-group>
                <el-button text @click="refreshCharts">
                  <el-icon><Refresh /></el-icon>
                </el-button>
              </el-button-group>
            </div>
          </template>
          <div ref="lrChartRef" class="chart-container"></div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Metrics Summary -->
    <el-row :gutter="20" class="mt-4">
      <el-col :xs="12" :md="6">
        <el-statistic title="当前 Loss" :value="currentLoss" :precision="4" />
      </el-col>
      <el-col :xs="12" :md="6">
        <el-statistic title="平均 Loss" :value="avgLoss" :precision="4" />
      </el-col>
      <el-col :xs="12" :md="6">
        <el-statistic title="最低 Loss" :value="minLoss" :precision="4" />
      </el-col>
      <el-col :xs="12" :md="6">
        <el-statistic title="当前学习率" :value="currentLR" :precision="8" />
      </el-col>
    </el-row>

    <!-- Logs -->
    <el-row :gutter="20" class="mt-4">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>训练日志</span>
              <div class="log-actions">
                <el-switch
                  v-model="autoScroll"
                  active-text="自动滚动"
                  inline-prompt
                />
                <el-button text @click="clearLogs">
                  <el-icon><Delete /></el-icon>
                </el-button>
                <el-button text @click="exportLogs">
                  <el-icon><Download /></el-icon>
                </el-button>
              </div>
            </div>
          </template>
          <div ref="logContainerRef" class="log-container">
            <div
              v-for="(log, index) in trainStore.logs"
              :key="index"
              :class="['log-line', `log-${log.level}`]"
            >
              <span class="log-time">{{ formatTime(log.timestamp) }}</span>
              <el-tag :type="getLogTagType(log.level)" size="small" class="log-level">
                {{ log.level.toUpperCase() }}
              </el-tag>
              <span class="log-message">{{ log.message }}</span>
            </div>
            <el-empty v-if="trainStore.logs.length === 0" description="暂无日志" />
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Checkpoints -->
    <el-row :gutter="20" class="mt-4">
      <el-col :span="24">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>已保存的模型</span>
              <el-button text @click="refreshCheckpoints">
                <el-icon><Refresh /></el-icon>
              </el-button>
            </div>
          </template>
          <el-table :data="trainStore.checkpoints" style="width: 100%">
            <el-table-column prop="name" label="名称" min-width="200" />
            <el-table-column prop="epoch" label="Epoch" width="80" />
            <el-table-column prop="step" label="Step" width="100" />
            <el-table-column prop="loss" label="Loss" width="120">
              <template #default="{ row }">
                <span v-if="row.loss">{{ row.loss.toFixed(4) }}</span>
                <span v-else>-</span>
              </template>
            </el-table-column>
            <el-table-column prop="size" label="大小" width="100">
              <template #default="{ row }">
                {{ formatBytes(row.size) }}
              </template>
            </el-table-column>
            <el-table-column prop="created" label="创建时间" width="180">
              <template #default="{ row }">
                {{ formatTime(row.created) }}
              </template>
            </el-table-column>
            <el-table-column label="操作" width="120" fixed="right">
              <template #default="{ row }">
                <el-button-group>
                  <el-button text @click="downloadCheckpoint(row.path)">
                    <el-icon><Download /></el-icon>
                  </el-button>
                  <el-button text type="danger" @click="deleteCheckpoint(row.path)">
                    <el-icon><Delete /></el-icon>
                  </el-button>
                </el-button-group>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useTrainStore } from '@/stores/train'
import { useConfigStore } from '@/stores/config'
import { api } from '@/api/client'
import * as echarts from 'echarts'
import type { LogEntry, CheckpointInfo } from '@/types'
import {
  VideoPlay,
  VideoPause,
  RefreshRight,
  CircleClose,
  Refresh,
  Delete,
  Download
} from '@element-plus/icons-vue'

const trainStore = useTrainStore()
const configStore = useConfigStore()

const lossChartRef = ref<HTMLElement>()
const lrChartRef = ref<HTMLElement>()
const logContainerRef = ref<HTMLElement>()

let lossChart: echarts.ECharts | null = null
let lrChart: echarts.ECharts | null = null

const isStarting = ref(false)
const autoScroll = ref(true)
let pollTimer: ReturnType<typeof setInterval>

// Computed metrics
const currentLoss = computed(() => {
  const losses = trainStore.metrics.loss
  return losses.length > 0 ? losses[losses.length - 1] : 0
})

const avgLoss = computed(() => {
  const losses = trainStore.metrics.loss
  if (losses.length === 0) return 0
  return losses.reduce((a, b) => a + b, 0) / losses.length
})

const minLoss = computed(() => {
  const losses = trainStore.metrics.loss
  if (losses.length === 0) return 0
  return Math.min(...losses)
})

const currentLR = computed(() => {
  const lrs = trainStore.metrics.learning_rate
  return lrs.length > 0 ? lrs[lrs.length - 1] : 0
})

// Chart initialization
const initCharts = () => {
  if (lossChartRef.value) {
    lossChart = echarts.init(lossChartRef.value)
    updateLossChart()
  }
  if (lrChartRef.value) {
    lrChart = echarts.init(lrChartRef.value)
    updateLRChart()
  }
}

const updateLossChart = () => {
  if (!lossChart) return
  
  const option: echarts.EChartsOption = {
    tooltip: {
      trigger: 'axis'
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: trainStore.metrics.step.map(s => s.toString()),
      name: 'Step'
    },
    yAxis: {
      type: 'value',
      name: 'Loss',
      min: 0
    },
    series: [{
      name: 'Loss',
      type: 'line',
      data: trainStore.metrics.loss,
      smooth: true,
      symbol: 'none',
      lineStyle: {
        color: '#409EFF'
      },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(64, 158, 255, 0.3)' },
          { offset: 1, color: 'rgba(64, 158, 255, 0.05)' }
        ])
      }
    }]
  }
  
  lossChart.setOption(option)
}

const updateLRChart = () => {
  if (!lrChart) return
  
  const option: echarts.EChartsOption = {
    tooltip: {
      trigger: 'axis'
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: trainStore.metrics.step.map(s => s.toString()),
      name: 'Step'
    },
    yAxis: {
      type: 'value',
      name: 'Learning Rate',
      scale: true
    },
    series: [{
      name: 'Learning Rate',
      type: 'line',
      data: trainStore.metrics.learning_rate,
      smooth: true,
      symbol: 'none',
      lineStyle: {
        color: '#67C23A'
      }
    }]
  }
  
  lrChart.setOption(option)
}

const refreshCharts = () => {
  updateLossChart()
  updateLRChart()
}

// Actions
const startTraining = async () => {
  if (!configStore.isValid) {
    ElMessage.error('请先完成训练配置')
    return
  }
  
  isStarting.value = true
  try {
    await api.train.start(configStore.config)
    trainStore.startTraining()
    ElMessage.success('训练已启动')
  } catch (error) {
    ElMessage.error('启动训练失败')
  } finally {
    isStarting.value = false
  }
}

const pauseTraining = async () => {
  ElMessage.info('后端暂不支持暂停训练')
}

const resumeTraining = async () => {
  ElMessage.info('后端暂不支持恢复训练')
}

const stopTraining = async () => {
  try {
    await ElMessageBox.confirm('确定要停止训练吗？', '确认', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    
    await api.train.stop()
    trainStore.stopTraining()
    ElMessage.success('训练已停止')
  } catch (error) {
    // User cancelled
  }
}

const clearLogs = () => {
  trainStore.clearLogs()
}

const exportLogs = () => {
  const logs = trainStore.logs.map(log => 
    `[${log.timestamp}] [${log.level.toUpperCase()}] ${log.message}`
  ).join('\n')
  
  const blob = new Blob([logs], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `training_logs_${new Date().toISOString().replace(/[:.]/g, '-')}.txt`
  a.click()
  URL.revokeObjectURL(url)
  
  ElMessage.success('日志已导出')
}

const refreshCheckpoints = async () => {
  try {
    const response = await api.checkpoint.list()
    const checkpoints = (response.data?.checkpoints || []).map((cp: any) => ({
      name: cp.name || '',
      path: cp.path || '',
      created: cp.created || '',
      size: cp.size || 0,
      epoch: cp.epoch || 0,
      step: cp.step || 0,
      loss: cp.loss
    }))
    checkpoints.forEach((cp: CheckpointInfo) => trainStore.addCheckpoint(cp))
  } catch (error) {
    console.error('Failed to load checkpoints:', error)
  }
}

const downloadCheckpoint = (path: string) => {
  // Implementation depends on backend
  ElMessage.info(`下载: ${path}`)
}

const deleteCheckpoint = async (path: string) => {
  ElMessage.info('后端暂不支持删除模型')
}

// Helpers
const formatTime = (timestamp: string): string => {
  return new Date(timestamp).toLocaleString('zh-CN')
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const getLogTagType = (level: LogEntry['level']): string => {
  switch (level) {
    case 'error': return 'danger'
    case 'warning': return 'warning'
    case 'success': return 'success'
    default: return 'info'
  }
}

// Auto scroll logs
watch(() => trainStore.logs.length, () => {
  if (autoScroll.value && logContainerRef.value) {
    nextTick(() => {
      logContainerRef.value!.scrollTop = logContainerRef.value!.scrollHeight
    })
  }
})

// Poll training status
const pollStatus = async () => {
  if (!trainStore.isRunning) return
  
  try {
    const response = await api.train.status()
    const status = response.data || {}
    const currentEpoch = status.current_epoch || 0
    const currentStep = status.current_step || 0
    const totalEpochs = status.total_epochs || currentEpoch
    const totalSteps = status.total_steps || currentStep
    const eta = status.eta || ''

    if (status.status) {
      trainStore.status = status.status
      trainStore.isRunning = Boolean(status.running)
      trainStore.isPaused = status.status === 'paused'
    }

    trainStore.updateProgress(
      currentEpoch,
      totalEpochs,
      currentStep,
      totalSteps,
      eta
    )
  } catch (error) {
    console.error('Failed to poll status:', error)
  }
}

// Lifecycle
onMounted(() => {
  initCharts()
  refreshCheckpoints()
  
  // Start polling
  pollTimer = setInterval(pollStatus, 2000)
  
  // Handle resize
  window.addEventListener('resize', () => {
    lossChart?.resize()
    lrChart?.resize()
  })
})

onUnmounted(() => {
  clearInterval(pollTimer)
  lossChart?.dispose()
  lrChart?.dispose()
})
</script>

<style scoped>
.train-view {
  max-width: 1400px;
  margin: 0 auto;
}

.status-bar {
  background: var(--el-bg-color);
}

.status-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.status-label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.progress-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.progress-text {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.action-buttons {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-container {
  height: 300px;
  width: 100%;
}

.log-container {
  max-height: 400px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 12px;
  background: var(--el-fill-color-light);
  padding: 12px;
  border-radius: 4px;
}

.log-line {
  display: flex;
  gap: 12px;
  padding: 4px 0;
  border-bottom: 1px solid var(--el-border-color-lighter);
}

.log-line:last-child {
  border-bottom: none;
}

.log-time {
  color: var(--el-text-color-secondary);
  min-width: 140px;
}

.log-level {
  min-width: 60px;
  text-align: center;
}

.log-message {
  flex: 1;
  word-break: break-all;
}

.log-debug {
  color: var(--el-text-color-secondary);
}

.log-info {
  color: var(--el-text-color-primary);
}

.log-warning {
  color: var(--el-color-warning);
}

.log-error {
  color: var(--el-color-danger);
}

.log-success {
  color: var(--el-color-success);
}

.log-actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

.mt-4 {
  margin-top: 16px;
}
</style>
