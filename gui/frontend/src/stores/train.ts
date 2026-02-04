import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { TrainingStatus, TrainingMetrics, LogEntry, CheckpointInfo } from '@/types'

export const useTrainStore = defineStore('train', () => {
  // State
  const status = ref<TrainingStatus>('idle')
  const isRunning = ref(false)
  const isPaused = ref(false)
  const currentEpoch = ref(0)
  const totalEpochs = ref(0)
  const currentStep = ref(0)
  const totalSteps = ref(0)
  const progress = ref(0)
  const eta = ref<string>('')
  
  const metrics = ref<TrainingMetrics>({
    loss: [],
    learning_rate: [],
    epoch: [],
    step: [],
    timestamp: []
  })
  
  const logs = ref<LogEntry[]>([])
  const checkpoints = ref<CheckpointInfo[]>([])
  const currentConfig = ref<string>('')
  
  const startTime = ref<Date | null>(null)
  const endTime = ref<Date | null>(null)
  const error = ref<string | null>(null)

  // Actions
  const startTraining = () => {
    status.value = 'running'
    isRunning.value = true
    isPaused.value = false
    startTime.value = new Date()
    endTime.value = null
    error.value = null
    logs.value = []
    metrics.value = {
      loss: [],
      learning_rate: [],
      epoch: [],
      step: [],
      timestamp: []
    }
    addLog('info', '训练开始')
  }

  const pauseTraining = () => {
    status.value = 'paused'
    isPaused.value = true
    addLog('warning', '训练已暂停')
  }

  const resumeTraining = () => {
    status.value = 'running'
    isPaused.value = false
    addLog('info', '训练已恢复')
  }

  const stopTraining = () => {
    status.value = 'stopped'
    isRunning.value = false
    isPaused.value = false
    endTime.value = new Date()
    addLog('warning', '训练已停止')
  }

  const finishTraining = () => {
    status.value = 'completed'
    isRunning.value = false
    isPaused.value = false
    endTime.value = new Date()
    progress.value = 100
    addLog('success', '训练完成')
  }

  const setError = (msg: string) => {
    status.value = 'error'
    isRunning.value = false
    error.value = msg
    endTime.value = new Date()
    addLog('error', msg)
  }

  const updateProgress = (
    epoch: number,
    totalEpoch: number,
    step: number,
    totalStep: number,
    etaStr: string
  ) => {
    currentEpoch.value = epoch
    totalEpochs.value = totalEpoch
    currentStep.value = step
    totalSteps.value = totalStep
    eta.value = etaStr
    
    if (totalStep > 0) {
      progress.value = Math.round((step / totalStep) * 100)
    }
  }

  const addMetric = (
    loss: number,
    lr: number,
    epoch: number,
    step: number
  ) => {
    metrics.value.loss.push(loss)
    metrics.value.learning_rate.push(lr)
    metrics.value.epoch.push(epoch)
    metrics.value.step.push(step)
    metrics.value.timestamp.push(new Date().toISOString())
  }

  const addLog = (level: LogEntry['level'], message: string) => {
    logs.value.push({
      timestamp: new Date().toISOString(),
      level,
      message
    })
    
    // Keep only last 1000 logs
    if (logs.value.length > 1000) {
      logs.value = logs.value.slice(-1000)
    }
  }

  const clearLogs = () => {
    logs.value = []
  }

  const addCheckpoint = (checkpoint: CheckpointInfo) => {
    checkpoints.value.unshift(checkpoint)
  }

  const removeCheckpoint = (path: string) => {
    checkpoints.value = checkpoints.value.filter(c => c.path !== path)
  }

  const reset = () => {
    status.value = 'idle'
    isRunning.value = false
    isPaused.value = false
    currentEpoch.value = 0
    totalEpochs.value = 0
    currentStep.value = 0
    totalSteps.value = 0
    progress.value = 0
    eta.value = ''
    logs.value = []
    metrics.value = {
      loss: [],
      learning_rate: [],
      epoch: [],
      step: [],
      timestamp: []
    }
    checkpoints.value = []
    startTime.value = null
    endTime.value = null
    error.value = null
  }

  // Getters
  const recentLogs = computed(() => 
    logs.value.slice(-100)
  )

  const errorLogs = computed(() => 
    logs.value.filter(log => log.level === 'error')
  )

  const duration = computed(() => {
    if (!startTime.value) return ''
    const end = endTime.value || new Date()
    const diff = Math.floor((end.getTime() - startTime.value.getTime()) / 1000)
    const hours = Math.floor(diff / 3600)
    const minutes = Math.floor((diff % 3600) / 60)
    const seconds = diff % 60
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
  })

  const statusText = computed(() => {
    switch (status.value) {
      case 'idle': return '空闲'
      case 'running': return '训练中'
      case 'paused': return '已暂停'
      case 'stopped': return '已停止'
      case 'completed': return '已完成'
      case 'error': return '错误'
      default: return '未知'
    }
  })

  const statusType = computed(() => {
    switch (status.value) {
      case 'idle': return 'info'
      case 'running': return 'success'
      case 'paused': return 'warning'
      case 'stopped': return 'danger'
      case 'completed': return 'success'
      case 'error': return 'danger'
      default: return 'info'
    }
  })

  const canStart = computed(() => 
    status.value === 'idle' || status.value === 'stopped' || status.value === 'completed' || status.value === 'error'
  )

  const canPause = computed(() => status.value === 'running')
  const canResume = computed(() => status.value === 'paused')
  const canStop = computed(() => status.value === 'running' || status.value === 'paused')

  return {
    status,
    isRunning,
    isPaused,
    currentEpoch,
    totalEpochs,
    currentStep,
    totalSteps,
    progress,
    eta,
    metrics,
    logs,
    checkpoints,
    currentConfig,
    startTime,
    endTime,
    error,
    startTraining,
    pauseTraining,
    resumeTraining,
    stopTraining,
    finishTraining,
    setError,
    updateProgress,
    addMetric,
    addLog,
    clearLogs,
    addCheckpoint,
    removeCheckpoint,
    reset,
    recentLogs,
    errorLogs,
    duration,
    statusText,
    statusType,
    canStart,
    canPause,
    canResume,
    canStop
  }
})
