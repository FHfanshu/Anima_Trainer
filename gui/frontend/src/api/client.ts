import axios, { type AxiosInstance, type AxiosResponse } from 'axios'
import { ElMessage } from 'element-plus'

// Create axios instance
// Single-port mode: API served from same origin, use relative path
const apiClient: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if needed
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response.data
  },
  (error) => {
    const message = error.response?.data?.detail || error.message || '请求失败'
    ElMessage.error(message)
    return Promise.reject(error)
  }
)

// API methods
export const api = {
  // Config APIs
  config: {
    getDefault: () => apiClient.get('/config/default'),
    presets: () => apiClient.get('/config/presets'),
    loadPreset: (name: string) => apiClient.post(`/config/load_preset/${name}`),
    save: (data: any) => apiClient.post('/config/save', data),
    list: () => apiClient.get('/config/list'),
    load: (name: string) => apiClient.get(`/config/load/${name}`),
    delete: (name: string) => apiClient.delete(`/config/delete/${name}`)
  },

  // Training APIs
  train: {
    start: (config: any) => apiClient.post('/train/start', { config }),
    stop: () => apiClient.post('/train/stop'),
    status: () => apiClient.get('/train/status'),
    logs: (limit?: number) => apiClient.get('/train/logs', { params: { lines: limit } }),
    metrics: () => apiClient.get('/train/metrics')
  },

  // Checkpoint APIs
  checkpoint: {
    list: () => apiClient.get('/train/checkpoints')
  },

  // System APIs
  system: {
    info: () => apiClient.get('/system/info'),
    gpuStatus: () => apiClient.get('/system/gpu_status'),
    pickFolder: () => apiClient.get('/system/pick_folder'),
    pickFile: (fileType?: string) => apiClient.get('/system/pick_file', { params: { file_type: fileType } }),
    listDirectory: (path?: string) => apiClient.get('/system/list_directory', { params: { path } })
  }
}

export default apiClient
