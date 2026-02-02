<template>
  <div class="checkpoint-view">
    <el-page-header @back="$router.push('/')" title="模型管理" />

    <!-- Statistics -->
    <el-row :gutter="20" class="mt-4">
      <el-col :xs="12" :md="6">
        <el-card class="stat-card">
          <el-statistic title="总模型数" :value="checkpoints.length" />
        </el-card>
      </el-col>
      <el-col :xs="12" :md="6">
        <el-card class="stat-card">
          <el-statistic title="总大小" :value="formatBytes(totalSize)" />
        </el-card>
      </el-col>
      <el-col :xs="12" :md="6">
        <el-card class="stat-card">
          <el-statistic title="最新模型" :value="latestCheckpoint?.name || '无'" />
        </el-card>
      </el-col>
      <el-col :xs="12" :md="6">
        <el-card class="stat-card">
          <el-statistic title="最低 Loss" :value="minLoss" :precision="4" />
        </el-card>
      </el-col>
    </el-row>

    <!-- Toolbar -->
    <el-card class="toolbar mt-4">
      <el-row :gutter="20" align="middle">
        <el-col :xs="24" :md="12">
          <el-input
            v-model="searchQuery"
            placeholder="搜索模型..."
            clearable
            prefix-icon="Search"
          />
        </el-col>
        <el-col :xs="24" :md="12" class="toolbar-actions">
          <el-button-group>
            <el-button @click="refreshList" :loading="isLoading">
              <el-icon><Refresh /></el-icon>
              刷新
            </el-button>
            <el-button @click="notifyUnsupported('上传')">
              <el-icon><Upload /></el-icon>
              上传
            </el-button>
          </el-button-group>
          <el-button-group>
            <el-button @click="selectAll" :disabled="filteredCheckpoints.length === 0">
              全选
            </el-button>
            <el-button @click="clearSelection" :disabled="selectedCheckpoints.length === 0">
              取消选择
            </el-button>
          </el-button-group>
          <el-button 
            type="danger" 
            @click="batchDelete" 
            :disabled="selectedCheckpoints.length === 0"
          >
            <el-icon><Delete /></el-icon>
            删除选中 ({{ selectedCheckpoints.length }})
          </el-button>
        </el-col>
      </el-row>
    </el-card>

    <!-- Checkpoints Table -->
    <el-card class="mt-4">
      <el-table
        :data="filteredCheckpoints"
        style="width: 100%"
        v-loading="isLoading"
        @selection-change="handleSelectionChange"
        ref="tableRef"
      >
        <el-table-column type="selection" width="55" />
        
        <el-table-column label="名称" min-width="250" sortable prop="name">
          <template #default="{ row }">
            <div class="checkpoint-name">
              <el-icon class="file-icon"><Document /></el-icon>
              <div class="name-info">
                <span class="name-text">{{ row.name }}</span>
                <span class="path-text">{{ row.path }}</span>
              </div>
            </div>
          </template>
        </el-table-column>
        
        <el-table-column label="Epoch" width="90" sortable prop="epoch" />
        
        <el-table-column label="Step" width="100" sortable prop="step" />
        
        <el-table-column label="Loss" width="120" sortable prop="loss">
          <template #default="{ row }">
            <el-tag v-if="row.loss" :type="getLossTagType(row.loss)" size="small">
              {{ row.loss.toFixed(4) }}
            </el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        
        <el-table-column label="大小" width="100" sortable prop="size">
          <template #default="{ row }">
            {{ formatBytes(row.size) }}
          </template>
        </el-table-column>
        
        <el-table-column label="创建时间" width="180" sortable prop="created">
          <template #default="{ row }">
            {{ formatTime(row.created) }}
          </template>
        </el-table-column>
        
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button-group>
              <el-button text @click="previewCheckpoint(row)">
                <el-icon><View /></el-icon>
              </el-button>
              <el-button text @click="renameCheckpoint(row)">
                <el-icon><Edit /></el-icon>
              </el-button>
              <el-button text @click="downloadCheckpoint(row)">
                <el-icon><Download /></el-icon>
              </el-button>
              <el-button text type="danger" @click="deleteCheckpoint(row)">
                <el-icon><Delete /></el-icon>
              </el-button>
            </el-button-group>
          </template>
        </el-table-column>
      </el-table>
      
      <el-pagination
        v-if="filteredCheckpoints.length > pageSize"
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="filteredCheckpoints.length"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next"
        class="pagination"
      />
    </el-card>

    <!-- Upload Dialog -->
    <el-dialog
      v-model="showUploadDialog"
      title="上传模型"
      width="500px"
    >
      <el-upload
        drag
        action=""
        :auto-upload="false"
        :disabled="true"
        :on-progress="handleUploadProgress"
        :on-success="handleUploadSuccess"
        :on-error="handleUploadError"
        :before-upload="beforeUpload"
        accept=".safetensors,.ckpt,.pt,.pth"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          拖拽文件到此处或 <em>点击上传</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持 .safetensors, .ckpt, .pt, .pth 格式，单个文件不超过 2GB
          </div>
        </template>
      </el-upload>
    </el-dialog>

    <!-- Rename Dialog -->
    <el-dialog
      v-model="showRenameDialog"
      title="重命名模型"
      width="400px"
    >
      <el-form>
        <el-form-item label="新名称">
          <el-input v-model="renameForm.newName" placeholder="输入新名称" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showRenameDialog = false">取消</el-button>
        <el-button type="primary" @click="confirmRename" :loading="isRenaming">
          确定
        </el-button>
      </template>
    </el-dialog>

    <!-- Preview Dialog -->
    <el-dialog
      v-model="showPreviewDialog"
      title="模型详情"
      width="600px"
    >
      <el-descriptions :column="1" border v-if="previewData">
        <el-descriptions-item label="名称">{{ previewData.name }}</el-descriptions-item>
        <el-descriptions-item label="路径">{{ previewData.path }}</el-descriptions-item>
        <el-descriptions-item label="Epoch">{{ previewData.epoch }}</el-descriptions-item>
        <el-descriptions-item label="Step">{{ previewData.step }}</el-descriptions-item>
        <el-descriptions-item label="Loss">{{ previewData.loss?.toFixed(4) || '-' }}</el-descriptions-item>
        <el-descriptions-item label="大小">{{ formatBytes(previewData.size) }}</el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ formatTime(previewData.created) }}</el-descriptions-item>
        <el-descriptions-item label="SHA256">
          <el-tag size="small">{{ previewData.hash || '未计算' }}</el-tag>
        </el-descriptions-item>
      </el-descriptions>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useTrainStore } from '@/stores/train'
import { api } from '@/api/client'
import type { CheckpointInfo } from '@/types'
import {
  Refresh,
  Upload,
  Delete,
  Search,
  Document,
  View,
  Edit,
  Download,
  UploadFilled
} from '@element-plus/icons-vue'

const trainStore = useTrainStore()
const tableRef = ref()

// State
const checkpoints = ref<CheckpointInfo[]>([])
const isLoading = ref(false)
const searchQuery = ref('')
const selectedCheckpoints = ref<CheckpointInfo[]>([])
const currentPage = ref(1)
const pageSize = ref(20)

// Dialog state
const showUploadDialog = ref(false)
const showRenameDialog = ref(false)
const showPreviewDialog = ref(false)
const renameForm = ref({ path: '', newName: '' })
const isRenaming = ref(false)
const previewData = ref<CheckpointInfo | null>(null)

// Computed
const filteredCheckpoints = computed(() => {
  let result = checkpoints.value
  
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(cp => 
      cp.name.toLowerCase().includes(query) ||
      cp.path.toLowerCase().includes(query)
    )
  }
  
  return result
})

const totalSize = computed(() => {
  return checkpoints.value.reduce((sum, cp) => sum + cp.size, 0)
})

const latestCheckpoint = computed(() => {
  if (checkpoints.value.length === 0) return null
  return [...checkpoints.value].sort((a, b) => 
    new Date(b.created).getTime() - new Date(a.created).getTime()
  )[0]
})

const minLoss = computed(() => {
  const losses = checkpoints.value
    .map(cp => cp.loss)
    .filter((l): l is number => l !== undefined)
  return losses.length > 0 ? Math.min(...losses) : 0
})

// Methods
const refreshList = async () => {
  isLoading.value = true
  try {
    const response = await api.checkpoint.list()
    const data = (response.data?.checkpoints || []).map((cp: any) => ({
      name: cp.name || '',
      path: cp.path || '',
      created: cp.created || '',
      size: cp.size || 0,
      epoch: cp.epoch || 0,
      step: cp.step || 0,
      loss: cp.loss
    }))
    checkpoints.value = data
    trainStore.checkpoints = data
    ElMessage.success('列表已刷新')
  } catch (error) {
    ElMessage.error('刷新列表失败')
  } finally {
    isLoading.value = false
  }
}

const handleSelectionChange = (selection: CheckpointInfo[]) => {
  selectedCheckpoints.value = selection
}

const selectAll = () => {
  tableRef.value?.toggleAllSelection()
}

const clearSelection = () => {
  tableRef.value?.clearSelection()
}

const batchDelete = async () => {
  notifyUnsupported('批量删除')
}

const deleteCheckpoint = async (checkpoint: CheckpointInfo) => {
  notifyUnsupported('删除')
}

const renameCheckpoint = (checkpoint: CheckpointInfo) => {
  notifyUnsupported('重命名')
}

const confirmRename = async () => {
  notifyUnsupported('重命名')
}

const previewCheckpoint = (checkpoint: CheckpointInfo) => {
  previewData.value = checkpoint
  showPreviewDialog.value = true
}

const downloadCheckpoint = (checkpoint: CheckpointInfo) => {
  notifyUnsupported('下载')
}

const notifyUnsupported = (action: string) => {
  ElMessage.info(`后端暂不支持${action}操作`)
}

// Upload handlers
const beforeUpload = (file: File) => {
  const maxSize = 2 * 1024 * 1024 * 1024 // 2GB
  if (file.size > maxSize) {
    ElMessage.error('文件大小不能超过 2GB')
    return false
  }
  return true
}

const handleUploadProgress = (event: any) => {
  console.log('Upload progress:', event.percent)
}

const handleUploadSuccess = () => {
  notifyUnsupported('上传')
}

const handleUploadError = () => {
  notifyUnsupported('上传')
}

// Helpers
const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const formatTime = (timestamp: string): string => {
  return new Date(timestamp).toLocaleString('zh-CN')
}

const getLossTagType = (loss: number): string => {
  if (loss < 0.1) return 'success'
  if (loss < 0.3) return 'warning'
  return 'info'
}

onMounted(() => {
  refreshList()
})
</script>

<style scoped>
.checkpoint-view {
  max-width: 1400px;
  margin: 0 auto;
}

.stat-card {
  text-align: center;
}

.toolbar {
  background: var(--el-bg-color);
}

.toolbar-actions {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  flex-wrap: wrap;
}

.checkpoint-name {
  display: flex;
  align-items: center;
  gap: 12px;
}

.file-icon {
  font-size: 24px;
  color: var(--el-color-primary);
}

.name-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.name-text {
  font-weight: 500;
}

.path-text {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.pagination {
  margin-top: 16px;
  justify-content: flex-end;
}

.mt-4 {
  margin-top: 16px;
}

:deep(.el-upload-dragger) {
  width: 100%;
}
</style>
