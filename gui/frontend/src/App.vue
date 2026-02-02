<template>
  <div class="app-container">
    <el-container>
      <!-- Sidebar -->
      <el-aside width="220px" class="sidebar">
        <div class="logo">
          <el-icon :size="32"><Cpu /></el-icon>
          <span>Anima Trainer</span>
        </div>
        
        <el-menu
          :default-active="activeRoute"
          router
          class="nav-menu"
          :collapse="false"
        >
          <el-menu-item index="/">
            <el-icon><HomeFilled /></el-icon>
            <span>首页</span>
          </el-menu-item>
          
          <el-menu-item index="/config">
            <el-icon><Setting /></el-icon>
            <span>训练配置</span>
          </el-menu-item>
          
          <el-menu-item index="/train">
            <el-icon><VideoPlay /></el-icon>
            <span>训练监控</span>
            <el-tag v-if="trainStore.isRunning" type="success" size="small" effect="dark" class="ml-2">
              运行中
            </el-tag>
          </el-menu-item>
          
          <el-menu-item index="/checkpoints">
            <el-icon><FolderChecked /></el-icon>
            <span>模型管理</span>
          </el-menu-item>
        </el-menu>
        
        <div class="sidebar-footer">
          <el-button
            :icon="themeStore.isDark ? Moon : Sunny"
            @click="themeStore.toggleTheme"
            text
            class="theme-btn"
          >
            {{ themeStore.themeLabel }}
          </el-button>
        </div>
      </el-aside>

      <!-- Main Content -->
      <el-container class="main-container">
        <el-header class="header">
          <div class="header-left">
            <breadcrumb />
          </div>
          <div class="header-right">
            <el-tag v-if="trainStore.isRunning" :type="trainStore.statusType" effect="dark">
              {{ trainStore.statusText }} - Epoch {{ trainStore.currentEpoch }}/{{ trainStore.totalEpochs }}
            </el-tag>
            <el-divider direction="vertical" />
            <el-text type="info">
              <el-icon><Clock /></el-icon>
              {{ currentTime }}
            </el-text>
          </div>
        </el-header>

        <el-main class="main-content">
          <router-view v-slot="{ Component }">
            <transition name="fade" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useThemeStore } from '@/stores/theme'
import { useTrainStore } from '@/stores/train'
import {
  HomeFilled,
  Setting,
  VideoPlay,
  FolderChecked,
  Cpu,
  Clock,
  Moon,
  Sunny
} from '@element-plus/icons-vue'

const route = useRoute()
const themeStore = useThemeStore()
const trainStore = useTrainStore()

const activeRoute = computed(() => route.path)
const currentTime = ref('')

let timer: ReturnType<typeof setInterval>

const updateTime = () => {
  currentTime.value = new Date().toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

onMounted(() => {
  themeStore.init()
  updateTime()
  timer = setInterval(updateTime, 1000)
})

onUnmounted(() => {
  clearInterval(timer)
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
}

.sidebar {
  background: var(--el-bg-color);
  border-right: 1px solid var(--el-border-color);
  display: flex;
  flex-direction: column;
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 100;
}

.logo {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 0 20px;
  border-bottom: 1px solid var(--el-border-color);
  font-size: 18px;
  font-weight: 600;
  color: var(--el-color-primary);
}

.nav-menu {
  flex: 1;
  border-right: none;
  padding: 10px 0;
}

.sidebar-footer {
  padding: 20px;
  border-top: 1px solid var(--el-border-color);
}

.theme-btn {
  width: 100%;
  justify-content: flex-start;
}

.main-container {
  margin-left: 220px;
  min-height: 100vh;
}

.header {
  background: var(--el-bg-color);
  border-bottom: 1px solid var(--el-border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 60px;
  position: sticky;
  top: 0;
  z-index: 99;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.main-content {
  padding: 20px;
  background: var(--el-fill-color-light);
  min-height: calc(100vh - 60px);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

:global(html) {
  color-scheme: light;
}

:global(html.dark) {
  color-scheme: dark;
}
</style>
