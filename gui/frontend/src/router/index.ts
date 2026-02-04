import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/HomeView.vue'),
    meta: { title: '首页', icon: 'HomeFilled' }
  },
  {
    path: '/config',
    name: 'Config',
    component: () => import('@/views/ConfigView.vue'),
    meta: { title: '配置训练', icon: 'Setting' }
  },
  {
    path: '/train',
    name: 'Train',
    component: () => import('@/views/TrainView.vue'),
    meta: { title: '训练控制台', icon: 'VideoPlay' }
  },
  {
    path: '/checkpoints',
    name: 'Checkpoints',
    component: () => import('@/views/CheckpointView.vue'),
    meta: { title: '模型管理', icon: 'FolderOpened' }
  }
]

const router = createRouter({
  history: createWebHistory('./'),
  routes
})

export default router