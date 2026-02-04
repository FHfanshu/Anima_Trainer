import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useThemeStore = defineStore('theme', () => {
  // State
  const isDark = ref(localStorage.getItem('theme') === 'dark')

  // Actions
  const toggleTheme = () => {
    isDark.value = !isDark.value
    localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
    updateTheme()
  }

  const setTheme = (dark: boolean) => {
    isDark.value = dark
    localStorage.setItem('theme', dark ? 'dark' : 'light')
    updateTheme()
  }

  const updateTheme = () => {
    const html = document.documentElement
    if (isDark.value) {
      html.classList.add('dark')
      html.style.colorScheme = 'dark'
    } else {
      html.classList.remove('dark')
      html.style.colorScheme = 'light'
    }
  }

  // Initialize theme on store creation
  const init = () => {
    updateTheme()
  }

  // Getters
  const themeIcon = computed(() => (isDark.value ? 'Moon' : 'Sunny'))
  const themeLabel = computed(() => (isDark.value ? '深色模式' : '浅色模式'))

  return {
    isDark,
    toggleTheme,
    setTheme,
    init,
    themeIcon,
    themeLabel
  }
})
