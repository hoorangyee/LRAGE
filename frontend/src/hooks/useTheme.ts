import { useCallback, useSyncExternalStore } from "react"

const STORAGE_KEY = "lrage:theme"

type Theme = "dark" | "light"

let listeners: Array<() => void> = []

function getTheme(): Theme {
  return (localStorage.getItem(STORAGE_KEY) as Theme) || "dark"
}

export function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark")
}

export function initTheme() {
  applyTheme(getTheme())
}

export function useTheme() {
  const theme = useSyncExternalStore(
    (cb) => {
      listeners.push(cb)
      return () => {
        listeners = listeners.filter((l) => l !== cb)
      }
    },
    getTheme,
    () => "dark" as Theme
  )

  const toggle = useCallback(() => {
    const next: Theme = getTheme() === "dark" ? "light" : "dark"
    localStorage.setItem(STORAGE_KEY, next)
    applyTheme(next)
    listeners.forEach((l) => l())
  }, [])

  return { theme, toggle }
}
