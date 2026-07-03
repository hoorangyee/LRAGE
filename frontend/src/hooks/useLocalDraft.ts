import { useEffect, useRef } from "react"
import type { UseFormReturn } from "react-hook-form"

/** Persist form values to localStorage (debounced) and restore on mount. */
export function useLocalDraft<T extends Record<string, unknown>>(
  key: string,
  form: UseFormReturn<T>,
  enabled = true
) {
  const restored = useRef(false)

  useEffect(() => {
    if (!enabled || restored.current) return
    restored.current = true
    try {
      const raw = localStorage.getItem(key)
      if (raw) form.reset(JSON.parse(raw), { keepDefaultValues: true })
    } catch {
      // ignore corrupt drafts
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled])

  useEffect(() => {
    if (!enabled) return
    let timer: ReturnType<typeof setTimeout> | undefined
    const sub = form.watch((values) => {
      clearTimeout(timer)
      timer = setTimeout(() => {
        try {
          localStorage.setItem(key, JSON.stringify(values))
        } catch {
          // storage full/unavailable — drafts are best-effort
        }
      }, 500)
    })
    return () => {
      clearTimeout(timer)
      sub.unsubscribe()
    }
  }, [key, form, enabled])

  return {
    clear: () => localStorage.removeItem(key),
  }
}
