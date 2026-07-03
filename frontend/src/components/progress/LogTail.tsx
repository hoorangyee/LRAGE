import { useEffect, useRef, useState } from "react"
import { ArrowDown } from "lucide-react"

import type { RunLogLine } from "@/api/stream"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface LogTailProps {
  lines: RunLogLine[]
  className?: string
}

export function LogTail({ lines, className }: LogTailProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [pinned, setPinned] = useState(true)

  useEffect(() => {
    if (pinned && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [lines, pinned])

  const onScroll = () => {
    const el = containerRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24
    setPinned(atBottom)
  }

  return (
    <div className={cn("relative", className)}>
      <div
        ref={containerRef}
        onScroll={onScroll}
        className="h-64 overflow-y-auto rounded-lg border bg-card px-3 py-2"
      >
        {lines.length === 0 ? (
          <p className="py-2 font-mono text-[11px] text-muted-foreground">
            Waiting for output…
          </p>
        ) : (
          lines.map((line) => (
            <div
              key={line.id}
              className="flex gap-2 font-mono text-[11px] leading-relaxed"
            >
              <span
                className={cn(
                  "shrink-0 w-14",
                  line.level === "ERROR" || line.level === "CRITICAL"
                    ? "text-status-err"
                    : line.level === "WARNING"
                      ? "text-status-warn"
                      : "text-muted-foreground/60"
                )}
              >
                {line.level}
              </span>
              <span className="whitespace-pre-wrap break-all text-foreground/90">
                {line.message}
              </span>
            </div>
          ))
        )}
      </div>
      {!pinned && (
        <Button
          size="sm"
          variant="secondary"
          className="absolute bottom-3 right-3 gap-1 shadow"
          onClick={() => setPinned(true)}
        >
          <ArrowDown className="size-3" /> Follow
        </Button>
      )}
    </div>
  )
}
