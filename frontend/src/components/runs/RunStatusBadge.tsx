import { Loader2 } from "lucide-react"

import type { RunStatus } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

const STYLES: Record<RunStatus, string> = {
  queued: "border-border text-muted-foreground",
  running: "border-brass/40 text-brass",
  cancelling: "border-status-warn/40 text-status-warn",
  completed: "border-status-ok/40 text-status-ok",
  failed: "border-status-err/40 text-status-err",
  cancelled: "border-border text-muted-foreground",
}

export function RunStatusBadge({ status }: { status: RunStatus }) {
  return (
    <Badge variant="outline" className={cn("gap-1 font-normal", STYLES[status])}>
      {(status === "running" || status === "cancelling") && (
        <Loader2 className="size-3 animate-spin" />
      )}
      {status}
    </Badge>
  )
}
