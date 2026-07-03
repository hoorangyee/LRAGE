import { useState } from "react"
import { Square } from "lucide-react"

import { useCancelRun } from "@/api/mutations"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import type { RunStatus } from "@/api/types"

interface CancelRunButtonProps {
  runId: string
  status: RunStatus
}

export function CancelRunButton({ runId, status }: CancelRunButtonProps) {
  const [open, setOpen] = useState(false)
  const cancelRun = useCancelRun()

  if (status === "cancelling") {
    return (
      <Button variant="outline" size="sm" disabled>
        Cancelling…
      </Button>
    )
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        className="gap-1.5 text-status-err hover:text-status-err"
        onClick={() => setOpen(true)}
      >
        <Square className="size-3" /> Cancel run
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Cancel this run?</DialogTitle>
            <DialogDescription>
              {status === "queued"
                ? "The run is still queued and will be removed immediately."
                : "Cancellation takes effect at the next batch boundary — a batch already on the GPU or an in-flight API call finishes first. Partial results are not saved."}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Keep running
            </Button>
            <Button
              variant="destructive"
              disabled={cancelRun.isPending}
              onClick={() =>
                cancelRun.mutate(runId, { onSettled: () => setOpen(false) })
              }
            >
              Cancel run
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
