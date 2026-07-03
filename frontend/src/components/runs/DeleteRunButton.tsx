import { useState } from "react"
import { Trash2 } from "lucide-react"
import { useNavigate } from "react-router-dom"

import { useDeleteRun } from "@/api/mutations"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

export function DeleteRunButton({ runId }: { runId: string }) {
  const [open, setOpen] = useState(false)
  const deleteRun = useDeleteRun()
  const navigate = useNavigate()

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        aria-label="Delete run"
        className="size-8 text-muted-foreground hover:text-status-err"
        onClick={() => setOpen(true)}
      >
        <Trash2 className="size-3.5" />
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this run?</DialogTitle>
            <DialogDescription>
              Removes the run from the history. Result files on disk are kept.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Keep run
            </Button>
            <Button
              variant="destructive"
              disabled={deleteRun.isPending}
              onClick={() =>
                deleteRun.mutate(runId, {
                  onSuccess: () => navigate("/runs"),
                  onSettled: () => setOpen(false),
                })
              }
            >
              Delete run
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
