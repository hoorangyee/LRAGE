import { useMemo, useState } from "react"
import { Check, ChevronsUpDown, X } from "lucide-react"

import type { TaskMeta } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { cn } from "@/lib/utils"

const VISIBLE_LIMIT = 100

interface TaskMultiSelectProps {
  tasks: TaskMeta[]
  value: string[]
  onChange: (tasks: string[]) => void
  loading?: boolean
}

export function TaskMultiSelect({
  tasks,
  value,
  onChange,
  loading,
}: TaskMultiSelectProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const matches = q
      ? tasks.filter((t) => t.name.toLowerCase().includes(q))
      : tasks
    return {
      groups: matches.filter((t) => t.type === "group").slice(0, VISIBLE_LIMIT),
      tasks: matches.filter((t) => t.type !== "group").slice(0, VISIBLE_LIMIT),
      truncated: matches.length > VISIBLE_LIMIT * 2,
      total: matches.length,
    }
  }, [tasks, query])

  const toggle = (name: string) => {
    onChange(
      value.includes(name) ? value.filter((v) => v !== name) : [...value, name]
    )
  }

  return (
    <div className="space-y-2">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between font-normal"
          >
            <span className="text-muted-foreground">
              {loading
                ? "Loading tasks…"
                : value.length
                  ? `${value.length} selected`
                  : "Search benchmarks and tasks…"}
            </span>
            <ChevronsUpDown className="size-3.5 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[480px] p-0" align="start">
          <Command shouldFilter={false}>
            <CommandInput
              placeholder="Type to filter 2,500+ tasks…"
              value={query}
              onValueChange={setQuery}
            />
            <CommandList>
              <CommandEmpty>No tasks match.</CommandEmpty>
              {filtered.groups.length > 0 && (
                <CommandGroup heading="Benchmark groups">
                  {filtered.groups.map((t) => (
                    <TaskItem
                      key={t.name}
                      name={t.name}
                      selected={value.includes(t.name)}
                      onSelect={() => toggle(t.name)}
                    />
                  ))}
                </CommandGroup>
              )}
              {filtered.tasks.length > 0 && (
                <CommandGroup heading="Tasks">
                  {filtered.tasks.map((t) => (
                    <TaskItem
                      key={t.name}
                      name={t.name}
                      selected={value.includes(t.name)}
                      onSelect={() => toggle(t.name)}
                    />
                  ))}
                </CommandGroup>
              )}
              {filtered.truncated && (
                <p className="px-3 py-2 text-xs text-muted-foreground">
                  Showing {VISIBLE_LIMIT * 2} of {filtered.total} matches —
                  keep typing to narrow down.
                </p>
              )}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {value.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {value.map((name) => (
            <Badge key={name} variant="secondary" className="gap-1 pr-1">
              <span className="font-mono text-[11px]">{name}</span>
              <button
                type="button"
                aria-label={`Remove ${name}`}
                onClick={() => toggle(name)}
                className="rounded-sm p-0.5 hover:bg-muted-foreground/20"
              >
                <X className="size-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}

function TaskItem({
  name,
  selected,
  onSelect,
}: {
  name: string
  selected: boolean
  onSelect: () => void
}) {
  return (
    <CommandItem value={name} onSelect={onSelect}>
      <Check
        className={cn("size-3.5", selected ? "opacity-100" : "opacity-0")}
      />
      <span className="font-mono text-xs">{name}</span>
    </CommandItem>
  )
}
