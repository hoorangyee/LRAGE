import { useController, useFormContext } from "react-hook-form"

import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { parseArgsString } from "@/lib/argsString"

interface PresetOption {
  label: string
  args: string
}

interface PresetArgsFieldProps {
  presetName: string
  argsName: string
  presets: PresetOption[]
  argsLabel?: string
  placeholder?: string
}

/**
 * A preset dropdown feeding an always-editable args string. Editing the
 * string flips the preset to "Custom" — the escape hatch is the field itself.
 */
export function PresetArgsField({
  presetName,
  argsName,
  presets,
  argsLabel = "Arguments",
  placeholder = "key=value,key=value",
}: PresetArgsFieldProps) {
  const { setValue } = useFormContext()
  const preset = useController({ name: presetName })
  const args = useController({ name: argsName })

  const pairs = parseArgsString(String(args.field.value ?? ""))
  const malformed = pairs === null

  return (
    <div className="space-y-2">
      <div className="grid gap-1.5">
        <Label className="text-xs text-muted-foreground">Preset</Label>
        <Select
          value={preset.field.value ?? "__custom__"}
          onValueChange={(v) => {
            if (v === "__custom__") {
              preset.field.onChange(null)
              return
            }
            preset.field.onChange(v)
            const found = presets.find((p) => p.label === v)
            if (found)
              setValue(argsName, found.args, {
                shouldValidate: true,
                shouldDirty: true,
              })
          }}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Pick a preset or write args below" />
          </SelectTrigger>
          <SelectContent>
            {presets.map((p) => (
              <SelectItem key={p.label} value={p.label}>
                {p.label}
              </SelectItem>
            ))}
            <SelectItem value="__custom__">Custom</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="grid gap-1.5">
        <Label className="text-xs text-muted-foreground">{argsLabel}</Label>
        <Input
          {...args.field}
          value={String(args.field.value ?? "")}
          onChange={(e) => {
            args.field.onChange(e)
            if (preset.field.value !== null) preset.field.onChange(null)
          }}
          placeholder={placeholder}
          className="font-mono text-xs"
          spellCheck={false}
        />
        {malformed ? (
          <p className="text-xs text-status-err">
            Must be key=value pairs separated by commas.
          </p>
        ) : args.fieldState.error ? (
          <p className="text-xs text-status-err">
            {args.fieldState.error.message}
          </p>
        ) : pairs && pairs.length > 0 ? (
          <p className="truncate text-[11px] text-muted-foreground">
            {pairs.map((p) => p.key).join(" · ")}
          </p>
        ) : null}
      </div>
    </div>
  )
}
