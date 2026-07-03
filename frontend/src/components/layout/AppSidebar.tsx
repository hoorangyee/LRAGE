import { GitCompareArrows, List, Moon, Plus, Scale, Sun } from "lucide-react"
import { NavLink } from "react-router-dom"

import { useTheme } from "@/hooks/useTheme"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

const NAV = [
  { to: "/new", label: "New run", icon: Plus },
  { to: "/runs", label: "Runs", icon: List },
  { to: "/compare", label: "Compare", icon: GitCompareArrows },
]

export function AppSidebar() {
  const { theme, toggle } = useTheme()

  return (
    <aside className="flex w-[220px] shrink-0 flex-col border-r bg-card">
      <div className="flex items-center gap-2 px-4 py-4">
        <Scale className="size-4 text-brass" aria-hidden />
        <span className="font-serif text-base font-semibold tracking-tight">
          LRAGE
        </span>
      </div>

      <nav className="flex flex-1 flex-col gap-0.5 px-2">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] text-muted-foreground transition-colors",
                "hover:bg-accent hover:text-accent-foreground",
                isActive && "bg-accent font-medium text-brass"
              )
            }
          >
            <Icon className="size-3.5" aria-hidden />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t px-2 py-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggle}
          className="w-full justify-start gap-2.5 px-2.5 text-[13px] text-muted-foreground"
        >
          {theme === "dark" ? (
            <Sun className="size-3.5" aria-hidden />
          ) : (
            <Moon className="size-3.5" aria-hidden />
          )}
          {theme === "dark" ? "Light mode" : "Dark mode"}
        </Button>
      </div>
    </aside>
  )
}
